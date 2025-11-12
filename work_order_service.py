"""
Work Order Extraction API
-------------------------

This module implements a REST API using FastAPI and Pydantic to extract
and normalize work order information from uploaded audio and image files.
The service follows the specifications defined in the project
documentation, including the Work Order schema, QuickBooks Online (QBO)
integration considerations, and the outlined processing plan.

Two main endpoints are exposed:

- **POST `/work-orders/audio`** – Accepts an audio recording and optional
  hints (client name, site address, date override). The endpoint
  transcribes the audio with OpenAI's Whisper model, parses semantic
  information from the transcript with a GPT model, resolves client and
  item references, and returns a partial Work Order payload that can be
  merged with other records.

- **POST `/work-orders/picture`** – Accepts one or more image files and
  optional hints. Each image is sent to a GPT vision model to extract
  structured information. Results from multiple images are merged into a
  single Work Order object. Client and item references are resolved in
  the same way as the audio endpoint.

Both endpoints produce a JSON object matching the Work Order schema
defined in `Work Order.md`. Fields are populated when confidence is high;
otherwise they remain undefined and the response's `status` and
`status_detail` fields contain diagnostic information. Unmapped clients
or items are flagged with ``customerId`` or ``ItemRef.value`` set to
``"UNMAPPED"``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_TEXT_MODEL = "gpt-4o"
DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 800


@dataclass(frozen=True)
class Settings:
    """Runtime configuration sourced from the environment."""

    openai_api_key: str
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        text_model = os.getenv("OPENAI_TEXT_MODEL", DEFAULT_TEXT_MODEL)
        vision_model = os.getenv("OPENAI_VISION_MODEL", DEFAULT_VISION_MODEL)
        transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
        return cls(
            openai_api_key=api_key,
            text_model=text_model,
            vision_model=vision_model,
            transcription_model=transcription_model,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


ContentPayload = Union[str, Sequence[Dict[str, Any]]]


@dataclass
class OpenAIAdapter:
    """Thin wrapper over the OpenAI SDK to support multiple versions."""

    api_key: str
    text_model: str
    vision_model: str
    transcription_model: str
    temperature: float
    max_tokens: int

    def __post_init__(self) -> None:
        if hasattr(openai, "OpenAI"):
            self._client = openai.OpenAI(api_key=self.api_key)
            self._modern = True
        else:  # pragma: no cover
            openai.api_key = self.api_key
            self._client = openai
            self._modern = False

    def chat_completion(
        self,
        *,
        system_prompt: str,
        user_content: ContentPayload,
        model: Optional[str] = None,
    ) -> str:
        """Invoke the chat completion API and return the raw content string."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        target_model = model or self.text_model
        if self._modern:
            response = self._client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            message = response.choices[0].message
            content = getattr(message, "content", None)
            if content is None and isinstance(message, dict):  # pragma: no cover
                content = message.get("content")
        else:  # pragma: no cover
            response = self._client.ChatCompletion.create(
                model=target_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            choice = response["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")
        if not content:
            raise RuntimeError("Chat completion response did not include content.")
        return content

    def transcribe_audio(self, file_name: str, contents: bytes) -> str:
        """Invoke the Whisper transcription model and return the transcript."""
        audio_stream = io.BytesIO(contents)
        audio_stream.name = file_name or "audio.m4a"
        if self._modern and hasattr(self._client, "audio"):
            result = self._client.audio.transcriptions.create(
                model=self.transcription_model,
                file=audio_stream,
                response_format="text",
            )
            if isinstance(result, dict):  # pragma: no cover
                text = result.get("text")
            else:
                text = getattr(result, "text", None)
            return text or str(result)
        result = self._client.Audio.transcribe(  # pragma: no cover
            model=self.transcription_model,
            file=audio_stream,
            response_format="text",
        )
        if isinstance(result, dict):
            return result.get("text") or str(result)
        return result


@lru_cache(maxsize=1)
def get_openai_adapter() -> OpenAIAdapter:
    settings = get_settings()
    return OpenAIAdapter(
        api_key=settings.openai_api_key,
        text_model=settings.text_model,
        vision_model=settings.vision_model,
        transcription_model=settings.transcription_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

CODE_FENCE_PATTERN = re.compile(r"^```[\w]*\s*\n|\n```$", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences from a model response, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```[\w]*\s*\n", "", stripped)
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def parse_model_json(raw_content: str, context: str) -> Dict[str, Any]:
    """Parse JSON returned by the model, raising ValueError on failure."""
    cleaned = strip_code_fences(raw_content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:  # pragma: no cover - exceptional path
        preview = cleaned[:2000]
        logger.exception("Failed to decode JSON for %s: %s", context, preview)
        raise ValueError(f"Failed to parse JSON from model output for {context}: {exc}") from exc


def clone_model(model: BaseModel) -> BaseModel:
    """Deep-copy a Pydantic model, handling both v1 and v2 APIs."""
    if hasattr(model, "model_copy"):
        return model.model_copy(deep=True)  # type: ignore[attr-defined]
    return model.copy(deep=True)


def model_to_dict(model: Optional[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to a dictionary."""
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def normalise_work_order_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Cleanup loosely-structured model output to satisfy the schema."""
    if not isinstance(payload, dict):
        return {}
    data = deepcopy(payload)
    bill_to = data.get("Bill_to")
    if isinstance(bill_to, str):
        data["Bill_to"] = {"name": bill_to}
    elif isinstance(bill_to, dict):
        address = bill_to.get("address")
        if isinstance(address, str):
            bill_to["address"] = {"Line1": address}
        elif isinstance(address, list):
            bill_to["address"] = {"Line1": " ".join(str(part) for part in address)}
    description = data.get("Description")
    if isinstance(description, str):
        data["Description"] = [description]
    elif isinstance(description, list):
        data["Description"] = [str(entry) for entry in description if entry is not None]
    lines = data.get("Line")
    if isinstance(lines, dict):
        lines = [lines]
    normalised_lines: List[Dict[str, Any]] = []
    if isinstance(lines, list):
        for entry in lines:
            if not isinstance(entry, dict):
                entry = {
                    "Amount": 0.0,
                    "Description": str(entry),
                    "SalesItemLineDetail": {"ItemRef": {"value": "UNMAPPED", "name": str(entry)}},
                }
            detail = entry.get("SalesItemLineDetail")
            if isinstance(detail, dict):
                item_ref = detail.get("ItemRef")
                if isinstance(item_ref, str):
                    detail["ItemRef"] = {"value": item_ref}
            normalised_lines.append(entry)
        data["Line"] = normalised_lines
    return data


def work_order_from_payload(payload: Union["WorkOrder", Dict[str, Any]]) -> "WorkOrder":
    """Normalise a payload into a WorkOrder instance."""
    if isinstance(payload, WorkOrder):
        return payload
    if hasattr(WorkOrder, "model_validate"):
        clean_payload = normalise_work_order_payload(payload)  # type: ignore[arg-type]
        return WorkOrder.model_validate(clean_payload)  # type: ignore[attr-defined]
    clean_payload = normalise_work_order_payload(payload)  # type: ignore[arg-type]
    return WorkOrder.parse_obj(clean_payload)


def dedup_key(item: Any) -> str:
    """Generate a stable key for deduplicating list entries."""
    if isinstance(item, BaseModel):
        serialisable = model_to_dict(item)
    else:
        serialisable = item
    try:
        return json.dumps(serialisable, sort_keys=True, default=str)
    except TypeError:  # pragma: no cover
        return repr(serialisable)


IMAGE_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def build_data_url(filename: Optional[str], contents: bytes) -> str:
    """Encode an image as a data URL suitable for the vision model."""
    extension = os.path.splitext((filename or "").lower())[1]
    content_type = IMAGE_CONTENT_TYPES.get(extension, "application/octet-stream")
    b64 = base64.b64encode(contents).decode("ascii")
    return f"data:{content_type};base64,{b64}"


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


class Address(BaseModel):
    """Represents a postal address as per the Work Order schema."""

    Line1: Optional[str] = None
    City: Optional[str] = None
    CountrySubDivisionCode: Optional[str] = None
    PostalCode: Optional[str] = None
    Country: Optional[str] = None


class BillTo(BaseModel):
    """Customer and billing information."""

    customerId: Optional[str] = None
    name: Optional[str] = None
    address: Optional[Address] = None


class ItemRef(BaseModel):
    """Reference to a QuickBooks Item."""

    value: str
    name: Optional[str] = None


class SalesItemLineDetail(BaseModel):
    """Detail describing a line item in the Work Order."""

    ItemRef: ItemRef


class LineItem(BaseModel):
    """A single line item on the Work Order."""

    Amount: float
    Description: str
    DetailType: str = "SalesItemLineDetail"
    SalesItemLineDetail: SalesItemLineDetail

    @validator("Amount")
    def amount_must_be_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Amount must be non-negative")
        return value


class WorkOrder(BaseModel):
    """Represents a partial or complete Work Order."""

    Date: Optional[str] = None
    Bill_to: Optional[BillTo] = None
    Description: Optional[List[str]] = None
    Line: Optional[List[LineItem]] = None
    Audio: Optional[List[str]] = None
    Picture: Optional[List[str]] = None
    status: Optional[str] = None
    status_detail: Optional[str] = None
    invoice_id: Optional[str] = None

    def merge(self, other: "WorkOrder") -> "WorkOrder":
        """Merge another WorkOrder into this one, preferring existing values."""
        merged = clone_model(self)
        for field in ["Date", "Bill_to", "status", "status_detail", "invoice_id"]:
            if getattr(merged, field) is None:
                setattr(merged, field, getattr(other, field))
        for list_field in ["Description", "Line", "Audio", "Picture"]:
            existing_items = getattr(merged, list_field) or []
            incoming_items = getattr(other, list_field) or []
            combined: List[Any] = []
            seen: set[str] = set()
            for item in list(existing_items) + list(incoming_items):
                key = dedup_key(item)
                if key in seen:
                    continue
                seen.add(key)
                combined.append(item)
            setattr(merged, list_field, combined or None)
        return merged


class AudioRequest(BaseModel):
    """Query parameters for the audio endpoint."""

    client_name: Optional[str] = Field(None, description="Hint for resolving the client")
    site_address: Optional[str] = Field(None, description="Hint for resolving the address")
    date_override: Optional[str] = Field(None, description="Override for the work order date (ISO format)")


class PictureRequest(BaseModel):
    """Query parameters for the picture endpoint."""

    client_name: Optional[str] = Field(None, description="Hint for resolving the client")
    site_address: Optional[str] = Field(None, description="Hint for resolving the address")
    reference_wo_id: Optional[str] = Field(None, description="Existing work order ID to merge with")


# -----------------------------------------------------------------------------
# Mock Data and Resolvers
# -----------------------------------------------------------------------------


CLIENTS: Dict[str, Dict[str, Any]] = {
    "maple towers": {
        "customerId": "123",
        "name": "Maple Towers",
        "address": Address(
            Line1="123 Maple Street",
            City="Toronto",
            CountrySubDivisionCode="ON",
            PostalCode="M5V 2T6",
            Country="Canada",
        ),
    },
    "john doe": {
        "customerId": "124",
        "name": "John Doe",
        "address": Address(
            Line1="456 Oak Avenue",
            City="Toronto",
            CountrySubDivisionCode="ON",
            PostalCode="M5V 1S8",
            Country="Canada",
        ),
    },
}


def resolve_client(name: Optional[str]) -> Optional[BillTo]:
    """Resolve a client name to a BillTo object."""
    if not name:
        return None
    key = name.strip().lower()
    client_info = CLIENTS.get(key)
    if not client_info:
        return BillTo(customerId="UNMAPPED", name=name)
    return BillTo(
        customerId=client_info["customerId"],
        name=client_info["name"],
        address=client_info.get("address"),
    )


ITEMS: Dict[str, Dict[str, Any]] = {
    "plumbing labour": {"value": "79", "name": "Plumbing Labour"},
    "labour": {"value": "79", "name": "Plumbing Labour"},
    "diagnostic": {"value": "80", "name": "Diagnostic Service"},
    "pex pipe": {"value": "145", "name": "PEX Pipe"},
    "p-trap assembly": {"value": "146", "name": "P-trap Assembly"},
    "copper pipe": {"value": "154", "name": "Copper Pipe"},
}


def resolve_item(name: Optional[str]) -> ItemRef:
    """Resolve a material or service description to an ItemRef."""
    if not name:
        return ItemRef(value="UNMAPPED", name="")
    key = name.strip().lower()
    for candidate, ref in ITEMS.items():
        if candidate in key:
            return ItemRef(value=ref["value"], name=ref["name"])
    return ItemRef(value="UNMAPPED", name=name)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


AUDIO_SYSTEM_PROMPT = (
    "You are an AI assistant helping to extract structured work order data "
    "from a plumber's audio note. Return only a JSON object with the "
    "following fields: Date (YYYY-MM-DD), Bill_to (with customerId, name, "
    "address), Description (list of strings), Line (list of line items with "
    "Amount, Description, DetailType='SalesItemLineDetail', "
    "SalesItemLineDetail with ItemRef {value,name}), status, status_detail. "
    "Do not return any explanatory text. If a field is not present in the "
    "transcript, omit it. Use any hints provided to resolve ambiguity."
)


def build_audio_prompt(transcript: str, hints: AudioRequest) -> str:
    """Construct the user prompt sent to the text model."""
    hint_lines: List[str] = []
    if hints.client_name:
        hint_lines.append(f"Client name hint: {hints.client_name}")
    if hints.site_address:
        hint_lines.append(f"Site address hint: {hints.site_address}")
    if hints.date_override:
        hint_lines.append(f"Date override hint: {hints.date_override}")
    prompt_parts: List[str] = []
    if hint_lines:
        prompt_parts.append("\n".join(hint_lines))
    prompt_parts.append("Transcript:")
    prompt_parts.append(transcript)
    return "\n".join(prompt_parts)


IMAGE_SYSTEM_PROMPT = (
    "You are an AI assistant extracting structured work order data from "
    "photos of handwritten or printed forms, receipts or notes. Return only "
    "a JSON object with the following fields: Date (YYYY-MM-DD), Bill_to "
    "(with customerId, name, address), Description (list of strings), Line "
    "(list of line items with Amount, Description, DetailType='SalesItemLineDetail', "
    "SalesItemLineDetail with ItemRef {value,name}), status, status_detail. "
    "If a field is not present in the image, omit it."
)


def build_image_user_content(data_url: str, hints: PictureRequest) -> List[Dict[str, Any]]:
    """Construct the multi-part user message for the vision model."""
    hint_lines: List[str] = []
    if hints.client_name:
        hint_lines.append(f"Client name hint: {hints.client_name}")
    if hints.site_address:
        hint_lines.append(f"Site address hint: {hints.site_address}")
    if hints.reference_wo_id:
        hint_lines.append(f"Existing work order ID: {hints.reference_wo_id}")
    content: List[Dict[str, Any]] = []
    if hint_lines:
        content.append({"type": "text", "text": "\n".join(hint_lines)})
    content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


async def transcribe_audio(file: UploadFile) -> str:
    """Transcribe an audio file using OpenAI's Whisper model."""
    adapter = get_openai_adapter()
    contents = await file.read()
    return await run_in_threadpool(
        adapter.transcribe_audio,
        file.filename or "audio.m4a",
        contents,
    )


async def extract_work_order_from_transcript(transcript: str, hints: AudioRequest) -> Dict[str, Any]:
    """Extract structured work order data from a transcript using GPT."""
    adapter = get_openai_adapter()
    prompt = build_audio_prompt(transcript, hints)
    raw_json = await run_in_threadpool(
        adapter.chat_completion,
        system_prompt=AUDIO_SYSTEM_PROMPT,
        user_content=prompt,
        model=adapter.text_model,
    )
    return parse_model_json(raw_json, "audio transcript")


async def extract_work_order_from_images(files: List[UploadFile], hints: PictureRequest) -> Dict[str, Any]:
    """Extract structured work order data from one or more images."""
    adapter = get_openai_adapter()
    merged: Optional[WorkOrder] = None
    for upload in files:
        contents = await upload.read()
        data_url = build_data_url(upload.filename, contents)
        user_content = build_image_user_content(data_url, hints)
        raw_json = await run_in_threadpool(
            adapter.chat_completion,
            system_prompt=IMAGE_SYSTEM_PROMPT,
            user_content=user_content,
            model=adapter.vision_model,
        )
        payload = parse_model_json(raw_json, upload.filename or "image")
        partial = work_order_from_payload(payload)
        merged = partial if merged is None else merged.merge(partial)
    return model_to_dict(merged) if merged else {}


def postprocess_work_order(
    data: Dict[str, Any],
    hints: Optional[Union[AudioRequest, PictureRequest]] = None,
) -> WorkOrder:
    """Resolve clients and items in the extracted work order and set status flags."""
    wo = work_order_from_payload(data)
    hint_client_name = getattr(hints, "client_name", None) if hints else None
    if wo.Bill_to:
        if not wo.Bill_to.customerId or wo.Bill_to.customerId == "UNMAPPED":
            resolved = resolve_client(wo.Bill_to.name or hint_client_name)
            if resolved:
                wo.Bill_to = resolved
    elif hint_client_name:
        wo.Bill_to = resolve_client(hint_client_name)
    unresolved_items: List[str] = []
    if wo.Line:
        updated_lines: List[LineItem] = []
        for line in wo.Line:
            item_ref = resolve_item(line.Description)
            if item_ref.value == "UNMAPPED":
                unresolved_items.append(line.Description)
            cloned_line = cast(LineItem, clone_model(line))
            cloned_detail = cast(SalesItemLineDetail, clone_model(line.SalesItemLineDetail))
            cloned_line.SalesItemLineDetail = cloned_detail
            cloned_line.SalesItemLineDetail.ItemRef = item_ref
            updated_lines.append(cloned_line)
        wo.Line = updated_lines
    missing_fields: List[str] = []
    if not wo.Date:
        missing_fields.append("Date")
    if not wo.Bill_to or not wo.Bill_to.customerId:
        missing_fields.append("Bill_to.customerId")
    if not wo.Line:
        missing_fields.append("Line[]")
    wo.status = "ready" if not missing_fields and not unresolved_items else "pending_review"
    details: List[str] = []
    if missing_fields:
        details.append(f"Missing fields: {', '.join(missing_fields)}")
    if unresolved_items:
        details.append(f"Unmapped items: {', '.join(unresolved_items)}")
    wo.status_detail = "; ".join(details) if details else None
    return wo


# -----------------------------------------------------------------------------
# API Definition
# -----------------------------------------------------------------------------


app = FastAPI(title="Work Order Extraction API", version="0.1.0")


@app.post("/work-orders/audio", response_model=WorkOrder)
async def process_audio(
    file: UploadFile = File(...),
    client_name: Optional[str] = Form(None),
    site_address: Optional[str] = Form(None),
    date_override: Optional[str] = Form(None),
) -> WorkOrder:
    """Extract a partial work order from an uploaded audio recording."""
    hints = AudioRequest(
        client_name=client_name,
        site_address=site_address,
        date_override=date_override,
    )
    try:
        transcript = await transcribe_audio(file)
        extracted = await extract_work_order_from_transcript(transcript, hints)
        if hints.date_override:
            extracted["Date"] = hints.date_override
        work_order = postprocess_work_order(extracted, hints)
        work_order.Audio = [file.filename] if file.filename else None
        return work_order
    except Exception as exc:  # pragma: no cover - surface to caller
        logger.exception("Failed to process audio work order")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {exc}") from exc


@app.post("/work-orders/picture", response_model=WorkOrder)
async def process_picture(
    files: List[UploadFile] = File(...),
    client_name: Optional[str] = Form(None),
    site_address: Optional[str] = Form(None),
    reference_wo_id: Optional[str] = Form(None),
) -> WorkOrder:
    """Extract a partial work order from one or more image files."""
    hints = PictureRequest(
        client_name=client_name,
        site_address=site_address,
        reference_wo_id=reference_wo_id,
    )
    try:
        extracted = await extract_work_order_from_images(files, hints)
        work_order = postprocess_work_order(extracted, hints)
        work_order.Picture = [upload.filename for upload in files if upload.filename]
        return work_order
    except Exception as exc:  # pragma: no cover - surface to caller
        logger.exception("Failed to process picture work order")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {exc}") from exc
