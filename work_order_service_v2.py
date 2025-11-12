"""
Work Order Extraction API (URL + RQ) + Bunny Relay
--------------------------------------------------

This module provides:
1) URL-only work-order extraction with RQ background jobs:
   - POST /work-orders/audio-url
   - POST /work-orders/picture-urls
   - GET  /jobs/{job_id}

2) Bunny Relay for safe browser uploads to Bunny Storage (no secrets in the browser):
   - POST /api/bunny/upload-ticket   -> returns a one-time upload URL + persistent CDN URL
   - PUT  /api/bunny/upload/{ticket} -> streams the file to Bunny Storage using AccessKey

Environment:
- OPENAI_API_KEY=...
- OPENAI_TEXT_MODEL=gpt-4o            (default)
- OPENAI_VISION_MODEL=gpt-4o          (default)
- OPENAI_TRANSCRIPTION_MODEL=whisper-1
- REDIS_URL=redis://localhost:6379/0
- RQ_QUEUE=wo_queue
- ALLOWED_MEDIA_HOSTS=bunny.example.b-cdn.net,storage.bunnycdn.com
- MAX_DOWNLOAD_MB=50                   # caps worker downloads
- WEBHOOK_TIMEOUT=5

# Bunny relay config
- BUNNY_STORAGE_ZONE=your-zone
- BUNNY_ACCESS_KEY=your-storage-zone-password
- BUNNY_PULL_HOST=your-pull.b-cdn.net  # used to build public persistent URLs
- BUNNY_TICKET_TTL=300                 # seconds (default 5 min)
- MAX_UPLOAD_MB=150                    # caps browser upload size to relay (default 150MB)
- ALLOWED_UPLOAD_MIME=image/jpeg,image/png,image/webp,audio/mpeg,audio/x-m4a,audio/mp4,audio/aac,audio/wav,audio/ogg,audio/webm
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Union, AsyncIterator, cast
from urllib.parse import urlparse

import httpx
import openai
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel, Field, AnyHttpUrl
from rq import Queue, get_current_job
from rq.job import Job

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_TEXT_MODEL = "gpt-4o"
DEFAULT_VISION_MODEL = "gpt-4o"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 800

CT_AUDIO = {
    "audio/mpeg",
    "audio/mp4",
    "audio/aac",
    "audio/x-m4a",
    "audio/wav",
    "audio/ogg",
    "audio/webm",
}
CT_IMAGE = {"image/jpeg", "image/png", "image/webp"}

DEFAULT_ALLOWED_UPLOAD_MIME = ",".join(sorted(CT_IMAGE | CT_AUDIO))


@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str
    text_model: str = DEFAULT_TEXT_MODEL
    vision_model: str = DEFAULT_VISION_MODEL
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    # Redis / RQ
    redis_url: str = "redis://localhost:6379/0"
    rq_queue: str = "wo_queue"
    # Worker download constraints
    allowed_media_hosts: List[str] = None
    max_download_bytes: int = 50 * 1024 * 1024  # 50MB
    webhook_timeout: float = 5.0
    # Bunny relay config
    bunny_storage_zone: Optional[str] = None
    bunny_access_key: Optional[str] = None
    bunny_pull_host: Optional[str] = None
    bunny_ticket_ttl: int = 300  # seconds
    max_upload_bytes: int = 150 * 1024 * 1024  # 150 MB for uploads via relay
    allowed_upload_mime: List[str] = None

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

        text_model = os.getenv("OPENAI_TEXT_MODEL", DEFAULT_TEXT_MODEL)
        vision_model = os.getenv("OPENAI_VISION_MODEL", DEFAULT_VISION_MODEL)
        transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL)

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        rq_queue = os.getenv("RQ_QUEUE", "wo_queue")

        allowed_hosts_raw = os.getenv("ALLOWED_MEDIA_HOSTS", "")
        allowed_hosts = [h.strip().lower() for h in allowed_hosts_raw.split(",") if h.strip()]

        max_download_mb = float(os.getenv("MAX_DOWNLOAD_MB", "50"))
        webhook_timeout = float(os.getenv("WEBHOOK_TIMEOUT", "5"))

        # Bunny
        bunny_zone = os.getenv("BUNNY_STORAGE_ZONE")
        bunny_key = os.getenv("BUNNY_ACCESS_KEY")
        bunny_pull = os.getenv("BUNNY_PULL_HOST")
        bunny_ttl = int(os.getenv("BUNNY_TICKET_TTL", "300"))

        max_upload_mb = float(os.getenv("MAX_UPLOAD_MB", "150"))
        allowed_upload_mime = os.getenv("ALLOWED_UPLOAD_MIME", DEFAULT_ALLOWED_UPLOAD_MIME)
        allowed_upload_mime_list = [m.strip().lower() for m in allowed_upload_mime.split(",") if m.strip()]

        return cls(
            openai_api_key=api_key,
            text_model=text_model,
            vision_model=vision_model,
            transcription_model=transcription_model,
            redis_url=redis_url,
            rq_queue=rq_queue,
            allowed_media_hosts=allowed_hosts,
            max_download_bytes=int(max_download_mb * 1024 * 1024),
            webhook_timeout=webhook_timeout,
            bunny_storage_zone=bunny_zone,
            bunny_access_key=bunny_key,
            bunny_pull_host=bunny_pull,
            bunny_ticket_ttl=bunny_ttl,
            max_upload_bytes=int(max_upload_mb * 1024 * 1024),
            allowed_upload_mime=allowed_upload_mime_list,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()


ContentPayload = Union[str, Sequence[Dict[str, Any]]]


class OpenAIAdapter:
    """Thin wrapper over the OpenAI SDK to support multiple versions."""

    def __init__(
        self,
        *,
        api_key: str,
        text_model: str,
        vision_model: str,
        transcription_model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.text_model = text_model
        self.vision_model = vision_model
        self.transcription_model = transcription_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if hasattr(openai, "OpenAI"):
            self._client = openai.OpenAI(api_key=api_key)
            self._modern = True
        else:  # pragma: no cover
            openai.api_key = api_key
            self._client = openai
            self._modern = False

    def chat_completion(
        self,
        *,
        system_prompt: str,
        user_content: ContentPayload,
        model: Optional[str] = None,
    ) -> str:
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
        audio_stream = io.BytesIO(contents)
        audio_stream.name = file_name or "audio.m4a"
        if self._modern and hasattr(self._client, "audio"):
            result = self._client.audio.transcriptions.create(
                model=get_settings().transcription_model,
                file=audio_stream,
                response_format="text",
            )
            if isinstance(result, dict):  # pragma: no cover
                text = result.get("text")
            else:
                text = getattr(result, "text", None)
            return text or str(result)
        # legacy
        result = self._client.Audio.transcribe(  # pragma: no cover
            model=get_settings().transcription_model,
            file=audio_stream,
            response_format="text",
        )
        if isinstance(result, dict):
            return result.get("text") or str(result)
        return result


@lru_cache(maxsize=1)
def get_openai_adapter() -> OpenAIAdapter:
    s = get_settings()
    return OpenAIAdapter(
        api_key=s.openai_api_key,
        text_model=s.text_model,
        vision_model=s.vision_model,
        transcription_model=s.transcription_model,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
    )

# -----------------------------------------------------------------------------
# Common helpers
# -----------------------------------------------------------------------------

CODE_FENCE_PATTERN = re.compile(r"^```[\w]*\s*\n|\n```$", re.MULTILINE)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```[\w]*\s*\n", "", stripped)
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def parse_model_json(raw_content: str, context: str) -> Dict[str, Any]:
    cleaned = strip_code_fences(raw_content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:  # pragma: no cover
        preview = cleaned[:2000]
        logger.exception("Failed to decode JSON for %s: %s", context, preview)
        raise ValueError(f"Failed to parse JSON from model output for {context}: {exc}") from exc


def clone_model(model: BaseModel) -> BaseModel:
    if hasattr(model, "model_copy"):
        return model.model_copy(deep=True)  # type: ignore[attr-defined]
    return model.copy(deep=True)


def model_to_dict(model: Optional[BaseModel]) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def normalise_work_order_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
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


def dedup_key(item: Any) -> str:
    if isinstance(item, BaseModel):
        serialisable = model_to_dict(item)
    else:
        serialisable = item
    try:
        return json.dumps(serialisable, sort_keys=True, default=str)
    except TypeError:  # pragma: no cover
        return repr(serialisable)


def _host_allowed(url: str) -> bool:
    s = get_settings()
    if not s.allowed_media_hosts:
        return True
    host = urlparse(url).hostname or ""
    host = host.lower()
    return any(host == h or host.endswith("." + h) for h in s.allowed_media_hosts)


async def head_check(url: str) -> dict:
    if not _host_allowed(url):
        raise ValueError("host not allowed")
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as c:
        r = await c.head(str(url))
        r.raise_for_status()
        return {
            "content_type": r.headers.get("content-type", "").split(";")[0].strip(),
            "content_length": int(r.headers.get("content-length", "0") or 0),
        }


async def download_bytes(url: str, expect: str) -> bytes:
    """Stream-safe download with size cap. expect: 'audio' | 'image'"""
    s = get_settings()
    if not _host_allowed(url):
        raise ValueError("host not allowed")
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as c:
        async with c.stream("GET", str(url)) as r:
            r.raise_for_status()
            ct = r.headers.get("content-type", "").split(";")[0].strip()
            if expect == "audio" and ct not in CT_AUDIO:
                raise ValueError(f"unexpected content-type for audio: {ct}")
            if expect == "image" and ct not in CT_IMAGE:
                raise ValueError(f"unexpected content-type for image: {ct}")
            total = 0
            chunks = []
            async for chunk in r.aiter_bytes():
                total += len(chunk)
                if total > s.max_download_bytes:
                    raise ValueError("file too large")
                chunks.append(chunk)
            return b"".join(chunks)

# -----------------------------------------------------------------------------
# Data Models (Work Order)
# -----------------------------------------------------------------------------

class Address(BaseModel):
    Line1: Optional[str] = None
    City: Optional[str] = None
    CountrySubDivisionCode: Optional[str] = None
    PostalCode: Optional[str] = None
    Country: Optional[str] = None


class BillTo(BaseModel):
    customerId: Optional[str] = None
    name: Optional[str] = None
    address: Optional[Address] = None


class ItemRef(BaseModel):
    value: str
    name: Optional[str] = None


class SalesItemLineDetail(BaseModel):
    ItemRef: ItemRef


class LineItem(BaseModel):
    Amount: float
    Description: str
    DetailType: str = "SalesItemLineDetail"
    SalesItemLineDetail: SalesItemLineDetail


class WorkOrder(BaseModel):
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


class AudioUrlRequest(BaseModel):
    audio_url: AnyHttpUrl
    client_name: Optional[str] = Field(None)
    site_address: Optional[str] = Field(None)
    date_override: Optional[str] = Field(None)
    callback_url: Optional[AnyHttpUrl] = Field(None)


class PictureUrlsRequest(BaseModel):
    image_urls: List[AnyHttpUrl]
    client_name: Optional[str] = Field(None)
    site_address: Optional[str] = Field(None)
    reference_wo_id: Optional[str] = Field(None)
    callback_url: Optional[AnyHttpUrl] = Field(None)


class JobEnqueueResponse(BaseModel):
    job_id: str
    status_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | started | finished | failed | deferred
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# -----------------------------------------------------------------------------
# Mock Data + Resolvers
# -----------------------------------------------------------------------------

CLIENTS: Dict[str, Dict[str, Any]] = {
    "maple towers": {
        "customerId": "123",
        "name": "Maple Towers",
        "address": Address(Line1="123 Maple Street", City="Toronto", CountrySubDivisionCode="ON", PostalCode="M5V 2T6", Country="Canada"),
    },
    "john doe": {
        "customerId": "124",
        "name": "John Doe",
        "address": Address(Line1="456 Oak Avenue", City="Toronto", CountrySubDivisionCode="ON", PostalCode="M5V 1S8", Country="Canada"),
    },
}

ITEMS: Dict[str, Dict[str, Any]] = {
    "plumbing labour": {"value": "79", "name": "Plumbing Labour"},
    "labour": {"value": "79", "name": "Plumbing Labour"},
    "diagnostic": {"value": "80", "name": "Diagnostic Service"},
    "pex pipe": {"value": "145", "name": "PEX Pipe"},
    "p-trap assembly": {"value": "146", "name": "P-trap Assembly"},
    "copper pipe": {"value": "154", "name": "Copper Pipe"},
}

class ItemRefModel(BaseModel):
    value: str
    name: Optional[str] = None

def resolve_client(name: Optional[str]) -> Optional[BillTo]:
    if not name:
        return None
    key = name.strip().lower()
    client_info = CLIENTS.get(key)
    if not client_info:
        return BillTo(customerId="UNMAPPED", name=name)
    return BillTo(customerId=client_info["customerId"], name=client_info["name"], address=client_info.get("address"))

def resolve_item(name: Optional[str]) -> ItemRefModel:
    if not name:
        return ItemRefModel(value="UNMAPPED", name="")
    key = name.strip().lower()
    for candidate, ref in ITEMS.items():
        if candidate in key:
            return ItemRefModel(value=ref["value"], name=ref["name"])
    return ItemRefModel(value="UNMAPPED", name=name)

# -----------------------------------------------------------------------------
# Prompt helpers & post-processing
# -----------------------------------------------------------------------------

AUDIO_SYSTEM_PROMPT = (
    "You are an AI assistant helping to extract structured work order data "
    "from a plumber's audio note. Return only a JSON object with the "
    "following fields: Date (YYYY-MM-DD), Bill_to (with customerId, name, "
    "address), Description (list of strings), Line (list of line items with "
    "Amount, Description, DetailType='SalesItemLineDetail', "
    "SalesItemLineDetail with ItemRef {value,name}), status, status_detail. "
    "Do not return any explanatory text. If a field is not present, omit it. "
    "Use any hints provided to resolve ambiguity."
)

IMAGE_SYSTEM_PROMPT = (
    "You are an AI assistant extracting structured work order data from "
    "photos of handwritten or printed forms, receipts or notes. Return only "
    "a JSON object with the following fields: Date (YYYY-MM-DD), Bill_to "
    "(with customerId, name, address), Description (list of strings), Line "
    "(list of line items with Amount, Description, DetailType='SalesItemLineDetail', "
    "SalesItemLineDetail with ItemRef {value,name}), status, status_detail. "
    "If a field is not present in the image, omit it."
)

class AudioRequest(BaseModel):
    client_name: Optional[str] = None
    site_address: Optional[str] = None
    date_override: Optional[str] = None

class PictureRequest(BaseModel):
    client_name: Optional[str] = None
    site_address: Optional[str] = None
    reference_wo_id: Optional[str] = None

def build_audio_prompt(transcript: str, hints: AudioRequest) -> str:
    hint_lines: List[str] = []
    if hints.client_name:
        hint_lines.append(f"Client name hint: {hints.client_name}")
    if hints.site_address:
        hint_lines.append(f"Site address hint: {hints.site_address}")
    if hints.date_override:
        hint_lines.append(f"Date override hint: {hints.date_override}")
    parts: List[str] = []
    if hint_lines:
        parts.append("\n".join(hint_lines))
    parts.append("Transcript:")
    parts.append(transcript)
    return "\n".join(parts)

def build_image_user_content(data_or_url: str, hints: PictureRequest) -> List[Dict[str, Any]]:
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
    content.append({"type": "image_url", "image_url": {"url": data_or_url}})
    return content

def postprocess_work_order(data: Dict[str, Any], hints: Optional[Union[AudioRequest, PictureRequest]] = None) -> WorkOrder:
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
            cloned_line.SalesItemLineDetail.ItemRef = ItemRef(value=item_ref.value, name=item_ref.name)
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

def work_order_from_payload(payload: Union["WorkOrder", Dict[str, Any]]) -> "WorkOrder":
    if isinstance(payload, WorkOrder):
        return payload
    if hasattr(WorkOrder, "model_validate"):
        clean_payload = normalise_work_order_payload(payload)  # type: ignore[arg-type]
        return WorkOrder.model_validate(clean_payload)  # type: ignore[attr-defined]
    clean_payload = normalise_work_order_payload(payload)  # type: ignore[arg-type]
    return WorkOrder.parse_obj(clean_payload)

# -----------------------------------------------------------------------------
# Core async flows used by RQ tasks
# -----------------------------------------------------------------------------

async def _transcribe_audio_from_url(audio_url: str) -> str:
    adapter = get_openai_adapter()
    meta = await head_check(audio_url)
    if meta["content_length"] == 0:
        raise ValueError("empty audio")
    audio_bytes = await download_bytes(audio_url, "audio")
    return await asyncio.to_thread(adapter.transcribe_audio, "note.m4a", audio_bytes)

async def _extract_from_transcript(transcript: str, hints: AudioRequest) -> Dict[str, Any]:
    adapter = get_openai_adapter()
    prompt = build_audio_prompt(transcript, hints)
    raw_json = await asyncio.to_thread(
        adapter.chat_completion,
        system_prompt=AUDIO_SYSTEM_PROMPT,
        user_content=prompt,
        model=adapter.text_model,
    )
    return parse_model_json(raw_json, "audio transcript")

async def _extract_from_image_urls(image_urls: List[str], hints: PictureRequest) -> Dict[str, Any]:
    adapter = get_openai_adapter()
    merged: Optional[WorkOrder] = None
    for url in image_urls:
        await head_check(url)
        user_content = build_image_user_content(url, hints)
        raw_json = await asyncio.to_thread(
            adapter.chat_completion,
            system_prompt=IMAGE_SYSTEM_PROMPT,
            user_content=user_content,
            model=adapter.vision_model,
        )
        payload = parse_model_json(raw_json, url)
        partial = work_order_from_payload(payload)
        merged = partial if merged is None else merged.merge(partial)
    return model_to_dict(merged) if merged else {}

# -----------------------------------------------------------------------------
# RQ tasks (sync entrypoints)
# -----------------------------------------------------------------------------

def _set_progress(pct: int, note: str = ""):
    job = get_current_job()
    if job:
        job.meta["progress"] = {"percent": pct, "note": note}
        job.save_meta()

def _post_webhook(url: str, payload: Dict[str, Any], timeout: float = 5.0):
    try:
        with httpx.Client(timeout=timeout) as c:
            c.post(url, json=payload)
    except Exception:
        pass

def audio_job_url(audio_url: str, hints: Dict[str, Any], callback_url: Optional[str] = None) -> Dict[str, Any]:
    s = get_settings()
    try:
        _set_progress(5, "Starting")
        result = asyncio.run(_audio_flow(audio_url, hints))
        _set_progress(100, "Done")
        if callback_url:
            _post_webhook(callback_url, result, timeout=s.webhook_timeout)
        return result
    except Exception as e:
        _set_progress(100, f"Failed: {e}")
        raise

async def _audio_flow(audio_url: str, hints: Dict[str, Any]) -> Dict[str, Any]:
    transcript = await _transcribe_audio_from_url(audio_url)
    extracted = await _extract_from_transcript(transcript, AudioRequest(**hints))
    if hints.get("date_override"):
        extracted["Date"] = hints["date_override"]
    wo = postprocess_work_order(extracted, AudioRequest(**hints))
    wo.Audio = [audio_url]
    return wo.model_dump()

def picture_job_urls(image_urls: List[str], hints: Dict[str, Any], callback_url: Optional[str] = None) -> Dict[str, Any]:
    s = get_settings()
    try:
        _set_progress(5, "Starting")
        result = asyncio.run(_image_flow(image_urls, hints))
        _set_progress(100, "Done")
        if callback_url:
            _post_webhook(callback_url, result, timeout=s.webhook_timeout)
        return result
    except Exception as e:
        _set_progress(100, f"Failed: {e}")
        raise

async def _image_flow(image_urls: List[str], hints: Dict[str, Any]) -> Dict[str, Any]:
    extracted = await _extract_from_image_urls(image_urls, PictureRequest(**hints))
    wo = postprocess_work_order(extracted, PictureRequest(**hints))
    wo.Picture = image_urls
    return wo.model_dump()

# -----------------------------------------------------------------------------
# FastAPI app & queues
# -----------------------------------------------------------------------------

app = FastAPI(title="Work Order API (URL + RQ + Bunny Relay)", version="0.3.0")

_settings = get_settings()
_redis = redis.from_url(_settings.redis_url)
queue = Queue(_settings.rq_queue, connection=_redis)

# -----------------------------------------------------------------------------
# Work-order enqueue + status
# -----------------------------------------------------------------------------

@app.post("/work-orders/audio-url", response_model=JobEnqueueResponse)
def enqueue_audio_url(body: AudioUrlRequest) -> JobEnqueueResponse:
    hints = {
        "client_name": body.client_name,
        "site_address": body.site_address,
        "date_override": body.date_override,
    }
    job = queue.enqueue(
        audio_job_url,
        str(body.audio_url),
        hints,
        str(body.callback_url) if body.callback_url else None,
        job_timeout=900,
    )
    return JobEnqueueResponse(job_id=job.id, status_url=f"/jobs/{job.id}")

@app.post("/work-orders/picture-urls", response_model=JobEnqueueResponse)
def enqueue_picture_urls(body: PictureUrlsRequest) -> JobEnqueueResponse:
    hints = {
        "client_name": body.client_name,
        "site_address": body.site_address,
        "reference_wo_id": body.reference_wo_id,
    }
    job = queue.enqueue(
        picture_job_urls,
        [str(u) for u in body.image_urls],
        hints,
        str(body.callback_url) if body.callback_url else None,
        job_timeout=900,
    )
    return JobEnqueueResponse(job_id=job.id, status_url=f"/jobs/{job.id}")

@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str) -> JobStatus:
    try:
        job = Job.fetch(job_id, connection=_redis)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get_status()
    payload: Dict[str, Any] = {
        "job_id": job.id,
        "status": status,
        "progress": job.meta.get("progress"),
    }
    if status == "finished":
        payload["result"] = job.result
    elif status == "failed":
        payload["error"] = str(job.exc_info or "Failed")
    return JobStatus(**payload)

# -----------------------------------------------------------------------------
# Bunny Relay: ticket + streaming upload
# -----------------------------------------------------------------------------

class TicketRequest(BaseModel):
    ext: str = Field(..., description="File extension like .m4a .jpg .png")
    org_id: str = Field(..., description="Organization ID to compose the object key")
    kind: str = Field(..., description="audio|image")
    # Optional subfolder override; if omitted we use 'workorders'
    folder: Optional[str] = Field(None, description="Optional subfolder under org_id")

class TicketResponse(BaseModel):
    upload_url: str
    cdn_url: str
    key: str
    expires_in: int

def _require_bunny_config():
    s = get_settings()
    missing = []
    if not s.bunny_storage_zone: missing.append("BUNNY_STORAGE_ZONE")
    if not s.bunny_access_key:   missing.append("BUNNY_ACCESS_KEY")
    if not s.bunny_pull_host:    missing.append("BUNNY_PULL_HOST")
    if missing:
        raise HTTPException(status_code=500, detail=f"Bunny config missing: {', '.join(missing)}")

def _ext_normalize(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        return ""
    if not ext.startswith("."):
        ext = "." + ext
    return ext.lower()

def _build_object_key(org_id: str, folder: Optional[str], ext: str) -> str:
    f = folder.strip("/") if folder else "workorders"
    return f"{org_id}/{f}/{uuid.uuid4().hex}{ext}"

def _ticket_key(ticket: str) -> str:
    return f"bunny:ticket:{ticket}"

def _save_ticket(ticket: str, key: str, ttl: int):
    # store in Redis with TTL
    rkey = _ticket_key(ticket)
    _redis.setex(rkey, ttl, key)

def _consume_ticket(ticket: str) -> Optional[str]:
    rkey = _ticket_key(ticket)
    with _redis.pipeline() as p:
        p.get(rkey)
        p.delete(rkey)
        val, _ = p.execute()
    return val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else val

@app.post("/api/bunny/upload-ticket", response_model=TicketResponse)
def bunny_upload_ticket(body: TicketRequest) -> TicketResponse:
    _require_bunny_config()
    s = get_settings()

    ext = _ext_normalize(body.ext)
    if not ext:
        raise HTTPException(status_code=400, detail="Invalid extension")

    # Build persistent object key and one-time ticket id
    key = _build_object_key(body.org_id, body.folder, ext)
    ticket = uuid.uuid4().hex
    _save_ticket(ticket, key, s.bunny_ticket_ttl)

    # Relay upload URL (your API path), plus final persistent CDN URL
    upload_url = f"/api/bunny/upload/{ticket}"
    cdn_url = f"https://{s.bunny_pull_host}/{key}"

    return TicketResponse(
        upload_url=upload_url,
        cdn_url=cdn_url,
        key=key,
        expires_in=s.bunny_ticket_ttl,
    )

async def _limited_stream(iter_bytes: AsyncIterator[bytes], byte_limit: int) -> AsyncIterator[bytes]:
    total = 0
    async for chunk in iter_bytes:
        total += len(chunk)
        if total > byte_limit:
            raise HTTPException(status_code=413, detail="Upload too large")
        yield chunk

@app.put("/api/bunny/upload/{ticket}")
async def bunny_upload_stream(
    ticket: str,
    request: Request,
    content_type: str = Header(default="application/octet-stream"),
    content_length: Optional[int] = Header(default=None),
):
    _require_bunny_config()
    s = get_settings()

    # Resolve and consume the one-time ticket
    key = _consume_ticket(ticket)
    if not key:
        raise HTTPException(status_code=410, detail="Ticket expired or invalid")

    # Validate content-type
    ct = (content_type or "application/octet-stream").split(";")[0].strip().lower()
    if ct not in s.allowed_upload_mime:
        raise HTTPException(status_code=415, detail=f"Content-Type not allowed: {ct}")

    # Enforce size limits (header + streaming)
    if content_length is not None:
        try:
            clen = int(content_length)
            if clen > s.max_upload_bytes:
                raise HTTPException(status_code=413, detail="Upload too large")
        except ValueError:
            pass  # ignore malformed, we will enforce during streaming anyway

    storage_url = f"https://storage.bunnycdn.com/{s.bunny_storage_zone}/{key}"

    async with httpx.AsyncClient(timeout=60) as client:
        # Stream request body to Bunny with size limit
        try:
            async with client.stream(
                "PUT",
                storage_url,
                headers={"AccessKey": s.bunny_access_key, "Content-Type": ct},
                content=_limited_stream(request.stream(), s.max_upload_bytes),
            ) as resp:
                # drain response (in case Bunny sends a body)
                await resp.aread()
                if resp.status_code not in (200, 201):
                    raise HTTPException(status_code=502, detail=f"Bunny error: {resp.status_code}")
        except HTTPException:
            # re-raise cleanly
            raise
        except Exception as e:
            logger.exception("Bunny upload failed")
            raise HTTPException(status_code=502, detail=f"Bunny upload failed: {e}")

    return {"status": "uploaded", "key": key}

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# Run notes:
# - Uvicorn: uvicorn work_order_api:app --reload
# - RQ worker: rq worker -u $REDIS_URL wo_queue
# -----------------------------------------------------------------------------
