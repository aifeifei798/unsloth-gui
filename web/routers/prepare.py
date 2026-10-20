"""数据处理 API：上传 → 读取列 → 单行预览 → 后台生成统一数据。"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from src.config import find_by_name, load_datasets_config
from src.dataprep import generate_unified, inspect_text, persist_upload, preview_row
from src.i18n import t

from ..deps import inspections, uploads
from ..errors import bad, not_found
from ..jobs import job_manager
from ..schemas import GenerateIn, InspectIn, MappingIn

router = APIRouter(prefix="/api/prepare", tags=["prepare"])

ALLOWED_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv", ".parquet", ".txt"}
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    name = file.filename or "upload"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise bad(t("web.bad_file", ext=suffix or "?"))
    tmp = tempfile.NamedTemporaryFile(
        prefix="ugui_upload_", suffix=suffix, delete=False)
    total = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise bad(t("web.file_too_big"))
            tmp.write(chunk)
        tmp.close()
        path = await asyncio.to_thread(persist_upload, tmp.name, name)
    finally:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        try:
            await file.close()
        except Exception:
            pass
    upload_id = uploads.put(path)
    return {"ok": True, "data": {
        "upload_id": upload_id, "name": name, "size": total,
        "suffix": suffix,
    }}


@router.post("/inspect")
async def inspect(body: InspectIn):
    local_path = ""
    existing = None
    if body.kind == "upload":
        local_path = uploads.get(body.upload_id) or ""
        if not local_path:
            raise bad(t("web.upload_expired"))
    elif body.kind == "existing":
        warnings: list[str] = []
        try:
            cfgs = load_datasets_config(warnings=warnings)
        except Exception as e:
            raise bad(t("warn.datasets", err=e)) from e
        existing = find_by_name(cfgs, body.existing_name)
        if existing is None:
            raise not_found(t("p.err.no_config", name=body.existing_name))
    elif body.kind == "hf":
        if not (body.hf_id or "").strip():
            raise bad(t("dp.err.no_hf"))
    try:
        text, state = await asyncio.to_thread(
            inspect_text, body.kind, body.hf_id, body.split, local_path, existing)
    except Exception as e:
        raise bad(t("p.err.read_fail", err=e)) from e
    token = inspections.put(state)
    return {"ok": True, "data": {
        "token": token,
        "text": text,
        "columns": state.get("columns", []),
        "n_rows": state.get("n_rows"),
        "streamed": bool(state.get("streamed")),
    }}


@router.post("/preview")
async def preview(body: MappingIn):
    state = inspections.get(body.token)
    if not state:
        raise bad(t("web.inspect_expired"))
    try:
        text = await asyncio.to_thread(
            preview_row, state, body.instruction_cols, body.input_cols,
            body.think_cols, body.output_cols, body.fixed_instruction)
    except Exception as e:
        raise bad(t("dp.preview.fail", err=e)) from e
    return {"ok": True, "data": {"preview": text}}


@router.post("/generate")
async def generate(body: GenerateIn):
    state = inspections.get(body.token)
    if not state:
        raise bad(t("web.inspect_expired"))
    if not (body.output_name or "").strip():
        raise bad(t("web.need_output_name"))

    def worker(job):
        def progress(frac: float, desc: str = "") -> None:
            job.emit_threadsafe("progress", {"frac": float(frac), "desc": desc})

        status, preview, name = generate_unified(
            state,
            body.instruction_cols, body.input_cols, body.think_cols,
            body.output_cols, body.output_name,
            fixed_instruction=body.fixed_instruction,
            progress=progress, cancel_event=job.cancel_event,
        )
        return {"status": status, "preview": preview, "name": name}

    job = job_manager.start("prepare", worker, exclusive=True)
    return {"ok": True, "data": {"job_id": job.id}}


@router.post("/cancel")
async def cancel(job_id: str = ""):
    job = job_manager.get(job_id) if job_id else job_manager.current("prepare")
    if job is None:
        raise not_found(t("web.job_not_found"))
    ok = job_manager.stop(job.id)
    return {"ok": True, "data": {"stopped": ok}}
