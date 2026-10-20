"""模型管理 API：增删改查 models.json。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query

from src.config import (
    ModelConfig,
    find_by_name,
    load_models_config,
    model_source,
    save_models_config,
    source_label,
)
from src.i18n import t
from src.paths import resolve_local

from ..errors import AppError, bad, conflict, not_found
from ..schemas import ModelIn, ModelUpdateIn

router = APIRouter(prefix="/api/models", tags=["models"])

_SOURCE_ALIASES = {
    "auto": "auto", "自动": "auto",
    "local": "local", "本地路径": "local",
    "huggingface": "huggingface", "hf": "huggingface", "hugging face": "huggingface",
    "modelscope": "modelscope", "魔搭 modelscope": "modelscope", "魔搭": "modelscope",
}
_DTYPES = {"", "auto", "自动", None, "bfloat16", "float16"}


def _source(code: str) -> str:
    key = (code or "auto").strip().lower()
    return _SOURCE_ALIASES.get(key, key)


def _payload(m: ModelConfig) -> dict:
    src = model_source(m)
    missing = False
    if src == "local":
        try:
            missing = not Path(m.resolved_model_id()).exists()
        except Exception:
            missing = True
    return {
        "display_name": m.display_name,
        "model_id": m.model_id,
        "resolved_id": m.resolved_model_id(),
        "source": m.source,
        "source_code": src,
        "source_label": source_label(src),
        "load_in_4bit": bool(m.load_in_4bit),
        "dtype": m.dtype,
        "max_seq_length": int(m.max_seq_length),
        "chat_template": m.chat_template or "",
        "missing_local": missing,
    }


def _normalize(body: ModelIn) -> ModelConfig:
    display = (body.display_name or "").strip()
    model_id = (body.model_id or "").strip()
    if not display:
        raise bad(t("m.err.noname"))
    if not model_id:
        raise bad(t("m.err.noid"))
    src = _source(body.source)
    if src not in ("auto", "local", "huggingface", "modelscope"):
        raise bad(t("cfg.model_bad_source", name=display, source=body.source))
    if src != "local" and " " in model_id:
        raise bad(t("m.err.space"))
    looks_local = src == "local" or model_id.startswith(("~", "/", "./", "../"))
    if looks_local:
        try:
            p = resolve_local(model_id)
        except Exception as e:
            raise bad(t("m.err.local_missing", id=model_id, path=e)) from e
        if not p.exists():
            raise bad(t("m.err.local_missing", id=model_id, path=p))
    seq = int(body.max_seq_length or 0)
    if seq < 256:
        raise bad(t("m.err.seq_min"))
    dtype = (body.dtype or "").strip()
    if dtype in ("", "auto", "自动"):
        dtype = None
    elif dtype not in ("bfloat16", "float16"):
        raise bad(t("web.bad_dtype", value=dtype))
    return ModelConfig(
        display_name=display,
        model_id=model_id,
        load_in_4bit=bool(body.load_in_4bit),
        dtype=dtype,
        max_seq_length=seq,
        chat_template=(body.chat_template or "").strip() or None,
        source=src,
    )


def _load() -> list[ModelConfig]:
    try:
        return load_models_config()
    except Exception as e:
        raise AppError(t("warn.models", err=e), 500, "config") from e


def _save(models: list[ModelConfig]) -> None:
    try:
        save_models_config(models)
    except Exception as e:
        raise AppError(t("m.err.write", err=e), 500, "write") from e


@router.get("")
async def list_models():
    models = _load()
    return {"ok": True, "data": [_payload(m) for m in models]}


@router.post("")
async def create_model(body: ModelIn):
    models = _load()
    cfg = _normalize(body)
    if find_by_name(models, cfg.display_name) is not None:
        raise conflict(t("web.model_exists", name=cfg.display_name))
    models.append(cfg)
    _save(models)
    return {"ok": True, "data": _payload(cfg),
            "message": t("m.saved", action=t("m.act_add"),
                         display=cfg.display_name, source=source_label(model_source(cfg)))}


@router.put("")
async def update_model(body: ModelUpdateIn):
    models = _load()
    old_name = (body.original_name or body.display_name or "").strip()
    idx = next((i for i, m in enumerate(models)
                if m.display_name == old_name), None)
    if idx is None:
        raise not_found(t("m.err.notfound_form", name=old_name))
    cfg = _normalize(body)
    if cfg.display_name != old_name and find_by_name(models, cfg.display_name):
        raise conflict(t("web.model_exists", name=cfg.display_name))
    old = models[idx]
    models[idx] = cfg
    try:
        _save(models)
    except Exception:
        models[idx] = old
        raise
    return {"ok": True, "data": _payload(cfg),
            "message": t("m.saved", action=t("m.act_update"),
                         display=cfg.display_name, source=source_label(model_source(cfg)))}


@router.delete("")
async def delete_model(name: str = Query(...)):
    models = _load()
    idx = next((i for i, m in enumerate(models) if m.display_name == name), None)
    if idx is None:
        raise not_found(t("m.err.notfound_form", name=name))
    if len(models) <= 1:
        raise bad(t("m.err.last"))
    removed = models.pop(idx)
    try:
        _save(models)
    except Exception:
        models.insert(idx, removed)
        raise
    return {"ok": True, "data": {"name": name}, "message": t("m.deleted", name=name)}
