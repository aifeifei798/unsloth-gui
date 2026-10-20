"""系统级接口：健康检查、语言包、切换语言。"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

from src.i18n import dictionary, language, normalize_lang, t
from src.inference_utils import loaded_payload
from src.train_utils import current_experiment, is_training

from .. import __version__
from ..errors import not_found
from ..jobs import job_manager, sse_response

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health():
    return {"ok": True, "data": {
        "version": __version__,
        "training": is_training(),
        "experiment": current_experiment(),
        "inference": loaded_payload(),
    }}


@router.get("/i18n/{lang}")
async def i18n(lang: str):
    code = normalize_lang(lang)
    return {"ok": True, "data": {"lang": code, "messages": dictionary(code)}}


@router.get("/lang/{lang}")
async def switch_lang(lang: str, response: Response):
    code = normalize_lang(lang)
    response.set_cookie("ugui_lang", code, max_age=365 * 24 * 3600,
                        samesite="lax")
    with language(code):
        message = t("web.lang_changed")
    return {"ok": True, "data": {"lang": code, "message": message}}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise not_found(t("web.job_not_found"))
    return {"ok": True, "data": job.payload()}


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, last_id: int = Query(0)):
    job = job_manager.get(job_id)
    if job is None:
        raise not_found(t("web.job_not_found"))
    header = request.headers.get("last-event-id")
    if header and header.isdigit():
        last_id = max(last_id, int(header))
    return await sse_response(job, request, last_id)


@router.post("/jobs/{job_id}/cancel")
async def job_cancel(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise not_found(t("web.job_not_found"))
    stopped = job_manager.stop(job_id)
    return {"ok": True, "data": {"stopped": stopped},
            "message": t("web.job_stopping") if stopped else t("web.job_done")}
