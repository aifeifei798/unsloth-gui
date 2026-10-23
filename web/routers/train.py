"""训练 API：启动/停止/状态/SSE/TensorBoard。"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query, Request

from src.config import (
    find_by_name,
    is_trainable_dataset,
    load_datasets_config,
    load_models_config,
    trainable_reason,
)
from src.i18n import t
from src.inference_utils import lora_info, list_trained_loras
from src.tb_utils import launch_tensorboard
from src.train_utils import (
    TrainRequest,
    current_experiment,
    is_training,
    request_cancel,
    run_training,
)

from ..errors import AppError, bad, not_found
from ..jobs import job_manager, sse_response
from ..schemas import TrainIn

router = APIRouter(prefix="/api/train", tags=["train"])


def _last_train_job():
    return job_manager.by_kind.get("train")


@router.get("/status")
async def status():
    job = _last_train_job()
    return {"ok": True, "data": {
        "running": is_training(),
        "experiment": current_experiment(),
        "job": job.payload() if job is not None else None,
    }}


@router.post("/start")
async def start(body: TrainIn):
    if is_training() or job_manager.current("train") is not None:
        raise AppError(t("tr.running", exp=current_experiment() or "?"), 409, "busy")
    if not (body.experiment_name or "").strip():
        raise bad(t("tr.no_exp_name"))
    if not body.selected_dataset_names:
        raise bad(t("tr.no_datasets"))
    try:
        models = load_models_config()
        datasets = load_datasets_config(warnings=[])
    except Exception as e:
        raise bad(t("warn.datasets", err=e)) from e
    if find_by_name(models, body.selected_model_name) is None:
        raise bad(t("tr.no_model", name=body.selected_model_name))
    for name in body.selected_dataset_names:
        cfg = find_by_name(datasets, name)
        if cfg is None:
            raise bad(t("tr.warn_skip", name=name))
        if not is_trainable_dataset(cfg):
            raise bad(t("tr.unprocessed_reason", name=name,
                        reason=trainable_reason(cfg)))

    req = TrainRequest(
        experiment_name=body.experiment_name,
        resume_training=bool(body.resume_training),
        truncate_dataset=bool(body.truncate_dataset),
        max_samples=int(body.max_samples),
        training_mode=body.training_mode,
        num_epochs=float(body.num_epochs),
        max_steps=int(body.max_steps),
        save_steps=int(body.save_steps),
        selected_model_name=body.selected_model_name,
        selected_dataset_names=list(body.selected_dataset_names),
        lora_r=int(body.lora_r),
        lora_alpha=int(body.lora_alpha),
        batch_size=int(body.batch_size),
        grad_accum=int(body.grad_accum),
        lr=float(body.lr),
        max_seq_length=int(body.max_seq_length),
        use_chat_template=bool(body.use_chat_template),
    )

    def worker(job):
        def progress(frac: float, desc: str = "") -> None:
            job.emit_threadsafe("progress", {"frac": float(frac), "desc": desc})

        def on_log(logs: dict) -> None:
            job.emit_threadsafe("metrics", logs)

        last = ""
        # run_training 的返回值是结构化结果（ok / code），不再靠文案前缀猜
        outcome: dict = {"ok": False, "code": "unknown"}
        gen = run_training(req, progress=progress, on_log=on_log,
                           cancel_event=job.cancel_event)
        while True:
            try:
                msg = next(gen)
            except StopIteration as stop:
                if isinstance(stop.value, dict):
                    outcome = stop.value
                break
            last = msg
            job.emit_threadsafe("log", {"text": msg})
        return {
            "last": last,
            "ok": bool(outcome.get("ok")),
            "code": str(outcome.get("code") or "unknown"),
            "experiment": req.experiment_name,
        }

    job = job_manager.start("train", worker, exclusive=True)
    return {"ok": True, "data": {
        "job_id": job.id, "state": job.state,
        "experiment": req.experiment_name,
    }}


@router.post("/stop")
async def stop():
    job = job_manager.current("train") or _last_train_job()
    if job is None or job.is_finished():
        return {"ok": True, "data": {"stopped": False}, "message": t("tr.cancel_none")}
    job_manager.stop(job.id)
    msg = request_cancel()
    return {"ok": True, "data": {"stopped": True}, "message": msg}


@router.get("/events")
async def events(request: Request, job_id: str = Query(""), last_id: int = Query(0)):
    job = job_manager.get(job_id) if job_id else _last_train_job()
    if job is None:
        raise not_found(t("web.job_not_found"))
    # 浏览器 EventSource 重连时带的是 Last-Event-ID 头，不是 query 参数
    header = request.headers.get("last-event-id")
    if header and header.isdigit():
        last_id = max(last_id, int(header))
    return await sse_response(job, request, last_id)


@router.get("/tensorboard")
async def tensorboard(port: int = Query(6006, ge=1024, le=65535)):
    ok, msg, actual = await asyncio.to_thread(launch_tensorboard, port)
    return {"ok": True, "data": {
        "started": ok, "message": msg, "port": actual,
    }}


@router.get("/loras")
async def loras():
    names = list_trained_loras()
    return {"ok": True, "data": [lora_info(n) for n in names]}
