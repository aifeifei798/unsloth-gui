"""推理 API：LoRA 列表、加载/卸载、WebSocket 流式对话。"""
from __future__ import annotations

import asyncio
import threading

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.i18n import t
from src.inference_utils import (
    list_trained_loras,
    load_inference_model,
    loaded_info,
    loaded_payload,
    lora_info,
    stream_chat,
    unload_model,
)

from ..errors import AppError
from ..schemas import LoadIn

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.get("/loras")
async def loras():
    names = list_trained_loras()
    return {"ok": True, "data": [lora_info(n) for n in names]}


@router.get("/status")
async def status():
    return {"ok": True, "data": {
        "message": loaded_info(),
        **loaded_payload(),
    }}


@router.post("/load")
async def load(body: LoadIn):
    try:
        msg = await asyncio.to_thread(
            load_inference_model, body.base_model_name, body.lora_name)
    except Exception as e:
        raise AppError(t("inf.load_fail", err=e, trace=""), 500, "load") from e
    return {"ok": True, "data": {"message": msg, **loaded_payload()}}


@router.post("/unload")
async def unload():
    msg = await asyncio.to_thread(unload_model)
    return {"ok": True, "data": {"message": msg, **loaded_payload()}}


def _consume_task_result(task: asyncio.Task) -> None:
    """把后台对话任务的异常取出来，避免 event loop 报 never retrieved."""
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        pass


async def _run_chat(ws: WebSocket, message: str, history: list,
                    system_prompt: str, params: dict,
                    cancel: threading.Event) -> None:
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def produce() -> None:
        try:
            for piece in stream_chat(
                message, history, system_prompt,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
                cancel_event=cancel,
            ):
                loop.call_soon_threadsafe(q.put_nowait, ("delta", piece))
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, ("end", None))

    threading.Thread(target=produce, daemon=True,
                     name="ws-chat").start()
    try:
        await ws.send_json({"type": "start"})
        while True:
            kind, payload = await q.get()
            if kind == "delta":
                await ws.send_json({"type": "delta", "text": payload})
            elif kind == "error":
                await ws.send_json({"type": "error", "message": payload})
                break
            else:
                break
        await ws.send_json({"type": "done", "cancelled": cancel.is_set()})
    except Exception:
        cancel.set()


@router.websocket("/ws")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    current: dict = {"cancel": None, "task": None}
    try:
        await ws.send_json({"type": "ready", "data": loaded_payload()})
        while True:
            try:
                data = await ws.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:
                await ws.send_json({"type": "error",
                                    "message": t("web.bad_message")})
                continue
            typ = (data or {}).get("type")
            if typ == "ping":
                await ws.send_json({"type": "pong"})
                continue
            if typ == "cancel":
                if current["cancel"] is not None:
                    current["cancel"].set()
                await ws.send_json({"type": "cancelling"})
                continue
            if typ != "chat":
                await ws.send_json({"type": "error",
                                    "message": t("web.bad_message")})
                continue
            task = current["task"]
            if task is not None and not task.done():
                await ws.send_json({"type": "error",
                                    "message": t("inf.busy")})
                continue
            message = str((data or {}).get("message") or "").strip()
            if not message:
                await ws.send_json({"type": "error",
                                    "message": t("web.empty_message")})
                continue
            history = data.get("history") or []
            if not isinstance(history, list):
                history = []
            history = [h for h in history[-100:] if isinstance(h, dict)
                       and h.get("role") in ("user", "assistant")]
            system_prompt = str(data.get("system_prompt") or "")
            try:
                params = {
                    "max_new_tokens": max(1, min(int(data.get("max_new_tokens") or 256), 4096)),
                    "temperature": max(0.0, min(float(data.get("temperature") or 0.7), 2.0)),
                    "top_p": max(0.01, min(float(data.get("top_p") or 0.95), 1.0)),
                    "top_k": max(0, min(int(data.get("top_k") or 40), 200)),
                }
            except (TypeError, ValueError):
                # 参数写坏了只回一条错误，别把整个 WebSocket 关掉
                await ws.send_json({"type": "error",
                                    "message": t("web.bad_params")})
                continue
            cancel = threading.Event()
            current["cancel"] = cancel
            task = asyncio.create_task(
                _run_chat(ws, message, history, system_prompt, params, cancel))
            task.add_done_callback(_consume_task_result)
            current["task"] = task
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if current["cancel"] is not None:
            current["cancel"].set()
        task = current["task"]
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=3)
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass
