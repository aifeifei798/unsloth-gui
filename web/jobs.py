"""后台任务 + SSE 事件流。

训练、数据生成都是同步阻塞逻辑，放进后台线程跑：
- 线程只通过 `emit_threadsafe` 往事件环里塞事件
- 每个 SSE 连接有独立 asyncio.Queue，断线不影响任务
- 任务结束事件保留在环里，重连可以补发
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4

from src.i18n import get_lang, language, t

from .errors import AppError


@dataclass
class JobEvent:
    id: int
    event: str
    data: dict


@dataclass
class Job:
    kind: str
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    events: deque = field(default_factory=lambda: deque(maxlen=3000))
    subscribers: set = field(default_factory=set)
    next_id: int = 1
    loop: Optional[asyncio.AbstractEventLoop] = None
    thread: Optional[threading.Thread] = None
    lang: Optional[str] = None  # 启动任务时的界面语言，worker 线程里还原

    def emit(self, event: str, data: dict) -> None:
        """只允许在事件循环线程调用。"""
        ev = JobEvent(self.next_id, event, data)
        self.next_id += 1
        self.events.append(ev)
        for q in list(self.subscribers):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                # 慢客户端积压：丢最旧的一条再塞新的，
                # 保证 done / error / end 这类终止事件一定送得到。
                try:
                    q.get_nowait()
                    q.put_nowait(ev)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    def emit_threadsafe(self, event: str, data: dict) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self.emit, event, data)
        except RuntimeError:
            pass

    async def subscribe(self, last_id: int = 0) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        for ev in list(self.events):
            if ev.id > last_id:
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    break
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def is_finished(self) -> bool:
        return self.state in ("done", "error", "cancelled")

    def payload(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.jobs: dict[str, Job] = {}
        self.by_kind: dict[str, Job] = {}

    def bind_loop(self) -> None:
        self.loop = asyncio.get_running_loop()

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def current(self, kind: str) -> Optional[Job]:
        job = self.by_kind.get(kind)
        return job if job is not None and not job.is_finished() else None

    def busy(self, kind: Optional[str] = None) -> bool:
        if kind is not None:
            return self.current(kind) is not None
        return any(not j.is_finished() for j in self.jobs.values())

    def cleanup(self, keep_seconds: float = 6 * 3600) -> None:
        cutoff = time.time() - keep_seconds
        for jid, job in list(self.jobs.items()):
            if job.is_finished() and (job.finished_at or 0) < cutoff:
                self.jobs.pop(jid, None)
                if self.by_kind.get(job.kind) is job:
                    self.by_kind.pop(job.kind, None)

    def start(self, kind: str, worker: Callable[[Job], Any],
              exclusive: bool = True) -> Job:
        if exclusive and self.busy(kind):
            raise AppError(t("web.job_busy"), 409, "busy")
        self.cleanup()
        job = Job(kind=kind)
        job.loop = self.loop or asyncio.get_running_loop()
        # ContextVar 不会跟着 threading.Thread 走，这里把发起请求时的界面语言
        # 存进 Job，worker 线程里再套回去，否则任务日志永远是中文默认语言。
        job.lang = get_lang()
        self.jobs[job.id] = job
        self.by_kind[kind] = job

        def run() -> None:
            with language(job.lang):
                _run_job(job, kind, worker)

        job.thread = threading.Thread(target=run, name=f"job-{kind}-{job.id}",
                                      daemon=True)
        job.thread.start()
        return job

    def stop(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.is_finished():
            return False
        job.cancel_event.set()
        job.emit_threadsafe("log", {"text": t("web.job_stopping"), "level": "warn"})
        return True


def _run_job(job: Job, kind: str, worker: Callable[[Job], Any]) -> None:
    """worker 线程正文：跑任务并把状态/结果/异常翻译成事件。"""
    job.state = "running"
    job.started_at = time.time()
    job.emit_threadsafe("state", {"state": "running", "kind": kind, "id": job.id})
    try:
        result = worker(job)
        job.result = result
        if job.cancel_event.is_set():
            job.state = "cancelled"
            job.emit_threadsafe("cancelled", {"message": t("web.job_cancelled")})
        else:
            job.state = "done"
            job.emit_threadsafe("done", {"result": result})
    except InterruptedError as e:
        job.state = "cancelled"
        job.emit_threadsafe("cancelled", {"message": str(e) or t("web.job_cancelled")})
    except Exception as e:
        job.state = "error"
        job.error = str(e)
        job.emit_threadsafe("error", {
            "message": str(e),
            "trace": traceback.format_exc(limit=12),
        })
    finally:
        job.finished_at = time.time()
        job.emit_threadsafe("state", {"state": job.state, "kind": kind, "id": job.id})
        job.emit_threadsafe("end", {})


def _sse(ev: JobEvent) -> str:
    data = json.dumps(ev.data, ensure_ascii=False, default=str)
    return f"id: {ev.id}\nevent: {ev.event}\ndata: {data}\n\n"


async def sse_response(job: Job, request, last_id: int = 0):
    """把 Job 事件流转成 SSE。request 只需有 is_disconnected()。"""
    from fastapi.responses import StreamingResponse

    q = await job.subscribe(last_id)

    async def gen():
        try:
            yield (f"event: state\ndata: "
                   f"{json.dumps({'state': job.state, 'kind': job.kind, 'id': job.id}, ensure_ascii=False)}\n\n")
            # 任务刚结束时，threadsafe 事件可能还排在事件循环里没落进队列。
            # 先短暂等一拍，避免重连/晚订阅只收到 end 而丢掉 progress/done。
            if job.is_finished() and q.empty():
                try:
                    first = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield "event: end\ndata: {}\n\n"
                    return
                else:
                    yield _sse(first)
                    if first.event == "end":
                        return
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield _sse(ev)
                if ev.event == "end":
                    break
        finally:
            job.unsubscribe(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


job_manager = JobManager()
