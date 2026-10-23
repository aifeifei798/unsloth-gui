from __future__ import annotations

import asyncio
import time

from src.i18n import language, t

from web.jobs import Job, JobManager, sse_response


class FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


def test_sse_race_finished_before_subscribe():
    async def main():
        manager = JobManager()
        manager.bind_loop()

        def worker(job):
            for i in range(3):
                job.emit_threadsafe("progress", {"frac": (i + 1) / 3})
                time.sleep(0.01)
            return {"value": 42}

        job = manager.start("test", worker)
        response = await sse_response(job, FakeRequest())
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
            if "event: end" in chunk:
                break
        text = "".join(chunks)
        assert "event: progress" in text
        assert "event: done" in text
        assert '"value": 42' in text
        assert job.state == "done"
        assert job.result == {"value": 42}

    asyncio.run(main())


def test_cancel_sets_event_and_state():
    async def main():
        manager = JobManager()
        manager.bind_loop()
        started = time.time()

        def worker(job):
            while not job.cancel_event.is_set():
                if time.time() - started > 5:
                    raise RuntimeError("timeout")
                time.sleep(0.01)
            raise InterruptedError("stopped")

        job = manager.start("test", worker)
        await asyncio.sleep(0.05)
        assert manager.stop(job.id) is True
        for _ in range(100):
            if job.is_finished():
                break
            await asyncio.sleep(0.02)
        assert job.state == "cancelled"

    asyncio.run(main())


def test_job_worker_inherits_request_language():
    """worker 是裸线程，ContextVar 不继承；必须靠 Job.lang 还原，否则日志恒中文."""

    async def main():
        manager = JobManager()
        manager.bind_loop()
        with language("en"):
            job = manager.start("lang", lambda j: t("web.job_busy"))
        for _ in range(200):
            if job.is_finished():
                break
            await asyncio.sleep(0.01)
        assert job.state == "done"
        assert job.result.startswith("Another job")

    asyncio.run(main())


def test_full_subscriber_queue_keeps_latest_event():
    """慢客户端积压时丢最旧的，但 end/done 这类终止事件不能被丢掉."""
    job = Job(kind="x")
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    job.subscribers.add(q)
    job.emit("log", {"i": 1})
    job.emit("end", {})
    assert q.get_nowait().event == "end"


def test_busy_only_per_kind():
    async def main():
        manager = JobManager()
        manager.bind_loop()

        def slow(job):
            time.sleep(0.1)
            return 1

        a = manager.start("train", slow)
        b = manager.start("prepare", slow)
        assert a.id != b.id
        assert manager.busy("train") is True
        assert manager.busy("prepare") is True
        try:
            manager.start("train", slow)
        except Exception as e:
            assert "running" in str(e).lower() or "运行" in str(e)
        else:
            raise AssertionError("duplicate train job should be rejected")
        await asyncio.sleep(0.2)
        assert a.is_finished() and b.is_finished()

    asyncio.run(main())
