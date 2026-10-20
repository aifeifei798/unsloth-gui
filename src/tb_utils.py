"""TensorBoard 生命周期：端口检测、非阻塞启动、退出清理."""
from __future__ import annotations

import atexit
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .i18n import t
from .paths import LOGS_PARENT_DIR

_PROC: Optional[subprocess.Popen] = None
_PORT: int = 6006


def _tb_command() -> list[str]:
    """定位 tensorboard 的启动命令。

    服务是用 venv 的 python 直接起的（不激活 venv），PATH 里通常没有
    venv/bin，所以直接找 "tensorboard" 会 FileNotFoundError。这里依次尝试：
    当前解释器同目录 → PATH → `python -m tensorboard.main`（最稳，不依赖 PATH）。
    """
    exe = Path(sys.executable).with_name("tensorboard")
    if exe.is_file() and os.access(exe, os.X_OK):
        return [str(exe)]
    found = shutil.which("tensorboard")
    if found:
        return [found]
    return [sys.executable, "-m", "tensorboard.main"]


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _find_free_port(start: int = 6006, tries: int = 20) -> int:
    for p in range(start, start + tries):
        if not _port_in_use(p):
            return p
    return start


def launch_tensorboard(port: int = 6006) -> tuple[bool, str, int]:
    """返回 (ok, message, port). 已在运行则复用；端口被占则自动顺延."""
    global _PROC, _PORT
    if _PROC is not None and _PROC.poll() is None:
        return True, t("tb.running", port=_PORT), _PORT
    LOGS_PARENT_DIR.mkdir(parents=True, exist_ok=True)
    if _port_in_use(port):
        # 可能是外部已启动的 TB，直接复用
        _PORT = port
        return True, t("tb.reused", port=port), port
    free = _find_free_port(port)
    try:
        _PROC = subprocess.Popen(
            _tb_command() + ["--logdir", str(LOGS_PARENT_DIR),
                             "--host", "127.0.0.1", "--port", str(free)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, t("tb.no_bin"), free
    except Exception as e:
        return False, t("tb.fail", err=e), free
    _PORT = free
    return True, t("tb.started", port=free), free


def stop_tensorboard() -> None:
    global _PROC
    if _PROC is not None:
        try:
            _PROC.terminate()
        except Exception:
            pass
        _PROC = None


atexit.register(stop_tensorboard)
