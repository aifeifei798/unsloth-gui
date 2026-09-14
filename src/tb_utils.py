"""TensorBoard 生命周期：端口检测、非阻塞启动、退出清理."""
from __future__ import annotations

import atexit
import socket
import subprocess
from typing import Optional

from .config import PROJECT_ROOT

LOGS_PARENT_DIR = PROJECT_ROOT / "logs"
_PROC: Optional[subprocess.Popen] = None
_PORT: int = 6006


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
        return True, f"TensorBoard 已在运行 (:{_PORT})", _PORT
    LOGS_PARENT_DIR.mkdir(parents=True, exist_ok=True)
    if _port_in_use(port):
        # 可能是外部已启动的 TB，直接复用
        _PORT = port
        return True, f"端口 {port} 已被占用，复用现有 TensorBoard。", port
    free = _find_free_port(port)
    try:
        _PROC = subprocess.Popen(
            ["tensorboard", "--logdir", str(LOGS_PARENT_DIR),
             "--host", "127.0.0.1", "--port", str(free)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False, "未找到 tensorboard 可执行文件，请 pip install tensorboard。", free
    except Exception as e:
        return False, f"TensorBoard 启动失败: {e}", free
    _PORT = free
    return True, f"TensorBoard 已启动 (:{free})", free


def stop_tensorboard() -> None:
    global _PROC
    if _PROC is not None:
        try:
            _PROC.terminate()
        except Exception:
            pass
        _PROC = None


atexit.register(stop_tensorboard)
