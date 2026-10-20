"""命令行入口：python -m web [--host 127.0.0.1] [--port 7861] [--reload]."""
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsloth GUI (FastAPI + DaisyUI)")
    parser.add_argument("--host", default=os.environ.get("UNSLOTH_WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("UNSLOTH_WEB_PORT", "7861")))
    parser.add_argument("--reload", action="store_true",
                        help="开发模式自动重载（仅单 worker）")
    args = parser.parse_args()
    uvicorn.run("web.main:app", host=args.host, port=args.port,
                reload=bool(args.reload), workers=1, log_level="info")


if __name__ == "__main__":
    main()
