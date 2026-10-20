"""FastAPI 入口：中间件、页面路由、异常格式、静态资源。"""
from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from src.i18n import dictionary, get_lang, language, normalize_lang, t
from src.model_utils import sanitize_proxy_env
from src.paths import ensure_runtime_dirs
from src.tb_utils import stop_tensorboard

from . import __version__
from .errors import AppError
from .jobs import job_manager
from .routers import datasets, inference, models, prepare, system, train

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals.update(t=t, get_lang=get_lang, version=__version__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_runtime_dirs()
    # 环境里常有 httpx 认不出的 socks:// 代理，下载模型会直接报错，这里先清掉
    dropped = sanitize_proxy_env()
    if dropped:
        print("[proxy] 已忽略不兼容的代理变量: " + ", ".join(dropped))
    job_manager.bind_loop()
    yield
    stop_tensorboard()


class LangMiddleware(BaseHTTPMiddleware):
    """按 Cookie / ?lang= 给每个请求设置语言，不重启进程。"""

    async def dispatch(self, request: Request, call_next):
        raw = (request.cookies.get("ugui_lang")
               or request.query_params.get("lang") or "zh")
        code = normalize_lang(raw)
        with language(code):
            response = await call_next(request)
        if (not request.url.path.startswith("/api/lang/")
                and (request.cookies.get("ugui_lang") != code
                     or request.query_params.get("lang"))):
            response.set_cookie("ugui_lang", code, max_age=365 * 24 * 3600,
                                samesite="lax")
        return response


app = FastAPI(title="Unsloth GUI", version=__version__, lifespan=lifespan,
              docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
app.add_middleware(LangMiddleware)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

for router in (system.router, models.router, datasets.router, prepare.router,
               train.router, inference.router):
    app.include_router(router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"ok": False, "error": {"message": exc.message, "code": exc.code}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
    msg = f"{loc}: {first.get('msg', 'invalid')}" if loc else str(first.get("msg", "invalid"))
    return JSONResponse(
        status_code=422,
        content={"ok": False, "error": {"message": msg, "code": "validation",
                                        "details": errors}},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": {
            "message": str(exc) or exc.__class__.__name__,
            "code": "internal",
        }},
    )


def _render(request: Request, page: str, active: str):
    code = get_lang()
    return templates.TemplateResponse(
        request=request,
        name=f"pages/{page}.html",
        context={
            "active": active,
            "i18n": dictionary(code),
            "lang_code": code,
        },
    )


@app.get("/")
async def root():
    return RedirectResponse("/prepare")


@app.get("/settings")
async def settings_page(request: Request):
    return _render(request, "settings", "settings")


# 模型 / 数据都搬进设置页了，老链接自动跳过去
@app.get("/models")
async def models_page(request: Request):
    return RedirectResponse("/settings")


@app.get("/datasets")
async def datasets_page(request: Request):
    return RedirectResponse("/settings")


@app.get("/prepare")
async def prepare_page(request: Request):
    return _render(request, "prepare", "prepare")


@app.get("/train")
async def train_page(request: Request):
    return _render(request, "train", "train")


@app.get("/chat")
async def chat_page(request: Request):
    return _render(request, "chat", "chat")
