"""极简 i18n：locales/{lang}.json，扁平 key-value。

- 进程默认语言用 `set_lang()`（Gradio 旧入口在启动时调用）。
- Web 入口用 `language(lang)` 上下文覆盖当前请求语言，不重启进程、互不影响。
- 缺键回退中文，中文再缺回退 key 本身（永不炸）。
- 带 {param} 占位，format 失败原样返回。
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from .paths import LOCALES_DIR

_CACHE: dict[str, dict] = {}
_DEFAULT_LANG = "zh"
_REQUEST_LANG: ContextVar[str | None] = ContextVar("unsloth_gui_lang", default=None)


def available_langs() -> list[str]:
    return ["zh", "en"]


def normalize_lang(lang: str | None) -> str:
    """未知值回退中文；`english` / `en-US` 都算英文。"""
    return "en" if str(lang or "").lower().startswith("en") else "zh"


def set_lang(lang: str) -> str:
    """设置进程默认语言（旧 Gradio 入口使用）。"""
    global _DEFAULT_LANG
    _DEFAULT_LANG = normalize_lang(lang)
    return _DEFAULT_LANG


def get_lang() -> str:
    """当前生效语言：请求上下文优先，其次进程默认。"""
    return _REQUEST_LANG.get() or _DEFAULT_LANG


@contextmanager
def language(lang: str | None) -> Iterator[str]:
    """在 with 块内临时切换语言，退出后恢复。"""
    token = _REQUEST_LANG.set(normalize_lang(lang))
    try:
        yield get_lang()
    finally:
        _REQUEST_LANG.reset(token)


def _load(lang: str) -> dict:
    if lang not in _CACHE:
        try:
            _CACHE[lang] = json.loads(
                (Path(LOCALES_DIR) / f"{lang}.json").read_text(encoding="utf-8")
            )
        except Exception:
            _CACHE[lang] = {}
    return _CACHE[lang]


def t(key: str, **kwargs) -> str:
    s = _load(get_lang()).get(key)
    if s is None:
        s = _load("zh").get(key, key)
    if kwargs:
        try:
            return s.format(**kwargs)
        except Exception:
            return s
    return s


def check_parity() -> tuple[bool, list[str]]:
    """中英 key 一致性检查，返回 (是否一致, 缺失描述列表)。"""
    zh, en = _load("zh"), _load("en")
    problems = [f"en missing: {k}" for k in zh if k not in en]
    problems += [f"zh missing: {k}" for k in en if k not in zh]
    return (not problems, problems)


def dictionary(lang: str | None = None) -> dict:
    """返回整份文案，供前端 `window.I18N` 使用。"""
    return dict(_load(normalize_lang(lang or get_lang())))
