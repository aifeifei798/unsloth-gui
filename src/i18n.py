"""极简 i18n：locales/{lang}.json，扁平 key-value。

- 缺键回退中文，中文再缺回退 key 本身（永不炸）。
- 带 {param} 占位，format 失败原样返回。
- 语言在进程启动时确定（Gradio 静态构建），UI 内切换走进程重启。
"""
from __future__ import annotations

import json
from pathlib import Path

LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"

_LANG = "zh"
_CACHE: dict[str, dict] = {}


def available_langs() -> list[str]:
    return ["zh", "en"]


def set_lang(lang: str) -> str:
    """规范化并设置语言，返回生效值。未知值回退中文。"""
    global _LANG
    _LANG = "en" if str(lang or "").lower().startswith("en") else "zh"
    return _LANG


def get_lang() -> str:
    return _LANG


def _load(lang: str) -> dict:
    if lang not in _CACHE:
        try:
            _CACHE[lang] = json.loads((LOCALES_DIR / f"{lang}.json").read_text(encoding="utf-8"))
        except Exception:
            _CACHE[lang] = {}
    return _CACHE[lang]


def t(key: str, **kwargs) -> str:
    s = _load(_LANG).get(key)
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
