"""数据管理：统一数据的查看 / 样本预览 / 重命名 / 删除.

只删本工具生成的产物（local_data/processed/ 下的目录 + datasets_config/*.json），
原始上传文件和自带示例 jsonl 永远不动。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from .config import DatasetConfig, find_by_name, load_datasets_config
from .dataprep import DATASETS_CONFIG_DIR, PROCESSED_ROOT, sanitize_name
from .i18n import t


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}GB"


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _find_config_file(display_name: str) -> Optional[Path]:
    """按展示名反查 datasets_config 下的实际文件名（文件名不一定等于展示名）."""
    if not DATASETS_CONFIG_DIR.is_dir():
        return None
    for fp in sorted(DATASETS_CONFIG_DIR.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                if json.load(f).get("display_name") == display_name:
                    return fp
        except Exception:
            continue
    return None


def _artifact_dir(cfg: DatasetConfig) -> Optional[Path]:
    """本工具生成的产物目录（local_data/processed/<name>/），不是则返回 None 不碰."""
    try:
        p = Path(cfg.resolved_dataset_id()).resolve()
    except Exception:
        return None
    try:
        p.relative_to(PROCESSED_ROOT.resolve())
        return p if p.exists() else None
    except ValueError:
        return None


def entry_info(cfg: DatasetConfig) -> dict:
    """表格一行 + 详情共用的元信息（读 manifest/本地文件，不下载远端）。"""
    info: dict = {"name": cfg.display_name, "trainable": bool(cfg.processed)}
    art = _artifact_dir(cfg)
    if art is not None:
        man_path = art.parent / "manifest.json"
        try:
            man = json.load(open(man_path, "r", encoding="utf-8"))
            info["rows"] = str(man.get("rows_kept", "?"))
            info["created"] = man.get("created", "—")
        except Exception:
            info["rows"] = "?"
            info["created"] = "—"
        info["source"] = t("dm.src.unified")
        try:
            info["size"] = _human_size(_dir_size(art.parent))
        except Exception:
            info["size"] = "—"
        return info
    # 非产物：本地文件数行数，远端不拉取
    try:
        p = Path(cfg.resolved_dataset_id())
    except Exception:
        p = None
    if p is not None and p.is_file() and p.suffix in (".jsonl", ".json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
            info["rows"] = str(n)
        except Exception:
            info["rows"] = "?"
        info["source"] = t("dm.src.local_file")
        try:
            info["size"] = _human_size(p.stat().st_size)
        except Exception:
            info["size"] = "—"
    elif p is not None and p.is_dir():
        info["source"] = t("dm.src.local_dir")
        try:
            from datasets import load_from_disk
            info["rows"] = str(len(load_from_disk(str(p))))
        except Exception:
            info["rows"] = "?"
        try:
            info["size"] = _human_size(_dir_size(p))
        except Exception:
            info["size"] = "—"
    else:
        info["source"] = t("dm.src.hf") if not cfg.is_local else t("dm.src.local_missing")
        info["rows"] = t("dm.rows.remote")
        info["size"] = "—"
    info["created"] = "—"
    return info


def list_entries() -> list[dict]:
    try:
        return [entry_info(c) for c in load_datasets_config()]
    except Exception:
        return []


def detail_text(name: str) -> str:
    try:
        cfgs = load_datasets_config()
    except Exception as e:
        return t("dm.err.config", err=e)
    cfg = find_by_name(cfgs, name)
    if cfg is None:
        return t("dm.err.notfound", name=name)
    info = entry_info(cfg)
    lines = [
        t("dm.detail.name", v=info["name"]),
        t("dm.detail.meta", source=info["source"], rows=info["rows"], size=info["size"]),
        t("dm.detail.trainable_yes") if info["trainable"] else t("dm.detail.trainable_no"),
        t("dm.detail.addr", v=cfg.dataset_id),
        "-" * 50,
    ]
    art = _artifact_dir(cfg)
    if art is not None:
        try:
            man = json.load(open(art.parent / "manifest.json", "r", encoding="utf-8"))
            mp = man.get("mapping", {})
            lines.append(t("dm.detail.mapping",
                           v=" + ".join(t("dm.detail.map_role", k=k, n=len(v))
                                         for k, v in mp.items() if v)))
            if man.get("fixed_instruction"):
                lines.append(t("dm.detail.fixed", v=man["fixed_instruction"][:200]))
            lines.append(t("dm.detail.rows", total=man.get("rows_total"),
                           kept=man.get("rows_kept"))
                         + (t("dm.detail.dropped",
                              n=man.get("rows_dropped_empty_output"))
                            if man.get("rows_dropped_empty_output") else ""))
            lines.append(t("dm.detail.created", v=man.get("created", "—")))
        except Exception:
            pass
    else:
        lines.append(t("dm.detail.template"))
        lines.append((cfg.prompt_template[:500] + "…") if len(cfg.prompt_template) > 500
                     else cfg.prompt_template)
        lines.append(t("dm.detail.columns", v=cfg.input_columns))
    return "\n".join(lines)


def sample_text(name: str, n: int = 2) -> str:
    """点按钮才看样本（远端数据会触发下载，所以不自动看）。"""
    from .dataset_utils import dataset_preview_text
    try:
        cfg = find_by_name(load_datasets_config(), name)
    except Exception as e:
        return t("dm.err.config", err=e)
    if cfg is None:
        return t("dm.err.notfound", name=name)
    return dataset_preview_text(cfg, n=n)


def delete_entry(name: str) -> str:
    fp = _find_config_file(name)
    if fp is None:
        return t("dm.err.nofile", name=name)
    try:
        cfg = find_by_name(load_datasets_config(), name)
    except Exception as e:
        return t("dm.err.config", err=e)
    removed = [t("dm.del.config", file=fp.name)]
    art = _artifact_dir(cfg) if cfg is not None else None
    if art is not None:
        shutil.rmtree(art.parent, ignore_errors=True)
        removed.append(t("dm.del.artifact", dir=art.parent.name))
    fp.unlink(missing_ok=True)
    kept = t("dm.del.kept") if art is None else ""
    return t("dm.deleted", name=name, parts=" + ".join(removed), kept=kept).strip()


def rename_entry(old: str, new: str) -> str:
    """返回新展示名。产物目录跟着搬家，配置地址同步改。"""
    new = sanitize_name(new)
    if not old:
        raise ValueError(t("dm.err.noselect"))
    if new == old:
        raise ValueError(t("dm.err.same"))
    try:
        cfgs = load_datasets_config()
    except Exception as e:
        raise ValueError(t("dm.err.config_plain", err=e))
    if find_by_name(cfgs, old) is None:
        raise ValueError(t("dm.err.notfound_plain", name=old))
    if find_by_name(cfgs, new) is not None:
        raise ValueError(t("dm.err.used", new=new))
    fp = _find_config_file(old)
    if fp is None:
        raise ValueError(t("dm.err.nofile", name=old))
    new_fp = DATASETS_CONFIG_DIR / f"{new}.json"
    if new_fp.exists():
        raise ValueError(t("dm.err.file_exists", file=new_fp.name))
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = find_by_name(cfgs, old)
    art = _artifact_dir(cfg) if cfg is not None else None
    if art is not None:
        target = PROCESSED_ROOT / new
        if target.exists():
            raise ValueError(t("dm.err.dir_exists", dir=new))
        art.parent.rename(target)
        data["dataset_id"] = f"./local_data/processed/{new}/hf_dataset"
        try:
            man_path = target / "manifest.json"
            man = json.load(open(man_path, "r", encoding="utf-8"))
            man["name"] = new
            json.dump(man, open(man_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        except Exception:
            pass
    data["display_name"] = new
    with open(new_fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    fp.unlink(missing_ok=True)
    return new
