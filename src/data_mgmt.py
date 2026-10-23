"""数据管理：统一数据的查看 / 样本预览 / 重命名 / 删除.

只删本工具生成的产物（local_data/processed/ 下的目录 + datasets_config/*.json），
原始上传文件和自带示例永远不动。重建/删除尽量保持原子性：
- 删除先删目录，失败就不动配置；
- 重命名先搬目录，配置写失败则搬回去。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from .config import (
    DatasetConfig,
    find_by_name,
    is_trainable_dataset,
    load_datasets_config,
    trainable_reason,
)
from .dataprep import DATASETS_CONFIG_DIR, PROCESSED_ROOT, sanitize_name
from .i18n import t
from .paths import is_under
from .storage import atomic_write_json


# 超过这个大小就不在详情页同步数行数（沿用 dataprep 预览的上限思路）
_COUNT_MAX_BYTES = 200 * 1024 * 1024


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
                data = json.load(f)
        except Exception:
            continue
        # 单文件里既可能是对象，也可能是数组（load_datasets_config 两种都认）
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("display_name") == display_name:
                return fp
    return None


def _artifact_dir(cfg: DatasetConfig) -> Optional[Path]:
    """本工具生成的数据目录（local_data/processed/<name>）。

    返回「够删一层」的目录本身，而不是 dataset_id 指向的更深层路径；
    dataset_id 直接指向 processed/<name>（没有 hf_dataset 子目录）时也能
    正常识别，避免删配置却留下孤儿目录。绝不会返回 processed 根目录。
    """
    try:
        p = Path(cfg.resolved_dataset_id()).resolve()
        rel = p.relative_to(PROCESSED_ROOT.resolve())
    except (OSError, ValueError):
        return None
    if not rel.parts:  # 就是 processed 根，拒绝
        return None
    root = PROCESSED_ROOT / rel.parts[0]
    return root if root.is_dir() else None


def entry_info(cfg: DatasetConfig) -> dict:
    """表格一行 + 详情共用的元信息（读 manifest/本地文件，不下载远端）。"""
    info: dict = {
        "name": cfg.display_name,
        "trainable": is_trainable_dataset(cfg),
        "reason": trainable_reason(cfg),
        "schema_version": cfg.schema_version,
        "processed": cfg.processed,
        "source_file": cfg.source_file,
    }
    art = _artifact_dir(cfg)
    if art is not None:
        man_path = art / "manifest.json"
        try:
            with open(man_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            info["rows"] = str(man.get("rows_kept", "?"))
            info["created"] = man.get("created", "—")
        except Exception:
            info["rows"] = "?"
            info["created"] = "—"
        info["source"] = t("dm.src.unified")
        try:
            info["size"] = _human_size(_dir_size(art))
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
            # 大文件不数行：这是同步接口，几千 MB 扫一遍会把设置页拖住
            if p.stat().st_size > _COUNT_MAX_BYTES:
                info["rows"] = t("dm.rows.remote")
            else:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
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


def list_entries(warnings: list[str] | None = None) -> list[dict]:
    try:
        cfgs = load_datasets_config(warnings=warnings)
    except Exception:
        return []
    return [entry_info(c) for c in cfgs]


def detail_text(name: str) -> str:
    try:
        cfgs = load_datasets_config(warnings=[])
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
    if info.get("reason"):
        lines.append(t("dm.detail.reason", v=info["reason"]))
    art = _artifact_dir(cfg)
    if art is not None:
        try:
            with open(art / "manifest.json", "r", encoding="utf-8") as f:
                man = json.load(f)
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
        cfg = find_by_name(load_datasets_config(warnings=[]), name)
    except Exception as e:
        return t("dm.err.config", err=e)
    if cfg is None:
        return t("dm.err.notfound", name=name)
    return dataset_preview_text(cfg, n=n)


def delete_entry_strict(name: str) -> dict:
    """删除配置 + 本工具产物。失败抛异常，由调用方决定怎么展示。"""
    fp = _find_config_file(name)
    if fp is None:
        raise ValueError(t("dm.err.nofile_plain", name=name))
    try:
        cfgs = load_datasets_config(warnings=[])
    except Exception as e:
        raise ValueError(t("dm.err.config_plain", err=e)) from e
    cfg = find_by_name(cfgs, name)
    removed_dirs: list[str] = []
    if cfg is not None:
        target = _artifact_dir(cfg)
        if target is not None:
            if target == PROCESSED_ROOT or not is_under(target, PROCESSED_ROOT):
                raise ValueError(t("dm.err.refuse_root"))
            try:
                shutil.rmtree(target)
            except OSError as e:
                raise ValueError(
                    t("dm.err.rmtree", dir=target.name, err=e)
                ) from e
            removed_dirs.append(target.name)
    try:
        fp.unlink(missing_ok=True)
    except OSError as e:
        raise ValueError(t("dm.err.unlink", file=fp.name, err=e)) from e
    return {"name": name, "config_file": fp.name, "removed_dirs": removed_dirs}


def rename_entry(old: str, new: str) -> str:
    """返回新展示名。产物目录跟着搬家，配置与 manifest 同步更新。"""
    new = sanitize_name(new)
    if not old:
        raise ValueError(t("dm.err.noselect"))
    if new == old:
        raise ValueError(t("dm.err.same"))
    try:
        cfgs = load_datasets_config(warnings=[])
    except Exception as e:
        raise ValueError(t("dm.err.config_plain", err=e)) from e
    cfg = find_by_name(cfgs, old)
    if cfg is None:
        raise ValueError(t("dm.err.notfound_plain", name=old))
    if find_by_name(cfgs, new) is not None:
        raise ValueError(t("dm.err.used", new=new))
    fp = _find_config_file(old)
    if fp is None:
        raise ValueError(t("dm.err.nofile_plain", name=old))
    new_fp = DATASETS_CONFIG_DIR / f"{new}.json"
    if new_fp.exists():
        raise ValueError(t("dm.err.file_exists", file=new_fp.name))

    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    art = _artifact_dir(cfg)
    target = None
    if art is not None:
        target = PROCESSED_ROOT / new
        if target.exists():
            raise ValueError(t("dm.err.dir_exists", dir=new))

    moved = False
    if art is not None and target is not None:
        art.rename(target)
        moved = True
        try:
            man_path = target / "manifest.json"
            with open(man_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            man["name"] = new
            atomic_write_json(man_path, man)
        except Exception:
            pass  # manifest 坏了不阻断改名，配置才是权威
        data["dataset_id"] = f"./local_data/processed/{new}/hf_dataset"

    data["display_name"] = new
    try:
        atomic_write_json(new_fp, data)
    except Exception:
        if moved and target is not None:
            try:
                target.rename(PROCESSED_ROOT / art.name)
            except OSError:
                pass
        raise
    try:
        fp.unlink(missing_ok=True)
    except OSError as e:
        raise ValueError(t("dm.err.unlink", file=fp.name, err=e)) from e
    return new
