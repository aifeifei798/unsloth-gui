"""数据管理 API：列表 / 详情 / 样本 / 重命名 / 删除。"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from src.config import (
    DatasetConfig,
    find_by_name,
    is_trainable_dataset,
    load_datasets_config,
    trainable_reason,
)
from src.data_mgmt import delete_entry_strict, detail_text, entry_info, rename_entry, sample_text
from src.i18n import t

from ..errors import AppError, bad, not_found
from ..schemas import RenameIn

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _load(warnings: list[str]) -> list[DatasetConfig]:
    try:
        return load_datasets_config(warnings=warnings)
    except FileNotFoundError:
        warnings.append(t("warn.datasets", err=t("cfg.ds_not_found",
                                                 path="datasets_config")))
        return []
    except Exception as e:
        raise AppError(t("warn.datasets", err=e), 500, "config") from e


def _payload(cfg: DatasetConfig) -> dict:
    info = dict(entry_info(cfg))
    info.update({
        "dataset_id": cfg.dataset_id,
        "split": cfg.split,
        "is_local": cfg.is_local,
        "processed": cfg.processed,
        "schema_version": cfg.schema_version,
        "prompt_template": cfg.prompt_template,
        "input_columns": cfg.input_columns,
        "recommended_params": cfg.recommended_params or {},
        "trainable": is_trainable_dataset(cfg),
        "reason": trainable_reason(cfg),
    })
    return info


@router.get("")
async def list_datasets(scope: str = Query("all")):
    warnings: list[str] = []
    cfgs = _load(warnings)
    if scope == "trainable":
        cfgs = [c for c in cfgs if is_trainable_dataset(c)]
    return {"ok": True, "data": [_payload(c) for c in cfgs],
            "warnings": warnings}


@router.get("/detail")
async def dataset_detail(name: str = Query(...)):
    warnings: list[str] = []
    cfgs = _load(warnings)
    if find_by_name(cfgs, name) is None:
        raise not_found(t("dm.err.notfound_plain", name=name))
    return {"ok": True, "data": {"detail": detail_text(name)},
            "warnings": warnings}


@router.get("/sample")
async def dataset_sample(name: str = Query(...), n: int = Query(2, ge=1, le=10)):
    text = await asyncio.to_thread(sample_text, name, n)
    if text.startswith("❌"):
        raise bad(text)
    return {"ok": True, "data": {"sample": text}}


@router.post("/rename")
async def rename_dataset(body: RenameIn):
    try:
        new = await asyncio.to_thread(rename_entry, body.old, body.new)
    except ValueError as e:
        raise bad(str(e)) from e
    except Exception as e:
        raise AppError(t("d.err.rename", err=e), 500, "rename") from e
    return {"ok": True, "data": {"old": body.old, "new": new},
            "message": t("d.renamed", new=new)}


@router.delete("")
async def delete_dataset(name: str = Query(...)):
    try:
        info = await asyncio.to_thread(delete_entry_strict, name)
    except ValueError as e:
        raise bad(str(e)) from e
    except Exception as e:
        raise AppError(t("dm.err.delete_failed", err=e), 500, "delete") from e
    removed = info.get("removed_dirs") or []
    msg = t("dm.deleted", name=name, parts=t("dm.del.config",
                                             file=info["config_file"]),
            kept="" if removed else t("dm.del.kept"))
    return {"ok": True, "data": info, "message": msg}
