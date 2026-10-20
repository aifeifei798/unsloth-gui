"""测试环境：把 UNSLOTH_GUI_ROOT 指到临时目录，绝不碰真实 models/datasets/输出。"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
TMP_ROOT = Path(tempfile.mkdtemp(prefix="ugui-tests-"))
os.environ["UNSLOTH_GUI_ROOT"] = str(TMP_ROOT)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _prepare_root() -> None:
    # 文案必须真实复制：t() 从 LOCALES_DIR 读
    shutil.copytree(REPO_ROOT / "locales", TMP_ROOT / "locales")
    _write_json(TMP_ROOT / "models.json", [
        {"display_name": "Test Base",
         "model_id": "unsloth/phi-3-mini-4k-instruct",
         "load_in_4bit": True, "dtype": None, "max_seq_length": 2048},
        {"display_name": "Local Missing",
         "model_id": "./definitely-missing-model",
         "load_in_4bit": True, "dtype": None, "max_seq_length": 2048,
         "source": "local"},
    ])

    raw = TMP_ROOT / "local_data" / "raw.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text('{"q": "q0", "a": "a0"}\n{"q": "q1", "a": "a1"}\n',
                   encoding="utf-8")
    _write_json(TMP_ROOT / "datasets_config" / "raw.json", {
        "display_name": "Raw Sample",
        "dataset_id": "./local_data/raw.jsonl",
        "is_local": True,
        "processed": False,
        "schema_version": 0,
        "prompt_template": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
        "input_columns": {"instruction": "instruction", "output": "output"},
    })

    artifact = TMP_ROOT / "local_data" / "processed" / "trainable" / "hf_dataset"
    artifact.mkdir(parents=True, exist_ok=True)
    _write_json(TMP_ROOT / "datasets_config" / "trainable.json", {
        "display_name": "Trainable Sample",
        "dataset_id": "./local_data/processed/trainable/hf_dataset",
        "is_local": True,
        "processed": True,
        "schema_version": 3,
        "prompt_template": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
        "input_columns": {"instruction": "instruction", "output": "output"},
    })

    # 坏文件只应产生 warning，不能拖垮整个列表
    (TMP_ROOT / "datasets_config" / "bad.json").write_text(
        '{"foo": 1}\n', encoding="utf-8")


_prepare_root()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from web.main import app

    with TestClient(app) as c:
        yield c


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
