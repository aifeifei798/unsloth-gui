from __future__ import annotations

import json

from src.config import (
    SCHEMA_VERSION,
    DatasetConfig,
    is_trainable_dataset,
    load_datasets_config,
    trainable_reason,
)
from src.dataprep import format_unified_row, resolve_mapping
from src.dataset_utils import drop_empty_sections
from src.i18n import check_parity
from src.inference_utils import _headers_from_template, _render_prompt


def test_schema_version_is_three():
    assert SCHEMA_VERSION == 3


def _cfg(**kw):
    base = dict(
        display_name="d", dataset_id="./x", is_local=True,
        prompt_template="### Instruction:\n{instruction}\n\n### Response:\n{output}",
        input_columns={"instruction": "instruction", "output": "output"},
    )
    base.update(kw)
    return DatasetConfig(**base)


def test_trainable_requires_processed_schema_and_existing_path(tmp_path):
    art = tmp_path / "hf_dataset"
    art.mkdir()
    good = _cfg(processed=True, schema_version=3, dataset_id=str(art))
    assert is_trainable_dataset(good) is True
    assert trainable_reason(good) == ""

    assert is_trainable_dataset(_cfg(processed=False, schema_version=3,
                                     dataset_id=str(art))) is False
    assert is_trainable_dataset(_cfg(processed=True, schema_version=2,
                                     dataset_id=str(art))) is False
    missing = _cfg(processed=True, schema_version=3,
                   dataset_id=str(tmp_path / "gone"))
    assert is_trainable_dataset(missing) is False
    assert trainable_reason(missing)


def test_tolerant_dataset_loading(tmp_path):
    (tmp_path / "good.json").write_text(json.dumps({
        "display_name": "good", "dataset_id": "./x",
        "prompt_template": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
        "input_columns": {"instruction": "instruction", "output": "output"},
    }), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "badshape.json").write_text('{"foo": 1}', encoding="utf-8")
    warnings: list[str] = []
    cfgs = load_datasets_config(tmp_path, warnings=warnings)
    assert [c.display_name for c in cfgs] == ["good"]
    assert len(warnings) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_datasets_config(empty, warnings=[]) == []


def test_drop_empty_sections():
    text = "### Instruction:\nq\n\n### Input:\n\n\n### Response:\nr"
    out = drop_empty_sections(text)
    assert "### Input:" not in out
    assert "q" in out and "r" in out


def test_resolve_mapping_and_format():
    columns = ["q", "ctx", "a"]
    mapping, fixed = resolve_mapping(columns, ["q"], ["ctx"], [], ["a"], "")
    row = {"q": "Q", "ctx": "C", "a": "A"}
    unified = format_unified_row(row, mapping, fixed)
    assert unified == {"instruction": "Q", "input": "C", "think": "", "output": "A"}


def test_inference_prompt_headers_and_multiturn():
    tpl = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    ins, resp, has_input = _headers_from_template(tpl)
    assert ins == "### Instruction:"
    assert resp == "### Response:"
    assert has_input is True
    prompt = _render_prompt(
        [{"role": "user", "content": "u1"},
         {"role": "assistant", "content": "a1"}],
        "u2", "sys", ins, resp, has_input)
    assert "### Input:\nsys" in prompt
    assert "### Instruction:\nu1" in prompt
    assert "### Response:\na1" in prompt
    assert prompt.rstrip().endswith("### Response:")


def test_inference_prompt_without_input_section():
    tpl = "### Instruction:\n{instruction}\n\n### Response:\n{output}"
    ins, resp, has_input = _headers_from_template(tpl)
    prompt = _render_prompt([], "hi", "sys", ins, resp, has_input)
    assert "sys" in prompt
    assert "### Input:" not in prompt
    assert prompt.rstrip().endswith("### Response:")


def test_locale_parity():
    ok, problems = check_parity()
    assert ok, problems[:5]
