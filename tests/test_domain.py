from __future__ import annotations

import json
import os

from src.config import (
    SCHEMA_VERSION,
    DatasetConfig,
    is_trainable_dataset,
    load_datasets_config,
    trainable_reason,
)
from src.dataprep import format_unified_row, resolve_mapping
from src.dataset_utils import (
    chat_messages_from_row,
    drop_empty_sections,
    has_chat_template,
    render_chat_text,
    supports_chat_roles,
)
from src.i18n import check_parity
from src.inference_utils import _build_chat_prompt, _headers_from_template, _render_prompt
from src.model_utils import configure_hf_transfer, sanitize_proxy_env, unwrap_text_tokenizer


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


class _FakeChatTokenizer:
    """极简 chat 模板 tokenizer：不依赖 transformers。"""

    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        out = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            out += "<|assistant|>"
        return out

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": list(range(max(1, len(text) // 4)))}


def test_chat_messages_from_row_merges_input():
    row = {"instruction": "问", "input": "上下文", "output": "<think>想</think>\n答"}
    mapping = {"instruction": "instruction", "input": "input", "output": "output"}
    msgs = chat_messages_from_row(row, mapping)
    assert msgs == [
        {"role": "user", "content": "问\n\n上下文"},
        {"role": "assistant", "content": "<think>想</think>\n答"},
    ]


def test_chat_messages_from_row_requires_output():
    mapping = {"instruction": "instruction", "output": "output"}
    try:
        chat_messages_from_row({"instruction": "问", "output": "  "}, mapping)
    except ValueError as e:
        assert "output" in str(e).lower() or "输出" in str(e)
    else:
        raise AssertionError("空输出应当报错")


def test_render_chat_text_uses_model_template():
    tok = _FakeChatTokenizer()
    assert has_chat_template(tok) is True
    assert has_chat_template(object()) is False
    msgs = [{"role": "user", "content": "hi"}]
    assert render_chat_text(tok, msgs) == "<|user|>hi"
    assert render_chat_text(tok, msgs, add_generation_prompt=True) == \
        "<|user|>hi<|assistant|>"


def test_supports_chat_roles():
    assert supports_chat_roles(_cfg()) is True
    assert supports_chat_roles(_cfg(input_columns={"output": "output"})) is False


def test_build_chat_prompt_ends_with_generation_prompt():
    tok = _FakeChatTokenizer()
    prompt, out_tokens = _build_chat_prompt(tok, [], "hello", "", 2048, 64)
    assert prompt == "<|user|>hello<|assistant|>"
    assert out_tokens == 64
    prompt, _ = _build_chat_prompt(tok, [], "hello", "sys", 2048, 64)
    assert prompt.startswith("<|system|>sys")


def test_configure_hf_transfer(monkeypatch):
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "auto")
    configure_hf_transfer(False)
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    configure_hf_transfer(True)
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


def test_sanitize_proxy_env_drops_unsupported_scheme(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:10808")
    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:10808")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:10808")
    dropped = sanitize_proxy_env()
    assert dropped, "socks:// 不该被 httpx 使用，必须清掉"
    assert "ALL_PROXY" not in os.environ
    assert "all_proxy" not in os.environ
    # http_proxy 得留着，httpx 靠它走代理
    assert os.environ["http_proxy"] == "http://127.0.0.1:10808"


def test_sanitize_proxy_env_keeps_http_proxy(monkeypatch):
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:10808")
    monkeypatch.delenv("all_proxy", raising=False)
    assert sanitize_proxy_env() == []
    assert os.environ["ALL_PROXY"] == "http://127.0.0.1:10808"


def test_unwrap_text_tokenizer_from_processor():
    class Tok:
        def encode(self, text):
            return [1]

    class Processor:
        tokenizer = Tok()

    tok = Tok()
    assert unwrap_text_tokenizer(tok) is tok
    assert unwrap_text_tokenizer(Processor()) is Processor.tokenizer


def test_locale_parity():
    ok, problems = check_parity()
    assert ok, problems[:5]
