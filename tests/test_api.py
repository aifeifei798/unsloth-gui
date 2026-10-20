from __future__ import annotations

import json
from pathlib import Path

from src.paths import DATASETS_CONFIG_DIR


def _write_dataset(name: str, **overrides) -> Path:
    data = {
        "display_name": name,
        "dataset_id": "./local_data/raw.jsonl",
        "is_local": True,
        "processed": False,
        "schema_version": 0,
        "prompt_template": "### Instruction:\n{instruction}\n\n### Response:\n{output}",
        "input_columns": {"instruction": "instruction", "output": "output"},
    }
    data.update(overrides)
    path = DATASETS_CONFIG_DIR / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_health_and_pages(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    for path in ("/models", "/datasets", "/prepare", "/train", "/chat"):
        r = client.get(path)
        assert r.status_code == 200
        assert "Unsloth GUI" in r.text
        assert "window.I18N" in r.text


def test_root_redirects_to_models(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"].endswith("/models")


def test_language_cookie_only_one(client):
    r = client.get("/api/lang/en")
    assert r.status_code == 200
    assert r.json()["data"]["message"] == "Language changed"
    cookies = r.headers.get_list("set-cookie")
    assert len(cookies) == 1 and "ugui_lang=en" in cookies[0]
    r = client.get("/models")
    assert "Models" in r.text


def test_i18n_payload(client):
    r = client.get("/api/i18n/zh")
    assert r.status_code == 200
    assert r.json()["data"]["messages"]["web.nav.models"] == "模型"
    r = client.get("/api/i18n/en")
    assert r.json()["data"]["messages"]["web.nav.models"] == "Models"


def test_models_crud(client):
    body = {
        "display_name": "API Model",
        "model_id": "unsloth/phi-3-mini-4k-instruct",
        "source": "huggingface",
        "load_in_4bit": True,
        "fast_download": True,
        "dtype": None,
        "max_seq_length": 2048,
        "chat_template": None,
    }
    r = client.post("/api/models", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["data"]["fast_download"] is True
    r = client.post("/api/models", json=body)
    assert r.status_code == 409

    body2 = dict(body, original_name="API Model", display_name="API Model 2")
    r = client.put("/api/models", json=body2)
    assert r.status_code == 200, r.text

    r = client.get("/api/models")
    names = [m["display_name"] for m in r.json()["data"]]
    assert "API Model 2" in names and "API Model" not in names

    r = client.delete("/api/models", params={"name": "API Model 2"})
    assert r.status_code == 200
    r = client.delete("/api/models", params={"name": "API Model 2"})
    assert r.status_code == 404


def test_models_reject_missing_local_and_bad_seq(client):
    r = client.post("/api/models", json={
        "display_name": "Missing Local", "model_id": "./nope",
        "source": "local", "max_seq_length": 2048,
    })
    assert r.status_code == 400
    r = client.post("/api/models", json={
        "display_name": "Bad Seq", "model_id": "unsloth/phi-3-mini-4k-instruct",
        "source": "huggingface", "max_seq_length": 10,
    })
    assert r.status_code == 400


def test_datasets_list_filter_and_detail(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    data = r.json()["data"]
    names = {d["name"] for d in data}
    assert {"Raw Sample", "Trainable Sample"} <= names
    assert r.json()["warnings"]  # bad.json 被容错跳过

    r = client.get("/api/datasets", params={"scope": "trainable"})
    names = {d["name"] for d in r.json()["data"]}
    assert names == {"Trainable Sample"}

    r = client.get("/api/datasets/detail", params={"name": "Raw Sample"})
    assert r.status_code == 200
    assert "Raw Sample" in r.json()["data"]["detail"]

    r = client.get("/api/datasets/detail", params={"name": "No Such"})
    assert r.status_code == 404


def test_dataset_rename_and_delete(client):
    _write_dataset("Temp Rename")
    r = client.post("/api/datasets/rename",
                    json={"old": "Temp Rename", "new": "Temp Renamed"})
    assert r.status_code == 200, r.text
    assert r.json()["data"]["new"] == "Temp_Renamed"
    assert (DATASETS_CONFIG_DIR / "Temp_Renamed.json").is_file()

    r = client.delete("/api/datasets", params={"name": "Temp_Renamed"})
    assert r.status_code == 200, r.text
    assert not (DATASETS_CONFIG_DIR / "Temp_Renamed.json").exists()


def test_prepare_upload_inspect_preview(client):
    content = b'{"q": "q0", "a": "a0"}\n{"q": "q1", "a": "a1"}\n'
    r = client.post("/api/prepare/upload",
                    files={"file": ("sample.jsonl", content, "application/jsonl")})
    assert r.status_code == 200, r.text
    upload_id = r.json()["data"]["upload_id"]

    r = client.post("/api/prepare/inspect",
                    json={"kind": "upload", "upload_id": upload_id})
    assert r.status_code == 200, r.text
    payload = r.json()["data"]
    assert payload["columns"] == ["q", "a"]
    token = payload["token"]

    r = client.post("/api/prepare/preview", json={
        "token": token,
        "instruction_cols": ["q"],
        "input_cols": [],
        "think_cols": [],
        "output_cols": ["a"],
        "fixed_instruction": "",
    })
    assert r.status_code == 200, r.text
    assert "### Instruction:\nq0" in r.json()["data"]["preview"]
    assert "a0" in r.json()["data"]["preview"]

    r = client.post("/api/prepare/preview", json={
        "token": token,
        "instruction_cols": ["q"],
        "output_cols": ["a"],
    })
    assert r.status_code == 200


def test_prepare_rejects_bad_suffix_and_expired_token(client):
    r = client.post("/api/prepare/upload",
                    files={"file": ("x.exe", b"x", "application/octet-stream")})
    assert r.status_code == 400
    r = client.post("/api/prepare/preview", json={
        "token": "nope", "instruction_cols": ["q"], "output_cols": ["a"],
    })
    assert r.status_code == 400


def test_prepare_existing_source(client):
    r = client.post("/api/prepare/inspect",
                    json={"kind": "existing", "existing_name": "Raw Sample"})
    assert r.status_code == 200, r.text
    assert r.json()["data"]["columns"] == ["q", "a"]


def test_train_rejects_unprocessed_dataset(client):
    r = client.post("/api/train/start", json={
        "experiment_name": "api_test_exp",
        "selected_model_name": "Test Base",
        "selected_dataset_names": ["Raw Sample"],
        "training_mode": "steps",
        "max_steps": 1,
        "max_seq_length": 2048,
    })
    assert r.status_code == 400
    assert "cannot be used" in r.json()["error"]["message"].lower() or \
        "不能训练" in r.json()["error"]["message"]


def test_train_rejects_missing_model(client):
    r = client.post("/api/train/start", json={
        "experiment_name": "api_test_exp",
        "selected_model_name": "No Such Model",
        "selected_dataset_names": ["Trainable Sample"],
    })
    assert r.status_code == 400


def test_jobs_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/events").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_inference_status_and_loras(client):
    r = client.get("/api/inference/status")
    assert r.status_code == 200
    assert r.json()["data"]["loaded"] is False
    r = client.get("/api/inference/loras")
    assert r.status_code == 200
    assert r.json()["data"] == []


def test_inference_websocket_ready_ping(client):
    with client.websocket_connect("/api/inference/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["data"]["loaded"] is False
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
        ws.send_json({"type": "download"})
        err = ws.receive_json()
        assert err["type"] == "error"
        ws.send_json({"type": "cancel"})
        assert ws.receive_json()["type"] == "cancelling"


def test_prepare_generate_job_and_sse(client):
    pytest = __import__("pytest")
    pytest.importorskip("datasets")
    import time

    r = client.post("/api/prepare/inspect",
                    json={"kind": "existing", "existing_name": "Raw Sample"})
    assert r.status_code == 200, r.text
    token = r.json()["data"]["token"]

    r = client.post("/api/prepare/generate", json={
        "token": token,
        "instruction_cols": ["q"],
        "input_cols": [],
        "think_cols": [],
        "output_cols": ["a"],
        "fixed_instruction": "",
        "output_name": "Generated Test",
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["data"]["job_id"]

    state = None
    for _ in range(400):
        r = client.get(f"/api/jobs/{job_id}")
        state = r.json()["data"]
        if state["state"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)
    assert state["state"] == "done", state
    assert state["result"]["name"] == "Generated_Test"

    r = client.get(f"/api/jobs/{job_id}/events")
    assert "event: done" in r.text
    assert "event: progress" in r.text

    r = client.get("/api/datasets", params={"scope": "trainable"})
    names = {d["name"] for d in r.json()["data"]}
    assert "Generated_Test" in names

    r = client.delete("/api/datasets", params={"name": "Generated_Test"})
    assert r.status_code == 200
