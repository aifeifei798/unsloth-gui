# 🚀 Unsloth GUI Trainer: a lightweight, dedicated fine-tuning workbench

[![Version 0.2.0](https://img.shields.io/badge/version-0.2.0-blue.svg)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [中文版](README.md)

A FastAPI + Jinja2 + [Unsloth](https://github.com/unslothai/unsloth) GUI for fine-tuning,
focused on **single-GPU text SFT**: data prep, training, monitoring and chat are a
few clicks in the browser, with models and the UI language under Settings —
no code, no hand-edited configs.

> The web UI runs at `http://127.0.0.1:7861`; language switching is instant and
> needs no restart.
> This is a single-user local tool with **no authentication** — do not expose it
> directly to the public internet.
>
> Text training only (image/video/audio not supported yet). Runs out of the box on
> 8GB VRAM.

---

## ✨ Four entries

| Entry | What it does |
| --- | ------------ |
| 1 🧹 Data Prep | Pick a source (upload / HF ID / existing config) → read columns → map & generate: multi-select roles (instruction / input / think / output) → single-row preview → generate unified training data |
| 2 🚀 Train | Steps/epochs modes, cancellable, multi-dataset merge, resume from checkpoint, live TensorBoard; training text uses the model's own chat template by default |
| 3 💬 Chat Test | Load a LoRA for streaming chat, tunable temperature/top-p, thinking-mode toggle (turn off for think-free data), one-click unload to free VRAM |
| ⚙️ Settings | Models (local/HF/ModelScope CRUD) + Datasets (list, rename, delete) + interface language |

Unified format: 4 columns, `instruction / input / think / output`, each multi-selectable
and joined top-down with newlines; no Input section when input is unmapped.
During training **user = instruction (+context) / assistant = reply (with think)**,
and the final training text is rendered by the **model's own chat template**
(`tokenizer.apply_chat_template`, the official Unsloth way); uncheck
"Use the model's own chat template" on the Train page to fall back to the
Instruction / Input (optional) / Response template.
Response = think + output (`<think>` tags auto-added when missing);
rows with empty replies are dropped and counted.

**Only data produced by Data Prep (`processed=true`, schema v3, artifact directory
present) appears in the training dropdown.** The bundled `datasets_config/sunwukong.json`
is unprocessed by default; run it through Data Prep first.

Every model in Models has an "Unsloth fast download (hf_transfer)" toggle, **off by
default**; keep it off when your environment doesn't support hf_transfer, otherwise
downloads can stall on the progress bar.

---

## 🔧 Install

Requires Python 3.10+; training/inference needs an NVIDIA GPU + CUDA (8GB VRAM minimum).

```bash
git clone https://github.com/aifeifei798/unsloth-gui.git
cd unsloth-gui

./install.sh              # default: web UI + training/inference stack
./install.sh --web-only   # web UI only (no GPU / UI-only machine)
./install.sh --dev        # also install pytest / httpx / pyflakes
```

The script creates `.venv`, prefers `uv` (with automatic torch backend matching),
and falls back to pip. Use `--venv DIR` / `--python BIN` to override the
environment. Manual installation still works:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-web.txt
```

`requirements.txt` is the full training stack; `requirements-web.txt` installs only
the web layer. Without a GPU/torch you can still open the UI, manage models and
prepare data — training and inference will report missing dependencies.

---

## ▶️ Run

```bash
./server.sh start
./server.sh status
./server.sh logs -f       # follow logs; Ctrl-C to exit
./server.sh restart
./server.sh stop
```

Use `./server.sh run` for foreground mode. The default is `127.0.0.1:7861`;
override with `--host` / `--port`. PID and server logs live in `.run/`
(git-ignored).

Open `http://127.0.0.1:7861`. Flow: add a model in Settings →
make data in Data Prep → Train (watch TensorBoard live, stop/resume anytime) →
load the LoRA in Chat Test and chat. Switch 中文/English in Settings instantly,
with no restart.

Training needs CUDA and the training stack from `requirements.txt`; starting the
web UI does not import torch.

Artifacts: `outputs/<experiment>/` (LoRA adapter + `run_manifest.json`),
`logs/<experiment>/` (TensorBoard), `local_data/processed/<data>/` (unified data).
The chat page reads the prompt template, `seq_len` and base-model info from
`run_manifest.json`, matching the training format automatically.

### ⏱ Training feels slow to start? Unsloth's switches

Between "Start training" and the first step there is a minute or two of fixed
Unsloth overhead. Measured on an RTX 5090 D with Qwen3.5-0.8B:

| Stage | Default | Turned off |
| --- | --- | --- |
| `import unsloth` (Zoo patching) | ~10s | same |
| **`get_statistics()` health probe** (subprocess downloads a README from HF to check it is up) | **~76s (worse on a slow proxy)** | **blocked by default** (`UNSLOTH_DISABLE_STATISTICS=1` + `get_statistics` replaced with a no-op) |
| `torch.compile` of the 24 decoder blocks | ~30s | `UNSLOTH_BLOCK_COMPILE=0`, but each step gets slower |
| Weight load + first forward | ~35s | same |

This project sets `UNSLOTH_DISABLE_STATISTICS=1` by default (the probe only
checks whether HF is reachable, it has nothing to do with training) and, right
after `import unsloth`, replaces every `get_statistics` entry point with a no-op
(Unsloth binds them with `from ._utils import *`, so the env var alone is not
enough). The `has_internet()` probe is replaced too — it dials 8.8.8.8:53, which
is usually blocked in mainland China, behind corporate firewalls or on some
cloud networks (3 s × 4 wasted), and it calls `socket.setdefaulttimeout(3)`,
which silently turns the *whole process*' default socket timeout into 3 s and
kills slower downloads. We remember the original value before the import and
restore it afterwards.

To let the probe through for one run:
`UNSLOTH_GUI_ALLOW_STATISTICS=1 ./server.sh start`.

For an even quicker start, add `UNSLOTH_BLOCK_COMPILE=0` (skips the
`torch.compile` of the 24 decoder blocks, but each step gets slower):

```bash
UNSLOTH_BLOCK_COMPILE=0 ./server.sh start
```

Two log lines mean you are on a slower path — installing them helps:
- `causal_conv1d is falling back to its reference PyTorch implementation` → Qwen3.5's linear-attention layers run the slow path; `pip install causal-conv1d` speeds them up
- `FA2 = False` → flash-attn 2 is missing, only xformers is used; `pip install flash-attn` is faster still (slow to build)

---

## 📁 Layout

```
install.sh              # one-shot install (web + optional training stack)
server.sh               # start / stop / restart / status / logs
.run/                   # server.sh PID and logs (runtime, git-ignored)
web/                    # FastAPI + Jinja2 web layer
  main.py               # app entry, language middleware, page routes
  jobs.py               # background jobs + SSE (training/data prep)
  routers/              # system / models / datasets / prepare / train / inference
  templates/            # base + four pages (prepare / train / chat / settings)
  static/               # app.css, page JS, DaisyUI vendor
tests/                  # pytest: domain / API / jobs / SSE / WebSocket
src/
  paths.py              # path resolution (UNSLOTH_GUI_ROOT overrides root)
  storage.py            # atomic config writes
  config.py             # model/dataset config loading, validation, source resolving
  dataset_utils.py      # training formatting, empty-section cleanup, combining
  dataprep.py           # data prep: peek, mapping, unified-data generation
  data_mgmt.py          # data mgmt: details, rename, delete
  train_utils.py        # background training jobs, cancel, resume, run manifest
  inference_utils.py    # inference loading, VRAM mgmt, streaming chat
  tb_utils.py           # TensorBoard lifecycle
  i18n.py               # tiny i18n helper (zh fallback, key parity check)
requirements.txt        # full training stack
requirements-web.txt    # web layer deps (FastAPI / Uvicorn / Jinja2 …)
models.json             # model list (maintained by the UI, hand-editable)
datasets_config/        # dataset configs (auto-generated by Data Prep)
local_data/             # sample data, uploads, artifacts (artifacts git-ignored)
```

Tests:

```bash
pip install pytest
pytest -q tests/
```

Tests point `UNSLOTH_GUI_ROOT` at a temporary directory, so they never touch your
real `models.json`, `datasets_config/`, `outputs/` or `logs/`.

---

## 📄 License

[MIT License](LICENSE).
