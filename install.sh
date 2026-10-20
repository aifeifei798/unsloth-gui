#!/usr/bin/env bash
# 一键安装 Unsloth GUI：Web 层 +（默认）训练/推理栈。
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-}"
PYTHON_BIN="${PYTHON:-python3}"
INSTALLER="${INSTALLER:-auto}"   # auto / uv / pip
WITH_TRAIN=1
WITH_DEV=0

usage() {
  cat <<'EOF'
用法: ./install.sh [选项]

默认安装 Web 界面 + 完整训练栈（requirements-web.txt + requirements.txt）。

选项:
  --web-only       只安装 Web 界面依赖（无 GPU、只想打开界面时用）
  --dev            额外安装测试工具（pytest / httpx / pyflakes）
  --venv DIR       虚拟环境目录（默认 ./.venv，也可用 $VENV_DIR）
  --python BIN     创建虚拟环境用的 Python（默认 python3，也可用 $PYTHON）
  --uv             强制使用 uv 安装
  --pip            强制使用 pip 安装
  -h, --help       显示帮助

安装完成后:
  ./server.sh start       后台启动
  ./server.sh run         前台运行
EOF
}

die() {
  printf '错误: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '==> %s\n' "$*"
}

while (($#)); do
  case "$1" in
    --web-only)
      WITH_TRAIN=0
      shift
      ;;
    --dev)
      WITH_DEV=1
      shift
      ;;
    --venv)
      [[ $# -ge 2 ]] || die "--venv 需要目录参数"
      VENV_DIR="$2"
      shift 2
      ;;
    --venv=*)
      VENV_DIR="${1#*=}"
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python 需要可执行文件参数"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --python=*)
      PYTHON_BIN="${1#*=}"
      shift
      ;;
    --uv)
      INSTALLER=uv
      shift
      ;;
    --pip)
      INSTALLER=pip
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "未知参数: $1（用 --help 查看用法）"
      ;;
  esac
done

[[ -f "$ROOT_DIR/requirements-web.txt" ]] || die "找不到 requirements-web.txt"
[[ -f "$ROOT_DIR/requirements.txt" ]] || die "找不到 requirements.txt"

PYTHON_INPUT="$PYTHON_BIN"
PYTHON_BIN="$(command -v "$PYTHON_INPUT" 2>/dev/null || true)"
[[ -n "$PYTHON_BIN" ]] || die "找不到 Python: $PYTHON_INPUT"
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

if [[ "$INSTALLER" == auto ]]; then
  if command -v uv >/dev/null 2>&1; then
    INSTALLER=uv
  else
    INSTALLER=pip
  fi
fi
[[ "$INSTALLER" == uv || "$INSTALLER" == pip ]] \
  || die "INSTALLER 只能是 auto / uv / pip"
if [[ "$INSTALLER" == uv ]] && ! command -v uv >/dev/null 2>&1; then
  die "指定了 --uv，但找不到 uv；请先安装 uv，或改用 --pip"
fi

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
VENV_PY="$VENV_DIR/bin/python"

if [[ -e "$VENV_DIR" && ! -x "$VENV_PY" ]]; then
  if [[ -d "$VENV_DIR" && -z "$(ls -A "$VENV_DIR" 2>/dev/null || true)" ]]; then
    :
  else
    die "$VENV_DIR 已存在但不是可用的虚拟环境；换一个 --venv 目录，或先删除它"
  fi
fi

if [[ ! -x "$VENV_PY" ]]; then
  "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "需要 Python 3.10+，当前是 $PY_VERSION"
  info "创建虚拟环境: $VENV_DIR (Python $PY_VERSION)"
  if [[ "$INSTALLER" == uv ]]; then
    uv venv --python "$PYTHON_BIN" "$VENV_DIR" \
      || die "uv 创建虚拟环境失败"
  else
    "$PYTHON_BIN" -m venv "$VENV_DIR" \
      || die "创建虚拟环境失败；Debian/Ubuntu 可能需要先 apt install python3-venv"
  fi
else
  info "复用虚拟环境: $VENV_DIR"
fi

[[ -x "$VENV_PY" ]] || die "虚拟环境创建后仍找不到 $VENV_PY"
"$VENV_PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "虚拟环境 Python 低于 3.10；请删除 $VENV_DIR 后重试"

install_pip() {
  "$VENV_PY" -m pip install "$@"
}

ensure_pip() {
  if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    "$VENV_PY" -m ensurepip --upgrade \
      || die "虚拟环境缺少 pip，且 ensurepip 失败；请改用 --uv"
  fi
}

install_uv() {
  uv pip install --python "$VENV_PY" "$@"
}

info "安装 Web 层依赖"
if [[ "$INSTALLER" == uv ]]; then
  install_uv -r "$ROOT_DIR/requirements-web.txt"
else
  ensure_pip
  "$VENV_PY" -m pip install --upgrade pip
  install_pip -r "$ROOT_DIR/requirements-web.txt"
fi

install_train_deps() {
  if [[ "$INSTALLER" == uv ]]; then
    local uv_help
    uv_help="$(uv pip install --help 2>&1 || true)"
    if [[ "$uv_help" == *"--torch-backend"* ]]; then
      install_uv --torch-backend=auto -r "$ROOT_DIR/requirements.txt"
    else
      install_uv -r "$ROOT_DIR/requirements.txt"
    fi
  else
    install_pip -r "$ROOT_DIR/requirements.txt"
  fi
}

if ((WITH_TRAIN)); then
  info "安装训练/推理依赖（unsloth 会自动带 torch / transformers / trl / peft）"
  if ! install_train_deps; then
    printf '训练栈安装失败。Web 层已安装，可先用 ./server.sh start；\n' >&2
    printf '如果只需要 Web 界面，重跑: ./install.sh --web-only\n' >&2
    exit 1
  fi
else
  info "跳过训练栈（--web-only）"
fi

if ((WITH_DEV)); then
  info "安装开发/测试依赖"
  if [[ "$INSTALLER" == uv ]]; then
    install_uv pytest httpx pyflakes
  else
    install_pip pytest httpx pyflakes
  fi
fi

mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/logs" \
  "$ROOT_DIR/local_data/uploads" "$ROOT_DIR/local_data/processed"

info "检查 Web 入口"
"$VENV_PY" - <<'PY'
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
try:
    from web.main import app  # noqa: F401
except Exception as exc:  # pragma: no cover - 安装失败时才走到这里
    print(f"Web 入口导入失败: {exc}", file=sys.stderr)
    raise SystemExit(1)

print("Web 入口可导入")
if importlib.util.find_spec("torch") is None:
    print("提示: 未检测到 torch；训练/推理不可用（Web 界面仍可用）")
if importlib.util.find_spec("unsloth") is None:
    print("提示: 未检测到 unsloth；训练不可用（Web 界面仍可用）")
PY

cat <<EOF

安装完成。
  虚拟环境: $VENV_DIR
  启动服务: ./server.sh start
  前台调试: ./server.sh run
  查看状态: ./server.sh status
  查看日志: ./server.sh logs -f

浏览器打开: http://127.0.0.1:${UNSLOTH_WEB_PORT:-7861}
EOF
