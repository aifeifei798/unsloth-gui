#!/usr/bin/env bash
# 统一管理 Unsloth GUI 服务：start / stop / restart / status / logs / run。
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_DIR="$ROOT_DIR/.run"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"
ENV_FILE="$RUN_DIR/server.env"

ACTION="${1:-help}"
if (($#)); then
  shift
fi

CLI_HOST=""
CLI_PORT=""
CLI_VENV=""
RELOAD=0
FOLLOW=0
LINES=100

usage() {
  cat <<'EOF'
用法: ./server.sh <命令> [选项]

命令:
  start       后台启动（默认 127.0.0.1:7861）
  stop        停止
  restart     重启
  status      查看运行状态与健康检查
  logs        查看日志（默认最后 100 行）
  run         前台运行（Ctrl-C 退出，适合调试 / systemd）
  help        显示帮助

选项:
  --host HOST       监听地址（默认 127.0.0.1，或 $UNSLOTH_WEB_HOST）
  --port PORT       监听端口（默认 7861，或 $UNSLOTH_WEB_PORT）
  --venv DIR        虚拟环境目录（默认 ./.venv，或 $VENV_DIR）
  --reload          开发模式自动重载（仅 start / run）
  logs -f           持续跟踪日志
  logs -n N         显示最后 N 行（默认 100）
  -h, --help        显示帮助

首次使用先运行: ./install.sh

示例:
  ./server.sh start
  ./server.sh start --port 8080
  ./server.sh status
  ./server.sh logs -f
  ./server.sh restart
  ./server.sh stop
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
    --host)
      [[ $# -ge 2 ]] || die "--host 需要地址参数"
      CLI_HOST="$2"
      shift 2
      ;;
    --host=*)
      CLI_HOST="${1#*=}"
      shift
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port 需要端口参数"
      CLI_PORT="$2"
      shift 2
      ;;
    --port=*)
      CLI_PORT="${1#*=}"
      shift
      ;;
    --venv)
      [[ $# -ge 2 ]] || die "--venv 需要目录参数"
      CLI_VENV="$2"
      shift 2
      ;;
    --venv=*)
      CLI_VENV="${1#*=}"
      shift
      ;;
    --reload)
      RELOAD=1
      shift
      ;;
    -f|--follow)
      FOLLOW=1
      shift
      ;;
    -n|--lines)
      [[ $# -ge 2 ]] || die "-n 需要行数"
      LINES="$2"
      shift 2
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

case "$ACTION" in
  start|stop|restart|status|logs|run|help) ;;
  -h|--help) ACTION=help ;;
  *)
    die "未知命令: $ACTION（用 --help 查看用法）"
    ;;
esac

if [[ "$ACTION" == help ]]; then
  usage
  exit 0
fi

SAVED_HOST=""
SAVED_PORT=""
SAVED_VENV=""
if [[ -f "$ENV_FILE" ]]; then
  while IFS='=' read -r key value; do
    case "$key" in
      HOST) SAVED_HOST="$value" ;;
      PORT) SAVED_PORT="$value" ;;
      VENV_DIR) SAVED_VENV="$value" ;;
    esac
  done < "$ENV_FILE"
fi

HOST="${CLI_HOST:-${UNSLOTH_WEB_HOST:-${SAVED_HOST:-127.0.0.1}}}"
PORT="${CLI_PORT:-${UNSLOTH_WEB_PORT:-${SAVED_PORT:-7861}}}"
VENV_DIR="${CLI_VENV:-${VENV_DIR:-${SAVED_VENV:-$ROOT_DIR/.venv}}}"
VENV_PY="$VENV_DIR/bin/python"

[[ -n "$HOST" ]] || die "监听地址不能为空"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "端口必须是数字: $PORT"
((PORT >= 1 && PORT <= 65535)) || die "端口超出范围: $PORT"
[[ "$LINES" =~ ^[0-9]+$ ]] || die "日志行数必须是数字: $LINES"

CHECK_HOST="$HOST"
case "$HOST" in
  0.0.0.0|::|'[::]') CHECK_HOST="127.0.0.1" ;;
esac

RUNNING_PID=""

is_running() {
  [[ -f "$PID_FILE" ]] || return 1

  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || {
    rm -f "$PID_FILE"
    return 1
  }

  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    return 1
  fi

  # 防止 PID 被其他进程复用：Linux 上核对命令行。
  if [[ -r "/proc/$pid/cmdline" ]]; then
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$cmdline" != *"-m web"* && "$cmdline" != *"web/__main__.py"* ]]; then
      rm -f "$PID_FILE"
      return 1
    fi
  fi

  RUNNING_PID="$pid"
  return 0
}

health_ok() {
  local url="http://$CHECK_HOST:$PORT/api/health"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 1 "$url" >/dev/null 2>&1
  else
    "$VENV_PY" - "$CHECK_HOST" "$PORT" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=1) as resp:
    resp.read()
PY
  fi
}

write_env() {
  mkdir -p "$RUN_DIR"
  printf 'HOST=%s\nPORT=%s\nVENV_DIR=%s\n' "$HOST" "$PORT" "$VENV_DIR" > "$ENV_FILE"
}

start_server() {
  if is_running; then
    printf '服务已在运行: PID %s\n' "$RUNNING_PID"
    printf '地址: http://%s:%s\n' "$CHECK_HOST" "$PORT"
    return 0
  fi

  [[ -x "$VENV_PY" ]] \
    || die "找不到 $VENV_PY；先运行 ./install.sh（或 ./install.sh --web-only）"

  mkdir -p "$RUN_DIR"
  write_env

  local args=(-m web --host "$HOST" --port "$PORT")
  ((RELOAD)) && args+=(--reload)

  info "启动服务: $HOST:$PORT"
  nohup "$VENV_PY" "${args[@]}" >>"$LOG_FILE" 2>&1 < /dev/null &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"

  local i
  for ((i = 0; i < 80; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      printf '启动失败，最近日志:\n' >&2
      tail -n 40 "$LOG_FILE" >&2 || true
      return 1
    fi
    if health_ok; then
      printf '已启动: http://%s:%s (PID %s)\n' "$CHECK_HOST" "$PORT" "$pid"
      printf '日志: %s\n' "$LOG_FILE"
      return 0
    fi
    sleep 0.25
  done

  printf '进程已启动但健康检查超时（PID %s）\n' "$pid" >&2
  printf '查看日志: ./server.sh logs\n' >&2
  return 1
}

stop_server() {
  if ! is_running; then
    rm -f "$PID_FILE"
    printf '服务未运行\n'
    return 0
  fi

  local pid="$RUNNING_PID"
  info "停止服务: PID $pid"
  kill "$pid" 2>/dev/null || true

  local i
  for ((i = 0; i < 60; i++)); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done

  if kill -0 "$pid" 2>/dev/null; then
    printf '进程未退出，强制结束\n' >&2
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$PID_FILE"
  printf '已停止\n'
}

status_server() {
  if is_running; then
    local rc=1
    printf '状态: 运行中\n'
    printf 'PID: %s\n' "$RUNNING_PID"
    printf '地址: http://%s:%s\n' "$CHECK_HOST" "$PORT"
    if health_ok; then
      printf '健康检查: OK\n'
      rc=0
    else
      printf '健康检查: 失败（进程还在，检查日志）\n' >&2
    fi
    printf '虚拟环境: %s\n' "$VENV_DIR"
    printf '日志: %s\n' "$LOG_FILE"
    return "$rc"
  fi

  printf '状态: 未运行\n'
  printf '虚拟环境: %s\n' "$VENV_DIR"
  printf '启动: ./server.sh start\n'
  return 3
}

show_logs() {
  [[ -f "$LOG_FILE" ]] || die "还没有日志文件: $LOG_FILE"
  if ((FOLLOW)); then
    tail -n "$LINES" -f "$LOG_FILE"
  else
    tail -n "$LINES" "$LOG_FILE"
  fi
}

run_foreground() {
  [[ -x "$VENV_PY" ]] \
    || die "找不到 $VENV_PY；先运行 ./install.sh（或 ./install.sh --web-only）"

  mkdir -p "$RUN_DIR"
  write_env

  local args=(-m web --host "$HOST" --port "$PORT")
  ((RELOAD)) && args+=(--reload)

  printf '前台运行: http://%s:%s（Ctrl-C 退出）\n' "$CHECK_HOST" "$PORT"
  exec "$VENV_PY" "${args[@]}"
}

case "$ACTION" in
  start) start_server ;;
  stop) stop_server ;;
  restart)
    stop_server
    start_server
    ;;
  status) status_server ;;
  logs) show_logs ;;
  run) run_foreground ;;
  help) usage ;;
esac
