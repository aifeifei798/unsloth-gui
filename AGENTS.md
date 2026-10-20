# AGENTS.md — unsloth-gui 协作说明

> 给 AI 编程助手看的仓库约定。改代码前先读完。

## 项目是什么

基于 FastAPI + Jinja2 + 原生 JS + DaisyUI 的轻量**纯文本单卡 SFT** 图形工作台
（文本训练 only，图片/视频/音频不做）。五个页面：模型管理 / 数据管理 / 数据处理 /
训练 / 测试。旧 Gradio 界面（`app.py`）过渡期暂时保留，两套 UI 共用 `src/` 领域层。
界面中英双语，文案全在 `locales/*.json`；Web 用 Cookie + `ContextVar` 请求级切换语言，
不重启进程。

## 目录

```
app.py                 # 旧 Gradio UI（过渡期保留，稳定后删除）
web/
  main.py              # FastAPI 入口、语言中间件、页面路由、异常格式、静态挂载
  deps.py / errors.py  # 依赖、AppError
  jobs.py              # JobManager + SSE（事件补发、Last-Event-ID、取消）
  schemas.py           # 请求体模型
  routers/             # system / models / datasets / prepare / train / inference
  templates/           # base.html + 五个页面
  static/css/app.css   # 手写布局工具类（DaisyUI 只带组件，不带 Tailwind utilities）
  static/js/           # common + 五个页面脚本
  static/vendor/       # DaisyUI CSS、marked、DOMPurify（本地 vendor）
tests/                 # pytest：domain / API / jobs / SSE / WebSocket
locales/zh.json        # 中文文案（基准，精确措辞以它为准）
locales/en.json        # 英文文案（key 必须与 zh 一一对应）
src/
  paths.py             # 统一路径解析（UNSLOTH_GUI_ROOT 可把根目录指到临时目录）
  storage.py           # 配置文件原子写
  i18n.py              # t(key, **params)，缺键回退中文；check_parity() 查 key 对齐
  config.py            # 模型/数据集配置加载、校验、来源解析（local/hf/modelscope）
  dataset_utils.py     # 训练格式化、空段清理、多数据集合并
  dataprep.py          # 数据处理：预览、映射、统一数据生成（schema v3）
  data_mgmt.py         # 数据管理：详情、重命名、删除（只删生成物）
  train_utils.py       # 后台训练、可取消、断点续训、run manifest（重型依赖延迟导入）
  inference_utils.py   # 推理加载、显存管理、流式对话、manifest 模板
  tb_utils.py          # TensorBoard 生命周期
requirements-web.txt   # Web 层依赖（FastAPI / Uvicorn / Jinja2 / python-multipart）
models.json            # 模型列表（UI 维护，可手写；source 省略=auto）
datasets_config/       # 数据集配置（只认 processed=true 的；数据处理自动生成）
local_data/            # 示例/上传/产物（processed/、uploads/ 已 gitignore）
```

## 铁律（违反必出 bug）

1. **训练只认统一数据**：`train_utils` 按 `processed=True + schema_version=3 + 产物目录存在`
   拦截；`datasets_config` 不要手写非 processed 条目。
2. **统一格式**：`instruction / input / think / output`，Response = think + output，
   think 缺 `<think>` 标签自动套；模板无 Input 段当 input 为空；空段由
   `drop_empty_sections` 清理——三处文本出口（训练/单行预览/生成预览）必须共用它。
3. **纯文本单路径**：不要加 vision/多模态分支（已明确砍掉，见 git 历史）。
4. **文案禁硬编码**：所有用户可见中文/英文走 `t()` + locales 双文件同步加 key。
   Web 模板由 Jinja 注入 `window.I18N`，前端 JS 一律用 `t('key')`，不要硬编码中文。
5. **重型依赖延迟导入**：`torch / unsloth / trl / transformers / datasets / peft`
   只许在函数内 import，模块顶层保持 stdlib-only（UI 无 GPU 也能打开）。
6. **Gradio 过渡期**：`app.py` 仍在用；改 `src/` 时必须保证 Web 与 Gradio 两套入口都不破。
   Gradio 事件输出元数必须与 handler 返回元组长度一致；`select` 事件参数必须标
   `evt: gr.SelectData` 注解，否则 6.x 报 warning。
7. **删数据只删生成物**：`local_data/processed/<name>/` + 对应 config JSON；
   原始上传、自带示例永远不动。删模型至少保留一个。
8. **配置文件原子写**：统一走 `src/storage.py` 的 tmp + replace。
9. **Web 单进程单 worker**：训练与推理互斥，不要引入多 worker / 多进程共享显存；
   后台任务统一走 `web/jobs.py`，训练和数据生成进度用 SSE，聊天用 WebSocket。
10. **Web 职责边界**：路由只做校验、参数转换、调用 `src/`；领域逻辑不要写进
    `web/routers/`，也不要复制一份到前端 JS。前端不引入 React/Vue，不需要 Node 构建。
11. **SSE/WS 约定**：事件名固定为 `state / progress / log / metrics / done / error /
    cancelled / end`；任务结束前必须发 `done` 或 `error`/`cancelled`，最后发 `end`。
    晚订阅/重连要能从 `job.events` 补发，`Last-Event-ID` 语义不要破坏。

## 改完必须跑（本机无 GPU，所有验证都不需要 torch）

```bash
# 1. 全量语法检查（Web + Gradio + 领域层）
python3 -m py_compile app.py src/*.py web/*.py web/routers/*.py

# 2. Web 入口能构建（在装好 requirements-web.txt 的 venv 里）
python -c "from web.main import app"

# 3. 测试（测试根目录自动用临时 UNSLOTH_GUI_ROOT，不碰真实数据）
pytest -q tests/

# 4. 文案对齐
python3 -c "from src.i18n import check_parity; assert check_parity()[0]"
```

有 gradio 的 venv 里再补跑（旧界面的回归）：

```bash
python -m pyflakes app.py src/*.py                         # 必须 0
python -W error::UserWarning -c "import app"               # 零警告构建
UNSLOTH_GUI_LANG=en python -W error::UserWarning -c "import app"
```

新 Web 层改动建议补一条真实服务冒烟：

```bash
python -m web --port 7861
# 另开终端：curl -sS http://127.0.0.1:7861/api/health
```

逻辑回归重点：`inspect_text → generate_unified → prepare_dataset` 整链、
`preview_row` 与生成样本逐字一致、重命名目录/配置/manifest/下拉同步、
训练启动前取消、SSE 晚订阅仍能收到 `done`、WebSocket 聊天 ready/ping/cancel。
**产生的一切测试产物必须删干净**（`local_data/processed/test_*`、
`datasets_config/test_*`、`outputs/`、`logs/`、临时 UNSLOTH_GUI_ROOT），
备份的真配置必须还原，`git status` 只剩预期文件。

## 提交规范

- 中文一句话 commit（如 `数据处理加单行预览：映射完先看最终训练文本`）。
- 只 commit/review，不 push、不改 git config、不 force-push——push 由用户明确说才做。
- README.md 与 README.en.md 内容同步更新。
