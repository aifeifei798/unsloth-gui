# AGENTS.md — unsloth-gui 协作说明

> 给 AI 编程助手看的仓库约定。改代码前先读完。

## 项目是什么

基于 Gradio 6 + Unsloth 的轻量**纯文本单卡 SFT** 图形工作台（文本训练 only，
图片/视频/音频不做）。五个 Tab：模型管理 / 数据管理 / 数据处理 / 训练 / 测试。
界面中英双语，文案全在 `locales/*.json`。

## 目录

```
app.py                 # 薄 UI 层：Gradio 组件 + 事件接线 + 轻量 handler（校验/提示）
locales/zh.json        # 中文文案（基准，精确措辞以它为准）
locales/en.json        # 英文文案（key 必须与 zh 一一对应）
src/
  i18n.py              # t(key, **params)，缺键回退中文；check_parity() 查 key 对齐
  config.py            # 模型/数据集配置加载、校验、来源解析（local/hf/modelscope）
  dataset_utils.py     # 训练格式化、空段清理、多数据集合并
  dataprep.py          # 数据处理：预览、映射、统一数据生成（schema v3）
  data_mgmt.py         # 数据管理：详情、重命名、删除（只删生成物）
  train_utils.py       # 后台训练、可取消、断点续训（重型依赖函数内延迟导入）
  inference_utils.py   # 推理加载、显存管理、流式对话
  tb_utils.py          # TensorBoard 生命周期
models.json            # 模型列表（UI 维护，可手写；source 省略=auto）
datasets_config/       # 数据集配置（只认 processed=true 的；数据处理自动生成）
local_data/            # 示例/上传/产物（processed/、uploads/ 已 gitignore）
```

## 铁律（违反必出 bug）

1. **训练只认统一数据**：`train_utils` 按 `processed` 拦截，`datasets_config` 不要手写非 processed 条目。
2. **统一格式**：`instruction / input / think / output`，Response = think + output，
   think 缺 `<think>` 标签自动套；模板无 Input 段当 input 为空；空段由
   `drop_empty_sections` 清理——三处文本出口（训练/单行预览/生成预览）必须共用它。
3. **纯文本单路径**：不要加 vision/多模态分支（已明确砍掉，见 git 历史）。
4. **文案禁硬编码**：所有用户可见中文/英文走 `t()` + locales 双文件同步加 key。
5. **重型依赖延迟导入**：`torch / unsloth / trl / transformers / datasets / peft`
   只许在函数内 import，模块顶层保持 stdlib-only（UI 无 GPU 也能打开）。
6. **Gradio 事件输出元数**必须与 handler 返回元组长度一致；`select` 事件参数必须标
   `evt: gr.SelectData` 注解，否则 6.x 报 warning。
7. **删数据只删生成物**：`local_data/processed/<name>/` + 对应 config JSON；
   原始上传、自带示例永远不动。删模型至少保留一个。
8. **配置文件原子写**：参考 `save_models_config`（tmp + replace）。

## 改完必须跑（本机无 GPU，所有验证都不需要 torch）

```bash
python3 -m py_compile app.py src/*.py
# 有 gradio 的 venv 里（/tmp/opencode/gradiotest 参考）：
python -m pyflakes app.py src/*.py                         # 必须 0
python -W error::UserWarning -c "import app"               # 零警告构建
UNSLOTH_GUI_LANG=en python -W error::UserWarning -c "import app"  # 英文构建
python -c "from src.i18n import check_parity; assert check_parity()[0]"
```

逻辑回归用 venv python 直调 handler（`datasets` 库一般有）：
`_model_save/_model_delete_step`（两步删除含取消路径）、
`inspect_text → generate_unified → prepare_dataset` 整链、
`preview_row` 与生成样本逐字一致、重命名目录/配置/manifest/下拉同步。
**产生的一切测试产物必须删干净**（`local_data/processed/test_*`、
`datasets_config/test_*`、`outputs/`、`logs/`），备份的真配置必须还原，
`git status` 只剩预期文件。

## 提交规范

- 中文一句话 commit（如 `数据处理加单行预览：映射完先看最终训练文本`）。
- 只 commit/review，不 push、不改 git config、不 force-push——push 由用户明确说才做。
- README.md 与 README.en.md 内容同步更新。
