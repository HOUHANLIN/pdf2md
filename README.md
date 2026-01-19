# PDF 转 Markdown

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Project Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/HOUHANLIN/pdf2markdown)

将 PDF 转成 Markdown，优先提取可复制文本，必要时使用 SiliconFlow VLM 进行 OCR 兜底。支持断点续跑、并行 OCR、按页缓存、表格结构化提取、文本后处理、资源隔离渲染与结构化输出。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

OCR 渲染所需系统依赖（pdf2image）：

- macOS：`brew install poppler`
- Ubuntu：`sudo apt-get install poppler-utils`

> 表格识别需要额外依赖：
> - Camelot：`pip install camelot-py`
> - Tabula：`pip install tabula-py`（需 Java）

## 快速开始

```bash
python pdf_to_markdown.py input.pdf
python pdf_to_markdown.py input.pdf --mode text
python pdf_to_markdown.py input.pdf --mode ocr --api-key "$SILICONFLOW_API_KEY"
```

## 常用场景

```bash
# 断点续跑（复用缓存）
python pdf_to_markdown.py input.pdf --resume

# 并行 OCR + 限速
python pdf_to_markdown.py input.pdf --mode ocr --ocr-workers 4 --ocr-rps 2.0

# 渲染进程隔离（避免大页峰值内存）
python pdf_to_markdown.py input.pdf --render-workers 2

# 按页输出到目录，同时保留合并输出
python pdf_to_markdown.py input.pdf --per-page-dir pages/

# 只输出按页文件，不生成合并文件
python pdf_to_markdown.py input.pdf --per-page-dir pages/ --no-merge

# 自定义 OCR 提示词与语言提示
python pdf_to_markdown.py input.pdf --prompt-file prompt.txt --prompt-lang zh

# 结构化输出 + 参数快照
python pdf_to_markdown.py input.pdf --json-out result.json --config-out config.json

# 仅列出计划（不执行 OCR）
python pdf_to_markdown.py input.pdf --dry-run
```

## 断点续跑与缓存

- 默认缓存目录：`.pdf_to_markdown_cache/<pdf文件名>`
- `--resume` 会复用已生成的按页缓存。
- `--cache-dir` 可指定自定义缓存目录。
- `--cache-images` 可缓存渲染后的页图，OCR 重试时更快。
- `--render-workers` 用单独进程渲染页面（0 表示禁用）。
- `--retry-failed` / `--retry-failed-from` 仅重跑失败页。

## 输出结构配置

- `--separator`：自定义页间分隔符（支持 `\n` 转义）。
- `--page-header` / `--no-page-header`：是否添加页标题。
- `--page-header-in-pages`：按页输出时也添加页标题。
- `--per-page-dir`：按页输出目录。
- `--no-merge`：不生成合并输出。
- `--no-title`：不生成顶层标题。

## OCR 配置

- `--model`：模型名（支持逗号分隔多个模型）。
- `--model-fallback`：额外的降级模型列表。
- `--api-url` / `--api-url-fallback`：OCR 接口地址与降级地址。
- `--api-key`：支持逗号分隔多个 key，数量需与 API URL 数量一致。
- `--prompt-file`：外部 prompt 模板文件（支持 `{page}`、`{language}` 占位符）。
- `--prompt-lang` / `--auto-lang` / `--no-auto-lang`：语言提示与自动识别。

## OCR 质量策略

- 默认启用质量检查，若结果过短或异常字符比例过高，将尝试切换模型/接口。
- 关键参数：`--ocr-min-chars` / `--ocr-min-alnum-ratio` / `--ocr-max-symbol-ratio` / `--ocr-max-replace-ratio`
- 可用 `--no-ocr-quality` 关闭质量检查。
- `--adaptive-dpi`：质量不达标时使用 `--dpi-high` 重新 OCR。

## OCR 成本控制

- `--ocr-max-pixels` 限制输入像素，过大将自动缩小。
- `--ocr-min-side` 设置缩小后的最小边长。
- `--ocr-skip-large` 超阈值直接跳过 OCR。

## 图像预处理（默认启用）

- 自动二值化、去噪、对比度增强、去倾斜，提高 OCR 质量。
- 可用参数：
  - `--no-preprocess` / `--no-denoise` / `--no-binarize` / `--no-deskew`
  - `--denoise-size` / `--contrast` / `--binarize-threshold`
  - `--deskew-max-angle` / `--deskew-step`

## 文本后处理（默认启用）

- 自动合并断词/连字符（如 `exam-\nple`）。
- 去除重复页眉页脚（支持版面位置与字号判断）。
- 修复换行与段落分隔。

可用参数：
- `--no-postprocess` / `--no-dehyphenate` / `--no-reflow` / `--no-remove-hf`
- `--hf-top` / `--hf-bottom` / `--hf-min-ratio` / `--hf-min-pages`
- `--hf-use-layout` / `--hf-top-margin` / `--hf-bottom-margin` / `--hf-font-max-ratio` / `--hf-line-tolerance`
- `--remove-page-number` / `--no-remove-page-number`

## 版面分块（减少顺序错误）

- `--split-blocks`：检测水平空白并分块 OCR。
- `--block-gap-ratio` / `--block-min-ratio` / `--block-ink-ratio` / `--block-white-threshold`

## 去重（跨页重复段落）

- `--dedupe`：移除跨页重复段落（目录/免责声明等）。
- `--dedupe-min-ratio` / `--dedupe-min-pages` / `--dedupe-min-chars` / `--dedupe-strip-digits`

## 纠错字典 / 标题自动识别

- `--replace-file`：替换字典文件（支持 `a => b` / `a -> b` / `a<TAB>b`）。
- `--replace-mode`：`plain` / `word`。
- `--auto-headings`：自动识别标题（第X章/Chapter 1/1.1 形式）。
- `--heading-level` / `--heading-max-len`

## 表格识别（结构化提取）

- `--table-mode auto`：表格识别成功则直接使用，失败才 OCR。
- `--table-mode only`：仅尝试表格提取，不再 OCR。
- `--table-backend`：`camelot` / `tabula` / `auto`
- `--table-flavor`：`lattice` / `stream`
- `--no-table-header`：不将首行当作表头

## 版面检测（多栏）

- `--layout auto`：自动检测双栏并按列 OCR。
- `--layout single`：禁用列检测。
- `--layout columns`：强制按列 OCR。
- `--column-gap-ratio` / `--column-min-words` / `--column-overlap`

## 输出校验

- `--warn-short 80`：页面内容过短时插入提示行。
- `--warn-prefix`：提示行前缀。
- `--empty-placeholder`：空页占位文本。

## 结构化输出

- `--stats-out stats.json`：每页耗时、模式、错误等统计。
- `--json-out result.json`：包含每页文本 + 统计 + 元信息。
- `--config-out config.json`：输出参数快照，便于复现与对比。

## 钩子（前置/后置）

- `--pre-hook` / `--post-hook`：命令从 stdin 读取文本，stdout 输出替换后的内容。
- 环境变量：`PDF_PATH` / `PAGE_NUMBER` / `METHOD` / `STAGE`

## 统计与日志

- `--continue-on-error` 遇到单页失败继续执行。

完整参数请运行：

```bash
python pdf_to_markdown.py -h
```

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

## 致谢

- [pdfplumber](https://github.com/jsvine/pdfplumber) - PDF 文本提取
- [pdf2image](https://github.com/Belval/pdf2image) - PDF 转图片
- [Pillow](https://github.com/python-pillow/Pillow) - 图像处理
- [SiliconFlow](https://siliconflow.cn/) - OCR 服务支持

