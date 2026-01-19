.PHONY: help install install-dev install-table test lint format clean run

help: ## 显示此帮助信息
	@echo "可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## 安装基础依赖
	python -m venv .venv
	source .venv/bin/activate && pip install --upgrade pip
	source .venv/bin/activate && pip install -r requirements.txt

install-dev: install ## 安装开发依赖
	source .venv/bin/activate && pip install -e ".[dev]"

install-table: ## 安装表格识别依赖
	source .venv/bin/activate && pip install -e ".[table]"

test: ## 运行测试
	source .venv/bin/activate && python -m pytest tests/ -v

lint: ## 运行代码检查
	source .venv/bin/activate && flake8 pdf_to_markdown.py
	source .venv/bin/activate && mypy pdf_to_markdown.py

format: ## 格式化代码
	source .venv/bin/activate && black pdf_to_markdown.py

clean: ## 清理临时文件
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*.pyd' -delete
	find . -type f -name '.DS_Store' -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

run: ## 运行示例
	source .venv/bin/activate && python pdf_to_markdown.py input.pdf

check: lint test ## 运行所有检查
