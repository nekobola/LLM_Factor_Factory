# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FactorFactory AI** - 全自动金融因子挖掘与验证系统，实现从原始数据到策略部署的全流程闭环。

### Core Pipeline
```
Data Engine → LLM Idea Generator → Validator → Strategy Builder → Storage/UI
```

### Key Technical Decisions
- **数据源**: Tushare/AkShare (A股) + YFinance (全球)
- **存储**: SQLite 本地文件
- **验证**: IC/IR + 分组回测 + 正交化 + Newey-West
- **回测框架**: Alphalens
- **前端**: Streamlit
- **LLM**: 可配置多模型 (Claude/OpenAI)

## Commands

### Environment Setup
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### Testing
```bash
# 运行所有测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_base_factor.py -v

# 带覆盖率
pytest tests/ -v --cov=factor_factory
```

### Running the Application
```bash
# CLI 入口
python main.py --help

# 启动 Streamlit 前端
streamlit run web/app.py
```

### Code Quality
```bash
# 格式化
black factor_factory/

# Linting
ruff check factor_factory/

# 类型检查
mypy factor_factory/
```

## Architecture

### Directory Structure
```
factor_factory/
├── core/           # BaseFactor 抽象基类, Pipeline 工作流, Registry
├── data_engine/    # 数据加载器 (Tushare/AkShare/YFinance), 清洗, 对齐
├── idea_generator/ # LLM 客户端, Prompt 模板, 表达式解析
├── validator/      # IC/IR 分析, 分组回测, 正交化, Newey-West
├── strategy_builder/ # 信号生成, 回测
├── storage/        # SQLAlchemy ORM, 因子存储
└── web/            # Streamlit 前端
```

### BaseFactor Interface
所有因子必须继承 `core.base_factor.BaseFactor`，实现：
- `compute(data: DataFrame) -> Series`: 核心计算逻辑
- `validate_inputs(data: DataFrame) -> bool`: 输入验证

输入数据格式：MultiIndex `(date, asset)`，包含 OHLCV 列。

### FactorPipeline Workflow
```python
pipeline = FactorPipeline(llm_client, validator, data_loader)
result = await pipeline.run_cycle("生成动量因子")
# 自动循环: LLM生成 → 代码验证 → 错误修正 → 注册入库
```

### Validation Criteria
因子通过验证需满足：
- `IC_mean > 0.02` 且 `IC_ir > 0.5`
- 分组单调性得分 > 0.8
- 正交化后 IC 衰减 < 50%
- Newey-West 调整后显著性 p < 0.05

## Configuration

- `config/settings.yaml`: 全局配置（数据源、验证阈值等）
- `config/llm_config.yaml`: LLM API 配置
- 环境变量 `.env`: API Keys (TUSHARE_TOKEN, ANTHROPIC_API_KEY, OPENAI_API_KEY)

## Code Style Notes

- 金融计量代码需特别注意数值稳定性（如除零保护、对数处理）
- Newey-West 滞后阶数使用 Schwert 准则自动计算
- 因子值需做 Winsorize (3σ) 和 Z-Score 标准化
- 所有异步操作使用 `async/await`，LLM 调用为异步
