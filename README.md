# FactorFactory AI

全自动金融因子挖掘与验证系统 — 从自然语言描述到可部署量化策略的全流程闭环。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)]()

## 这是什么？（非技术向概述）

FactorFactory AI 是一个**量化因子挖掘自动化工具**——你描述一个投资逻辑，系统完成从数据获取、因子计算到统计验证的全流程。

传统因子研究的流程是：提出假设 → 手工编码 → 数据清洗 → 回测 → 统计检验，迭代一次耗时数日。而这个系统将整个过程压缩为一句自然语言指令——

> "构建一个20日动量因子，IC > 0.03"

系统自动拉取数据、编译因子代码、以学术标准完成验证，最终输出结论：该因子是否具备统计显著性，以及可量化的置信度。

**能力范围：**
- 自然语言输入 —— 用中文描述因子逻辑，无需手写计算代码
- 自动化挖掘 —— 给定方向（如动量、反转、波动率），AI 生成大量候选因子并筛选最优
- 严格验证 —— 采用 Fama-MacBeth 两阶段回归 + Newey-West HAC 稳健性检验，非简单回测
- 策略构建 —— 通过验证的因子可导出为分组轮动策略，含交易成本建模

**当前局限：**
- 历史显著性不等于未来收益，本系统定位为研究辅助工具
- 不直接下达交易指令，不连接券商接口

## 全局流程

```
                               +---------------------+
                               |     因子假设         |
                               | "20日动量是否有效？"   |
                               +----------+----------+
                                          |
                                          v
                               +---------------------+
                               |    数据引擎          |
                               | 多数据源适配          |
                               | 清洗/对齐/去极值      |
                               +----------+----------+
                                          |
                                          v
                               +---------------------+
                               |   LLM 因子生成器     |
                               | 自然语言 → Python 代码 |
                               | AST 安全检查 + 自修正  |
                               +----------+----------+
                                          |
                                          v
                               +---------------------+
                               |     验证引擎         |
                               | IC 分析 | 分组回测    |
                               | 行业/市值中性化      |
                               | Newey-West t 检验   |
                               | 单调性评分           |
                               +----------+----------+
                                          |
                           +--------------+--------------+
                           |                             |
                           v                             v
                    +-------------+              +-------------+
                    |  |t| >= 3.0  |              |  |t| < 3.0  |
                    | 通过，入库   |              | 未通过，反馈  |
                    | 解锁回测引擎 |              | LLM 定向改进 |
                    +------+------+              +------+------+
                           |
                           v
                    +-------------+
                    |  策略构建器   |
                    | 分组轮动      |
                    | 交易成本建模  |
                    | 多情景压力测试 |
                    +------+------+
                           |
                           v
                    +-------------+
                    |   Web 看板   |
                    | 可视化分析    |
                    | 因子交叉对比  |
                    +-------------+
```

## 核心管线

```
Data Engine → LLM Idea Generator → Validator → Strategy Builder → Storage/UI
```

1. **Data Engine** — 多数据源适配（Tushare / AkShare / YFinance），统一 MultiIndex `(date, asset)` 格式，自动清洗对齐
2. **LLM Idea Generator** — 自然语言描述因子逻辑，LLM 生成可执行 Python 代码，自动语法修正与 AST 安全检查
3. **Validator** — 五维验证体系：IC/IR 分析、分组回测、正交性检验、Newey-West 稳健性检验、单调性评分
4. **Strategy Builder** — 分组轮动 + 交易成本建模 + 多情景回测
5. **Web UI** — Streamlit 交互式前端，因子实验室 + 验证看板 + 回测分析 + 配置管理

## 快速开始

### 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```env
ANTHROPIC_API_KEY=sk-ant-xxx    # Claude API（可选）
OPENAI_API_KEY=sk-xxx           # OpenAI 或兼容 API
TUSHARE_TOKEN=xxx               # Tushare 数据源 Token
```

`config/settings.yaml` 中的验证阈值、LLM 参数等可按需调整。

### 启动

```bash
# CLI — 用自然语言生成因子
python main.py generate "构建一个20日动量因子，IC>0.03"

# CLI — 遗传进化
python main.py evolve --generations 50

# CLI — 启动 Web 界面
python main.py web
# 或
streamlit run web/app.py
```

## CLI 命令

| 命令 | 说明 |
|------|------|
| `python main.py generate <prompt>` | 从自然语言描述生成因子，支持 `--max-iterations` |
| `python main.py evolve` | 遗传编程进化搜索，支持 `--generations` |
| `python main.py validate --factor-file <path>` | 从代码文件验证因子 |
| `python main.py web` | 启动 Streamlit Web 界面 |

全局参数：`--config` 指定配置文件，`-v` 详细输出。

## 因子验证标准

因子通过验证需同时满足：

| 指标 | 阈值 | 说明 |
|------|------|------|
| IC 均值 | > 0.02 | Rank IC（Spearman） |
| IC IR | > 0.5 | IC 信息比率 |
| 分组单调性 | > 0.8 | 多空分组收益单调性得分 |
| 正交化衰减 | < 50% | 去行业/市值后 IC 保留率 |
| Newey-West p 值 | < 0.05 | HAC 稳健 t 检验 |

## 项目结构

```
fineconometrics_factors_mining/
├── main.py                    # CLI 入口
├── factor_factory/
│   ├── core/                  # BaseFactor 抽象基类, Pipeline, Registry
│   ├── data_engine/           # 数据加载器 (Tushare/AkShare/YFinance)
│   ├── idea_generator/        # LLM 客户端, Prompt, 表达式解析, 代码修正
│   ├── validator/             # IC/IR, 分组测试, 正交化, Newey-West
│   ├── strategy_builder/      # 回测引擎, 情景配置
│   ├── storage/               # SQLAlchemy ORM
│   └── web/                   # Streamlit 前端
├── config/                    # YAML 配置文件
├── tests/                     # 测试套件
└── requirements.txt
```

## 开发

```bash
# 运行测试
pytest tests/ -v

# 覆盖率
pytest tests/ -v --cov=factor_factory

# 格式化
ruff check factor_factory/ && black factor_factory/

# 类型检查
mypy factor_factory/
```

## 技术栈

| 层级 | 选型 |
|------|------|
| 数据处理 | NumPy, Pandas, SciPy, statsmodels |
| 金融分析 | Alphalens, empyrical |
| 机器学习 | scikit-learn, LightGBM |
| LLM 集成 | Anthropic SDK, OpenAI SDK, LangChain |
| 数据源 | Tushare, AkShare, YFinance |
| 前端 | Streamlit, Plotly |
| 存储 | SQLite, SQLAlchemy |
| 质量保证 | pytest, Ruff, Black, mypy |
