# FactorFactory AI

全自动金融因子挖掘与验证系统 — 从自然语言描述到可部署量化策略的全流程闭环。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)]()

## 这是什么？（给人话看的）

这是一个 **AI 量化研究员**。你把想法告诉它，它帮你验证这个想法能不能赚钱。

举个例子：你怀疑"最近涨得猛的股票接下来还会继续涨"，但你不知道这个规律到底靠不靠谱。传统做法是：找数据、写代码、跑统计、看结果，来回折腾好几天。而这个系统，你只需要一句话——

> "构建一个20日动量因子，IC > 0.03"

它就会自动帮你把数据拉下来、把因子算出来、用学术界最严格的标准做验证，最后告诉你：这个想法行还是不行，有多靠谱。

**它能做什么：**
- 听懂自然语言 —— 你不用写代码，用中文描述你的交易想法就行
- 自动挖因子 —— 给定一个方向（比如"动量"），AI 会自动生成几十上百个变体，挑出最好的
- 严格验证 —— 不是随便回测一下糊弄人，而是用金融学术界公认的 Fama-MacBeth 方法 + Newey-West 稳健性检验
- 生成策略 —— 通过验证的因子，可以一键变成可交易的轮动策略

**它不能做什么（至少目前）：**
- 不能保证赚钱 —— 历史表现好不代表未来一定好，这只是一个研究工具
- 不会自己开仓 —— 这是个挖掘和验证工具，不直接下单

## 全局流程

```
                               +---------------------+
                               |      你 的想法       |
                               | "20日动量会不会好用？" |
                               +----------+----------+
                                          |
                                          v
                               +---------------------+
                               |    数据引擎          |
                               | Tushare/AkShare/    |
                               | YFinance 多源拉数据  |
                               | 清洗、对齐、去极值    |
                               +----------+----------+
                                          |
                                          v
                               +---------------------+
                               |   LLM 因子生成器     |
                               | 自然语言 -> Python代码 |
                               | AST安全检查 + 自动修bug|
                               +----------+----------+
                                          |
                                          v
                               +---------------------+
                               |     验证引擎         |
                               | IC分析 | 分组回测     |
                               | 去行业/市值干扰      |
                               | Newey-West 稳健t检验 |
                               | 单调性评分           |
                               +----------+----------+
                                          |
                           +--------------+--------------+
                           |                             |
                           v                             v
                    +-------------+              +-------------+
                    |  t >= 3.0   |              |  t < 3.0    |
                    | 通过！入库   |              | 不通过，反馈  |
                    | 解锁回测引擎 |              | LLM 继续改进 |
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
                    | 结果可视化    |
                    | 因子对比分析  |
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
