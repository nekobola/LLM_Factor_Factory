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
- 严格验证 —— 五维验证体系（IC/IR 分析、分组回测、正交化检验、Newey-West 稳健性检验、单调性评分），拒绝简单回测
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
                    |  五项全过    |              | 任一项未过   |
                    | 通过，入库   |              | 未通过，反馈  |
                    | 解锁策略构建 |              | LLM 定向改进 |
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

## 使用方式

系统提供两种使用模式，适用于不同场景。

### CLI 模式（单因子生成与验证）

适用场景：有明确的因子思路，快速验证一个想法。

```bash
# 1. 用自然语言生成因子
python main.py generate "构建20日动量因子，IC期望大于0.03"

# 2. 从已有的因子代码文件验证
python main.py validate --factor-file factors/my_factor.py
```

执行后终端输出流程日志，最终打印验证结果：IC 均值、IC IR、单调性得分、Newey-West p 值，以及通过/未通过判定。

### Web 模式（交互式工作台）

适用场景：反复实验、批量管理、可视化分析。

```bash
python main.py web
# 或
streamlit run web/app.py
```

浏览器打开 `http://localhost:8501`，工作台包含四个页面：

| 页面 | 用途 |
|------|------|
| **因子实验室** | 自然语言生成因子、代码编辑器、解析验证 |
| **验证看板** | 因子库浏览、批量 IC/IR/单调性指标对比 |
| **回测** | 分组轮动回测、多情景压力测试（周频/月频/日频/择时） |
| **设置** | LLM API 和数据源 Token 配置 |

## 预期效果

### 验证输出

因子通过验证后会看到类似输出：

```
Factor generated: momentum_20d
IC: 0.0384, IR: 0.72
Validation passed: True
```

Web 端则展示完整的验证仪表盘：IC 时序图、IC 分布直方图、IC 衰减曲线、分组收益柱状图、多空净值曲线。

### 回测输出

在 Web 回测页面运行后，产出两列面板：

- **绩效指标卡片** — 年化收益、夏普比率、最大回撤、胜率
- **净值曲线** — 策略净值 vs 基准，回撤曲线
- **分组净值** — 各分组的累计收益曲线，用于检验单调性
- **换仓统计** — 平均换手率、单次换仓成本

### 什么算"通过"

因子通过验证需同时满足五项阈值（见下方表格）。任一项未通过，验证器会返回具体未达标项及其当前值，LLM 据此调整因子方向。例如："IC IR 当前 0.42，未达 0.5 阈值，建议增加信号平滑处理"。

## CLI 命令参考

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
