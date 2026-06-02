"""
Streamlit Web 应用

FactorFactory AI 的前端工作台。
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import asyncio
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 初始化环境变量
try:
    from dotenv import load_dotenv, set_key
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from core.registry import FactorRegistry
from core.base_factor import FactorCategory
from validator import FactorValidator

# 导入图表组件
from components.charts import (
    plot_ic_timeseries,
    plot_ic_distribution,
    plot_ic_decay,
    plot_group_returns,
    plot_long_short_nav,
    create_validation_dashboard,
)
from components.backtest_charts import (
    plot_nav_curve,
    plot_drawdown,
    plot_performance_table,
    plot_group_nav_curves,
    compute_performance_stats,
)

# 页面配置
st.set_page_config(
    page_title="FactorFactory AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .factor-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .config-section {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .status-ok {color: #28a745;font-weight: bold;}
    .status-error {color: #dc3545;font-weight: bold;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化 Session State"""
    if "registry" not in st.session_state:
        st.session_state.registry = FactorRegistry()
    if "validation_history" not in st.session_state:
        st.session_state.validation_history = []
    if "saved_factors" not in st.session_state:
        st.session_state.saved_factors = {}

    # API 配置状态
    if "api_config" not in st.session_state:
        st.session_state.api_config = {
            "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
            "openai_model": os.getenv("OPENAI_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "tushare_token": os.getenv("TUSHARE_TOKEN", ""),
        }

    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None


def save_api_config():
    """保存 API 配置到 .env 文件"""
    env_path = PROJECT_ROOT / ".env"
    config_mapping = {
        "LLM_PROVIDER": st.session_state.api_config.get("llm_provider", "openai"),
        "OPENAI_API_KEY": st.session_state.api_config.get("openai_api_key", ""),
        "OPENAI_BASE_URL": st.session_state.api_config.get("openai_base_url", ""),
        "OPENAI_MODEL": st.session_state.api_config.get("openai_model", ""),
        "ANTHROPIC_API_KEY": st.session_state.api_config.get("anthropic_api_key", ""),
        "TUSHARE_TOKEN": st.session_state.api_config.get("tushare_token", ""),
    }
    if not env_path.exists():
        env_path.touch()
    for key, value in config_mapping.items():
        os.environ[key] = value
        try:
            set_key(str(env_path), key, value)
        except Exception:
            pass


def get_llm_client():
    """获取或创建 LLM 客户端"""
    from idea_generator.llm_client import LLMClient, LLMConfig, LLMProvider
    config = st.session_state.api_config
    if config.get("llm_provider") == "anthropic":
        llm_config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key=config.get("anthropic_api_key"),
            model="claude-sonnet-4",
        )
    else:
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=config.get("openai_api_key"),
            base_url=config.get("openai_base_url"),
            model=config.get("openai_model", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        )
    return LLMClient(config=llm_config)


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("## 🔧 控制面板")
        st.markdown("### 数据源")
        data_source = st.selectbox("选择数据源", ["AkShare", "Tushare", "YFinance"], index=0)
        st.markdown("### 日期范围")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=date.today() - timedelta(days=365*3))
        with col2:
            end_date = st.date_input("结束日期", value=date.today())
        st.markdown("### 股票池")
        stock_pool = st.selectbox("选择股票池", ["沪深300", "中证500", "全A股"], index=0)
        st.markdown("### LLM 配置")
        llm_provider = st.selectbox("LLM 提供商", ["OpenAI 兼容接口", "Claude"], index=0)
        st.markdown("---")
        st.markdown("### 统计")
        registry = st.session_state.registry
        st.metric("已注册因子", registry.count())
        st.metric("Session 因子", len(st.session_state.saved_factors))
        return {
            "data_source": data_source, "start_date": start_date,
            "end_date": end_date, "stock_pool": stock_pool, "llm_provider": llm_provider,
        }


def render_settings_page():
    """渲染设置页面"""
    st.markdown("## ⚙️ API 配置")
    tab1, tab2, tab3 = st.tabs(["🤖 LLM API", "📊 数据源 API", "📋 配置状态"])

    with tab1:
        st.markdown("### LLM 模型接口配置")
        llm_provider = st.radio(
            "选择 LLM 提供商",
            ["OpenAI 兼容接口", "Anthropic Claude"],
            index=0 if st.session_state.api_config.get("llm_provider") != "anthropic" else 1,
            horizontal=True,
        )
        st.session_state.api_config["llm_provider"] = "openai" if llm_provider == "OpenAI 兼容接口" else "anthropic"

        if llm_provider == "OpenAI 兼容接口":
            col1, col2 = st.columns([2, 1])
            with col1:
                openai_base_url = st.text_input(
                    "API Endpoint",
                    value=st.session_state.api_config.get("openai_base_url", "http://10.13.66.5:20165/v1"),
                )
            with col2:
                openai_model = st.text_input(
                    "模型名称",
                    value=st.session_state.api_config.get("openai_model", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
                )
            openai_api_key = st.text_input("API Key", value=st.session_state.api_config.get("openai_api_key", ""), type="password")
            st.session_state.api_config["openai_base_url"] = openai_base_url
            st.session_state.api_config["openai_model"] = openai_model
            st.session_state.api_config["openai_api_key"] = openai_api_key
        else:
            anthropic_api_key = st.text_input("Anthropic API Key", value=st.session_state.api_config.get("anthropic_api_key", ""), type="password")
            st.session_state.api_config["anthropic_api_key"] = anthropic_api_key

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔗 测试连接", type="secondary", use_container_width=True):
                with st.spinner("测试中..."):
                    try:
                        client = get_llm_client()
                        async def test():
                            return await client.chat([{"role": "user", "content": "Say OK"}])
                        response = asyncio.run(test())
                        st.success(f"连接成功！响应: {response[:50]}...")
                    except Exception as e:
                        st.error(f"连接失败: {e}")
        with col2:
            if st.button("💾 保存配置", type="primary", use_container_width=True):
                save_api_config()
                st.success("配置已保存！")

    with tab2:
        st.markdown("### 数据源 API 配置")
        st.markdown("#### Tushare (A股数据)")
        tushare_token = st.text_input("Tushare Token", value=st.session_state.api_config.get("tushare_token", ""), type="password")
        st.session_state.api_config["tushare_token"] = tushare_token
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔗 测试 Tushare", type="secondary", use_container_width=True):
                if tushare_token:
                    try:
                        import tushare as ts
                        # 清除代理
                        os.environ.pop('HTTP_PROXY', None)
                        os.environ.pop('HTTPS_PROXY', None)
                        ts.set_token(tushare_token)
                        pro = ts.pro_api()
                        df = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240110')
                        st.success(f"Tushare 连接成功！获取到 {len(df)} 条数据")
                    except Exception as e:
                        st.error(f"Tushare 连接失败: {e}")
                else:
                    st.warning("请输入 Tushare Token")
        with col2:
            if st.button("💾 保存数据源配置", type="primary", use_container_width=True):
                save_api_config()
                st.success("配置已保存！")
        st.markdown("---")
        st.markdown("#### AkShare / YFinance (免费)")
        st.info("无需配置，直接可用。")

    with tab3:
        st.markdown("### 当前配置状态")
        config = st.session_state.api_config
        st.markdown("#### LLM API")
        if config.get("llm_provider") == "openai":
            status = "✅" if config.get("openai_api_key") and config.get("openai_base_url") else "❌"
            st.markdown(f"OpenAI 兼容接口 {status}")
            st.markdown(f"Endpoint: `{config.get('openai_base_url', '未配置')}`")
            st.markdown(f"Model: `{config.get('openai_model', '未配置')}`")
        else:
            status = "✅" if config.get("anthropic_api_key") else "❌"
            st.markdown(f"Anthropic Claude {status}")
        st.markdown("---")
        st.markdown("#### 数据源")
        for name, configured in [("Tushare", bool(config.get("tushare_token"))), ("AkShare", True), ("YFinance", True)]:
            st.markdown(f"**{name}**: {'✅ 已配置' if configured else '❌ 未配置'}")


def render_factor_lab(config: dict):
    """渲染因子实验室"""
    st.markdown("## 🧪 因子实验室")
    tab1, tab2, tab3 = st.tabs(["自然语言生成", "代码编辑", "演化生成"])

    with tab1:
        st.markdown("### 输入因子思路")
        prompt = st.text_area("因子描述", placeholder="例如：生成一个基于成交量和价格波动的动量因子...", height=150)
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 生成因子", type="primary", use_container_width=True):
                if prompt:
                    api_config = st.session_state.api_config
                    if api_config.get("llm_provider") == "openai":
                        if not api_config.get("openai_api_key") or not api_config.get("openai_base_url"):
                            st.error("请先在「设置」页面配置 LLM API")
                            st.stop()
                    else:
                        if not api_config.get("anthropic_api_key"):
                            st.error("请先在「设置」页面配置 Anthropic API Key")
                            st.stop()
                    with st.spinner("正在生成因子..."):
                        try:
                            client = get_llm_client()
                            async def generate():
                                idea = await client.generate_idea(prompt)
                                code = await client.generate_code(idea)
                                return idea, code
                            idea, code = asyncio.run(generate())
                            st.success("因子生成成功！")
                            st.markdown("#### 因子思路")
                            st.markdown(idea)
                            st.markdown("#### 生成代码")
                            st.code(code, language="python")
                            st.session_state.generated_code = code
                        except Exception as e:
                            st.error(f"生成失败: {e}")
                else:
                    st.warning("请输入因子描述")

    with tab2:
        st.markdown("### 代码编辑器")
        default_code = st.session_state.get("generated_code", """from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency
import pandas as pd
import numpy as np

class MomentumFactor(BaseFactor):
    '''20日动量因子'''

    def __init__(self):
        metadata = FactorMetadata(
            name="momentum_20d",
            category=FactorCategory.MOMENTUM,
            description="20日动量因子，计算过去20日的收益率",
            formula="close / close.shift(20) - 1",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        '''
        计算因子值

        Args:
            data: MultiIndex DataFrame (date, asset) 包含 close 列

        Returns:
            因子值 Series，索引为 (date, asset)
        '''
        close = data['close']
        # 按资产分组计算动量
        momentum = close.groupby(level='asset').transform(
            lambda x: x.pct_change(20)
        )
        return momentum

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        '''验证输入数据'''
        return 'close' in data.columns
""")
        code = st.text_area("Python 代码", value=default_code, height=400)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ 解析并验证", use_container_width=True):
                with st.spinner("正在解析代码..."):
                    try:
                        from idea_generator.parser import FactorParser
                        from idea_generator.code_fixer import CodeFixer

                        # 先用 CodeFixer 预处理
                        fixer = CodeFixer()
                        fixed_code, fixes = fixer.fix(code)
                        if fixes:
                            st.info(f"代码自动修复: {', '.join(fixes)}")

                        parser = FactorParser()
                        result = parser.parse(fixed_code)
                        if result.success:
                            st.success(f"代码解析成功！因子名: {result.metadata.name}")
                            st.session_state.current_factor = result.factor_instance
                            st.session_state.current_factor_code = code
                            st.json(result.metadata.to_dict())
                        else:
                            for err in result.errors:
                                st.error(err)
                    except Exception as e:
                        st.error(f"解析失败: {e}")

        with col2:
            if st.button("💾 保存因子", type="primary", use_container_width=True):
                try:
                    from idea_generator.parser import FactorParser
                    from idea_generator.code_fixer import CodeFixer

                    # 先用 CodeFixer 预处理
                    fixer = CodeFixer()
                    fixed_code, fixes = fixer.fix(code)

                    parser = FactorParser()
                    result = parser.parse(fixed_code)
                    if result.success:
                        # 保存到注册中心
                        factor_id = st.session_state.registry.register(
                            factor=result.factor_instance,
                            validation_result=None,
                            code=code,
                        )
                        # 同时保存到 session
                        st.session_state.saved_factors[result.metadata.name] = {
                            "factor": result.factor_instance,
                            "code": fixed_code,
                            "metadata": result.metadata,
                        }
                        st.success(f"因子已保存！ID: {factor_id}, 名称: {result.metadata.name}")
                        st.rerun()
                    else:
                        for err in result.errors:
                            st.error(err)
                except Exception as e:
                    st.error(f"保存失败: {e}")

        with col3:
            if st.button("🗑️ 清空", use_container_width=True):
                st.session_state.generated_code = None
                st.rerun()

    with tab3:
        st.markdown("### 遗传规划演化")
        st.info("功能开发中...")


def render_validation_dashboard(config: dict):
    """渲染验证看板"""
    st.markdown("## 📊 验证看板")

    # 刷新按钮
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 刷新", use_container_width=True):
            st.rerun()

    # 获取因子列表
    registry = st.session_state.registry
    db_factors = registry.list_all(limit=100)
    session_factors = st.session_state.saved_factors

    all_factors = []
    # 合并数据库因子
    for f in db_factors:
        all_factors.append({
            "source": "database",
            "name": f.get("metadata", {}).get("name", "Unknown"),
            "data": f,
        })
    # 合并 session 因子
    for name, f in session_factors.items():
        all_factors.append({
            "source": "session",
            "name": name,
            "data": f,
        })

    if not all_factors:
        st.info("暂无已保存因子，请在「因子实验室」生成并保存因子。")
        return

    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总因子数", len(all_factors))
    with col2:
        st.metric("数据库因子", len(db_factors))
    with col3:
        st.metric("Session 因子", len(session_factors))
    with col4:
        categories = set(f["data"].get("metadata", {}).get("category", "未知") for f in all_factors if f["source"] == "database")
        st.metric("因子类别", len(categories))

    st.markdown("---")

    # 因子列表
    st.markdown("### 因子列表")

    for factor_info in all_factors:
        source = factor_info["source"]
        name = factor_info["name"]
        data = factor_info["data"]

        if source == "database":
            metadata = data.get("metadata", {})
            validation = data.get("validation", {})
            with st.expander(f"📁 **{name}** (数据库) - {metadata.get('description', 'N/A')[:50]}..."):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**分类**: {metadata.get('category', 'N/A')}")
                    st.markdown(f"**公式**: `{metadata.get('formula', 'N/A')}`")
                    st.markdown(f"**作者**: {metadata.get('author', 'N/A')}")
                with col2:
                    if validation:
                        st.metric("IC", f"{validation.get('ic_mean', 0):.4f}")
                        st.metric("IR", f"{validation.get('ic_ir', 0):.4f}")
                        st.metric("单调性", f"{validation.get('monotonicity_score', 0):.2f}")
        else:
            metadata = data.get("metadata")
            with st.expander(f"💾 **{name}** (Session) - {metadata.description[:50] if metadata else ''}..."):
                st.markdown(f"**分类**: {metadata.category.value if metadata else 'N/A'}")
                st.markdown(f"**公式**: `{metadata.formula if metadata else 'N/A'}`")
                if st.button(f"📊 运行验证", key=f"val_{name}"):
                    st.session_state.selected_factor_name = name
                    st.switch_page("pages/2_Validate.py")


def render_backtest_page(config: dict):
    """渲染回测页面"""
    st.markdown("## 📈 因子回测")

    # 获取因子来源选项
    registry = st.session_state.registry
    db_factors = registry.list_all(limit=100)
    session_factors = list(st.session_state.saved_factors.keys())

    if not db_factors and not session_factors:
        st.warning("暂无可回测因子")
        st.info("操作步骤：1. 在「因子实验室」生成或编辑代码 → 2. 点击「解析并验证」→ 3. 点击「保存因子」")
        return

    # 选择因子来源
    source = st.radio("选择因子来源", ["Session 因子", "数据库因子"], horizontal=True)

    if source == "Session 因子":
        if not session_factors:
            st.warning("Session 中暂无因子")
            return
        selected_factor = st.selectbox("选择因子", session_factors)
        factor_data = st.session_state.saved_factors.get(selected_factor)
    else:
        if not db_factors:
            st.warning("数据库中暂无因子")
            return
        # 显示数据库因子列表
        factor_options = {f["metadata"]["name"]: f for f in db_factors}
        selected_name = st.selectbox("选择因子", list(factor_options.keys()))
        db_factor_info = factor_options.get(selected_name)

        if db_factor_info:
            # 从数据库加载因子代码并重新解析
            factor_id = db_factor_info.get("id")
            code = registry.get_code(factor_id)

            if code:
                try:
                    from idea_generator.parser import FactorParser
                    from idea_generator.code_fixer import CodeFixer

                    fixer = CodeFixer()
                    fixed_code, _ = fixer.fix(code)

                    parser = FactorParser()
                    result = parser.parse(fixed_code)

                    if result.success:
                        factor_data = {
                            "factor": result.factor_instance,
                            "code": fixed_code,
                            "metadata": result.factor_instance.metadata,
                        }
                    else:
                        st.error(f"因子解析失败: {result.errors}")
                        return
                except Exception as e:
                    st.error(f"加载因子失败: {e}")
                    return
            else:
                st.error("未找到因子代码")
                return
        else:
            return

    if factor_data:
        st.markdown(f"**因子**: {factor_data['metadata'].name}")
        with st.expander("查看因子代码"):
            st.code(factor_data["code"], language="python")

        # 回测场景选择
        st.markdown("### 回测场景")
        scenario_type = st.selectbox(
            "选择交易场景",
            ["周频选股", "月频选股", "日频高频", "固定标的择时", "自定义配置"],
            index=0,
            help="选择适合你的交易策略类型"
        )

        if scenario_type == "固定标的择时":
            st.info("固定标的择时：在固定股票池中使用因子信号进行择时")
            custom_symbols = st.text_input(
                "输入股票代码（逗号分隔）",
                value="600519,000858,600036",
                help="例如: 600519,000858,600036"
            )

        # 回测参数
        st.markdown("### 回测参数")

        if scenario_type == "自定义配置":
            col1, col2, col3 = st.columns(3)
            with col1:
                n_groups = st.selectbox("分组数量", [3, 5, 10], index=1)
            with col2:
                holding_period = st.selectbox("持有期(天)", [1, 5, 10, 20], index=1)
            with col3:
                lookback_years = st.selectbox("回测年限", [1, 2, 3], index=1)

            # 交易成本参数
            st.markdown("#### 交易成本")
            col1, col2, col3 = st.columns(3)
            with col1:
                commission_rate = st.number_input("佣金费率", value=0.0003, format="%.4f", help="单边，默认万三")
            with col2:
                stamp_duty = st.number_input("印花税率", value=0.001, format="%.4f", help="卖出时，默认千一")
            with col3:
                slippage = st.number_input("滑点率", value=0.0005, format="%.4f", help="单边，默认万五")
        else:
            # 预设场景使用默认参数
            n_groups = 5
            holding_period = {"周频选股": 5, "月频选股": 20, "日频高频": 1, "固定标的择时": 5}.get(scenario_type, 5)
            commission_rate = 0.0003
            stamp_duty = 0.001
            slippage = 0.0005
            lookback_years = 2

            col1, col2 = st.columns(2)
            with col1:
                lookback_years = st.selectbox("回测年限", [1, 2, 3], index=1)
            with col2:
                if scenario_type in ["周频选股", "月频选股", "日频高频"]:
                    n_groups = st.selectbox("分组数量", [3, 5, 10], index=1)

        if st.button("🚀 运行回测", type="primary"):
            run_backtest_with_scenario(
                factor_data, scenario_type, n_groups, holding_period,
                commission_rate, stamp_duty, slippage, lookback_years,
                custom_symbols if scenario_type == "固定标的择时" else None
            )


def run_backtest_with_scenario(
    factor_data: dict,
    scenario_type: str,
    n_groups: int,
    holding_period: int,
    commission_rate: float,
    stamp_duty: float,
    slippage: float,
    lookback_years: int,
    custom_symbols: Optional[str] = None,
):
    """使用场景配置执行回测"""
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    from strategy_builder.scenario_config import (
        BacktestScenarioConfig,
        TradingScenario,
        RebalanceFrequency,
        StrategyType,
        TradingCostConfig,
        UniverseConfig,
        ScenarioPresets,
        ScenarioExecutor,
    )

    with st.spinner("正在加载数据..."):
        try:
            # 1. 加载数据 - 优先使用 AkShare (免费无需Token)
            from data_engine.loaders import AkShareLoader, TushareLoader

            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_years * 365)

            st.info(f"正在加载 {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} 的沪深300数据...")

            # 尝试 AkShare (免费)
            try:
                loader = AkShareLoader()
                data = asyncio.run(loader.load_hs300(
                    start_date=start_date.strftime('%Y%m%d'),
                    end_date=end_date.strftime('%Y%m%d'),
                    n_stocks=30,  # 限制股票数量加快加载
                ))
                if data.empty:
                    raise Exception("AkShare 返回空数据")
                st.info("使用 AkShare 数据源 (免费)")
            except Exception as e:
                # 备用 Tushare
                st.warning(f"AkShare 加载失败: {e}，尝试 Tushare...")
                tushare_token = st.session_state.api_config.get("tushare_token")
                if tushare_token:
                    loader = TushareLoader(token=tushare_token)
                    data = asyncio.run(loader.load_hs300(
                        start_date=start_date.strftime('%Y%m%d'),
                        end_date=end_date.strftime('%Y%m%d'),
                    ))
                else:
                    st.error("数据加载失败，AkShare 不可用且未配置 Tushare Token")
                    return

            if data is None or data.empty:
                st.error("数据加载失败，请检查网络连接或数据源配置")
                return

            st.success(f"数据加载成功: {len(data)} 条记录, {data.index.get_level_values('asset').nunique()} 只股票")

        except Exception as e:
            st.error(f"数据加载失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    with st.spinner("正在计算因子值..."):
        try:
            # 2. 计算因子值
            factor = factor_data["factor"]
            factor_values = factor.compute(data)

            if factor_values is None or factor_values.empty:
                st.error("因子计算返回空结果")
                return

            st.success(f"因子计算完成: {len(factor_values)} 个值")

        except Exception as e:
            st.error(f"因子计算失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    with st.spinner("正在运行回测..."):
        try:
            # 3. 计算收益率
            close = data['close'].unstack(level='asset')
            returns = close.pct_change()

            # 确保数据为数值类型，处理异常值
            returns = returns.apply(pd.to_numeric, errors='coerce')
            returns = returns.clip(lower=-0.2, upper=0.2)  # 限制单日涨跌幅
            returns = returns.fillna(0)

            # 4. 将因子值转换为 DataFrame 格式
            if isinstance(factor_values, pd.Series):
                factor_df = factor_values.unstack(level='asset')
            else:
                factor_df = factor_values

            # 确保因子值为数值类型
            factor_df = factor_df.apply(pd.to_numeric, errors='coerce')

            # 对齐因子值和收益率
            common_dates = factor_df.index.intersection(returns.index)
            common_assets = factor_df.columns.intersection(returns.columns)
            factor_df = factor_df.loc[common_dates, common_assets]
            returns = returns.loc[common_dates, common_assets]

            # 5. 构建场景配置
            trading_cost = TradingCostConfig(
                commission_rate=commission_rate,
                stamp_duty=stamp_duty,
                slippage=slippage,
            )

            if scenario_type == "周频选股":
                config = ScenarioPresets.weekly_stock_selection()
                config.trading_cost = trading_cost
                config.n_groups = n_groups
            elif scenario_type == "月频选股":
                config = ScenarioPresets.monthly_stock_selection()
                config.trading_cost = trading_cost
                config.n_groups = n_groups
            elif scenario_type == "日频高频":
                config = ScenarioPresets.high_frequency_trading()
                config.trading_cost = trading_cost
                config.n_groups = n_groups
            elif scenario_type == "固定标的择时":
                symbols = [s.strip() for s in custom_symbols.split(",") if s.strip()]
                config = ScenarioPresets.fixed_universe_timing(symbols)
                config.trading_cost = trading_cost
            else:
                # 自定义配置
                config = BacktestScenarioConfig(
                    scenario=TradingScenario.CUSTOM,
                    n_groups=n_groups,
                    rebalance_frequency=RebalanceFrequency.CUSTOM,
                    custom_rebalance_days=holding_period,
                    trading_cost=trading_cost,
                )

            # 6. 执行回测
            executor = ScenarioExecutor(config)
            result = executor.run(factor_df, returns)

            st.success("回测完成!")

        except Exception as e:
            st.error(f"回测运行失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # 7. 显示结果
    st.markdown("### 回测结果")
    st.info(f"场景: {result['scenario_description']}")

    # 绩效指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("年化收益", f"{result['stats'].get('annual_return', 0):.2%}")
    with col2:
        st.metric("夏普比率", f"{result['stats'].get('sharpe_ratio', 0):.2f}")
    with col3:
        st.metric("最大回撤", f"{result['stats'].get('max_drawdown', 0):.2%}")
    with col4:
        st.metric("胜率", f"{result['stats'].get('win_rate', 0):.2%}")

    # 换仓统计
    st.markdown("#### 换仓统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均换手率", f"{result['turnover_stats'].get('avg_turnover', 0):.2%}")
    with col2:
        st.metric("最大换手率", f"{result['turnover_stats'].get('max_turnover', 0):.2%}")
    with col3:
        st.metric("单次换仓成本", f"{result['turnover_stats'].get('total_cost_rate', 0):.4%}")

    st.markdown("---")

    # 净值曲线
    from web.components.backtest_charts import (
        plot_nav_curve, plot_drawdown, plot_group_nav_curves, plot_performance_table
    )

    nav_curve = result.get('nav_curve')
    if nav_curve is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 策略净值曲线")
            fig_nav = plot_nav_curve(nav_curve)
            st.plotly_chart(fig_nav, use_container_width=True)

        with col2:
            st.markdown("#### 回撤曲线")
            fig_dd = plot_drawdown(nav_curve)
            st.plotly_chart(fig_dd, use_container_width=True)

    # 分组净值
    group_navs = result.get('group_navs')
    if group_navs and len(group_navs) > 1:
        st.markdown("#### 各分组净值曲线")
        fig_groups = plot_group_nav_curves(group_navs)
        st.plotly_chart(fig_groups, use_container_width=True)

    # 绩效表格
    st.markdown("#### 绩效统计")
    fig_stats = plot_performance_table(result['stats'])
    st.plotly_chart(fig_stats, use_container_width=True)


def main():
    """主函数"""
    init_session_state()
    st.markdown('<p class="main-header">FactorFactory AI</p>', unsafe_allow_html=True)
    st.markdown("全自动金融因子挖掘与验证系统")
    config = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(["🧪 因子实验室", "📊 验证看板", "📈 回测", "⚙️ 设置"])

    with tab1:
        render_factor_lab(config)
    with tab2:
        render_validation_dashboard(config)
    with tab3:
        render_backtest_page(config)
    with tab4:
        render_settings_page()

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray;'>FactorFactory AI v0.2.0 | Powered by Qwen3</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
