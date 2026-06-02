"""
交易场景配置模块

支持多种交易场景：
1. 固定频率选股 - 定期调仓选股
2. 固定标的择时 - 固定池子的择时信号
3. 灵活配置 - 完全自定义参数
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Callable
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from loguru import logger


class TradingScenario(str, Enum):
    """交易场景类型"""
    FIXED_FREQUENCY_SELECTION = "fixed_frequency_selection"  # 固定频率选股
    FIXED_UNIVERSE_TIMING = "fixed_universe_timing"          # 固定标的择时
    CUSTOM = "custom"                                         # 自定义场景


class RebalanceFrequency(str, Enum):
    """调仓频率"""
    DAILY = "daily"           # 日频
    WEEKLY = "weekly"         # 周频
    BIWEEKLY = "biweekly"     # 双周频
    MONTHLY = "monthly"       # 月频
    QUARTERLY = "quarterly"   # 季频
    CUSTOM = "custom"         # 自定义天数


class StrategyType(str, Enum):
    """策略类型"""
    LONG_ONLY_TOP = "long_only_top"           # 仅做多最高组
    LONG_ONLY_BOTTOM = "long_only_bottom"     # 仅做多最低组
    LONG_SHORT = "long_short"                 # 多空策略
    MARKET_NEUTRAL = "market_neutral"         # 市场中性
    CUSTOM = "custom"                         # 自定义


@dataclass
class TradingCostConfig:
    """
    交易成本配置

    Attributes:
        commission_rate: 佣金费率（单边）
        stamp_duty: 印花税率（卖出时）
        slippage: 滑点率（单边）
        min_commission: 最低佣金
    """
    commission_rate: float = 0.0003  # 万三
    stamp_duty: float = 0.001        # 千一
    slippage: float = 0.0005         # 万五
    min_commission: float = 5.0      # 最低5元

    @classmethod
    def zero_cost(cls) -> 'TradingCostConfig':
        """零交易成本"""
        return cls(commission_rate=0, stamp_duty=0, slippage=0, min_commission=0)

    @classmethod
    def a_share_default(cls) -> 'TradingCostConfig':
        """A股默认成本"""
        return cls(commission_rate=0.0003, stamp_duty=0.001, slippage=0.0005)

    @classmethod
    def high_cost(cls) -> 'TradingCostConfig':
        """高成本环境"""
        return cls(commission_rate=0.001, stamp_duty=0.003, slippage=0.002)

    def total_single_turnover_cost(self) -> float:
        """单次完整换仓成本（买入+卖出）"""
        # 买入：佣金 + 滑点
        buy_cost = self.commission_rate + self.slippage
        # 卖出：佣金 + 印花税 + 滑点
        sell_cost = self.commission_rate + self.stamp_duty + self.slippage
        return buy_cost + sell_cost


@dataclass
class UniverseConfig:
    """
    资产池配置

    Attributes:
        universe_type: 资产池类型
        symbols: 固定股票列表
        index_constituents: 指数成分股（如 "hs300"）
        top_n: 选取前N只
        exclude_st: 是否排除ST股票
        exclude_new: 排除上市不足N天的新股
    """
    universe_type: str = "index"  # "fixed", "index", "all"
    symbols: Optional[List[str]] = None
    index_constituents: str = "hs300"
    top_n: int = 50
    exclude_st: bool = True
    exclude_new_days: int = 60  # 排除上市不足60天的新股


@dataclass
class BacktestScenarioConfig:
    """
    回测场景配置

    这是核心配置类，封装所有回测参数。

    Example:
        # 固定频率选股
        config = BacktestScenarioConfig(
            scenario=TradingScenario.FIXED_FREQUENCY_SELECTION,
            rebalance_frequency=RebalanceFrequency.WEEKLY,
            n_groups=5,
            strategy_type=StrategyType.LONG_ONLY_TOP,
        )

        # 固定标的择时
        config = BacktestScenarioConfig(
            scenario=TradingScenario.FIXED_UNIVERSE_TIMING,
            universe=UniverseConfig(symbols=["600519", "000858"]),
            rebalance_frequency=RebalanceFrequency.DAILY,
        )
    """
    # 场景类型
    scenario: TradingScenario = TradingScenario.FIXED_FREQUENCY_SELECTION

    # 调仓配置
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY
    custom_rebalance_days: int = 5  # 自定义调仓天数

    # 分组配置
    n_groups: int = 5

    # 策略类型
    strategy_type: StrategyType = StrategyType.LONG_ONLY_TOP

    # 交易成本
    trading_cost: TradingCostConfig = field(default_factory=TradingCostConfig)

    # 资产池
    universe: UniverseConfig = field(default_factory=UniverseConfig)

    # 时间范围
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    lookback_years: int = 2

    # 其他参数
    allow_short: bool = False  # 是否允许做空
    position_limit: float = 1.0  # 单只股票仓位上限
    cash_buffer: float = 0.05  # 现金缓冲

    # 高级配置
    sector_neutral: bool = False  # 行业中性
    size_neutral: bool = False    # 市值中性
    custom_weights: Optional[Callable] = None  # 自定义权重函数

    def get_holding_period(self) -> int:
        """获取持有期（天数）"""
        freq_map = {
            RebalanceFrequency.DAILY: 1,
            RebalanceFrequency.WEEKLY: 5,
            RebalanceFrequency.BIWEEKLY: 10,
            RebalanceFrequency.MONTHLY: 20,
            RebalanceFrequency.QUARTERLY: 60,
            RebalanceFrequency.CUSTOM: self.custom_rebalance_days,
        }
        return freq_map.get(self.rebalance_frequency, 5)

    def get_strategy_description(self) -> str:
        """获取策略描述"""
        scenario_names = {
            TradingScenario.FIXED_FREQUENCY_SELECTION: "固定频率选股",
            TradingScenario.FIXED_UNIVERSE_TIMING: "固定标的择时",
            TradingScenario.CUSTOM: "自定义场景",
        }

        strategy_names = {
            StrategyType.LONG_ONLY_TOP: "做多因子最高组",
            StrategyType.LONG_ONLY_BOTTOM: "做多因子最低组",
            StrategyType.LONG_SHORT: "多空策略",
            StrategyType.MARKET_NEUTRAL: "市场中性",
            StrategyType.CUSTOM: "自定义策略",
        }

        return f"{scenario_names[self.scenario]} - {strategy_names[self.strategy_type]}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "scenario": self.scenario.value,
            "rebalance_frequency": self.rebalance_frequency.value,
            "holding_period": self.get_holding_period(),
            "n_groups": self.n_groups,
            "strategy_type": self.strategy_type.value,
            "trading_cost": {
                "commission_rate": self.trading_cost.commission_rate,
                "stamp_duty": self.trading_cost.stamp_duty,
                "slippage": self.trading_cost.slippage,
            },
            "universe": {
                "type": self.universe.universe_type,
                "top_n": self.universe.top_n,
            },
            "allow_short": self.allow_short,
        }


# ============================================================
# 预设场景配置
# ============================================================

class ScenarioPresets:
    """预设场景配置"""

    @staticmethod
    def weekly_stock_selection() -> BacktestScenarioConfig:
        """周频选股策略"""
        return BacktestScenarioConfig(
            scenario=TradingScenario.FIXED_FREQUENCY_SELECTION,
            rebalance_frequency=RebalanceFrequency.WEEKLY,
            n_groups=5,
            strategy_type=StrategyType.LONG_ONLY_TOP,
            trading_cost=TradingCostConfig.a_share_default(),
        )

    @staticmethod
    def monthly_stock_selection() -> BacktestScenarioConfig:
        """月频选股策略"""
        return BacktestScenarioConfig(
            scenario=TradingScenario.FIXED_FREQUENCY_SELECTION,
            rebalance_frequency=RebalanceFrequency.MONTHLY,
            n_groups=5,
            strategy_type=StrategyType.LONG_ONLY_TOP,
            trading_cost=TradingCostConfig.a_share_default(),
        )

    @staticmethod
    def fixed_universe_timing(
        symbols: List[str],
        rebalance_days: int = 1,
    ) -> BacktestScenarioConfig:
        """固定标的择时策略"""
        return BacktestScenarioConfig(
            scenario=TradingScenario.FIXED_UNIVERSE_TIMING,
            universe=UniverseConfig(
                universe_type="fixed",
                symbols=symbols,
            ),
            rebalance_frequency=RebalanceFrequency.CUSTOM,
            custom_rebalance_days=rebalance_days,
            n_groups=1,  # 择时不需要分组
            strategy_type=StrategyType.LONG_ONLY_TOP,
            trading_cost=TradingCostConfig.a_share_default(),
        )

    @staticmethod
    def long_short_strategy() -> BacktestScenarioConfig:
        """多空策略"""
        return BacktestScenarioConfig(
            scenario=TradingScenario.FIXED_FREQUENCY_SELECTION,
            rebalance_frequency=RebalanceFrequency.WEEKLY,
            n_groups=5,
            strategy_type=StrategyType.LONG_SHORT,
            allow_short=True,
            trading_cost=TradingCostConfig.a_share_default(),
        )

    @staticmethod
    def high_frequency_trading() -> BacktestScenarioConfig:
        """高频交易策略"""
        return BacktestScenarioConfig(
            scenario=TradingScenario.FIXED_FREQUENCY_SELECTION,
            rebalance_frequency=RebalanceFrequency.DAILY,
            n_groups=10,
            strategy_type=StrategyType.LONG_ONLY_TOP,
            trading_cost=TradingCostConfig.high_cost(),
        )


# ============================================================
# 场景执行器
# ============================================================

class ScenarioExecutor:
    """
    场景执行器

    根据配置执行回测。

    Example:
        config = ScenarioPresets.weekly_stock_selection()
        executor = ScenarioExecutor(config)
        result = executor.run(factor_values, returns)
    """

    def __init__(self, config: BacktestScenarioConfig):
        self.config = config
        logger.info(f"ScenarioExecutor initialized: {config.get_strategy_description()}")

    def run(
        self,
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        执行回测

        Args:
            factor_values: 因子值 DataFrame
            returns: 收益率 DataFrame
            prices: 价格 DataFrame（可选）

        Returns:
            回测结果字典
        """
        from strategy_builder.backtest import FactorBacktest

        # 获取持有期
        holding_period = self.config.get_holding_period()

        # 根据策略类型决定是否允许做空
        allow_short = self.config.allow_short or (
            self.config.strategy_type == StrategyType.LONG_SHORT
        )

        # 创建回测引擎
        backtest = FactorBacktest(factor_values, returns, prices)

        # 执行回测
        result = backtest.run(
            n_groups=self.config.n_groups,
            holding_period=holding_period,
            commission_rate=self.config.trading_cost.commission_rate,
            stamp_duty=self.config.trading_cost.stamp_duty,
            slippage=self.config.trading_cost.slippage,
            allow_short=allow_short,
        )

        # 添加配置信息到结果
        result_dict = {
            "success": True,
            "stats": result.stats,
            "turnover_stats": result.turnover_stats,
            "nav_curve": result.nav_curve,
            "group_navs": result.group_navs,
            "config": self.config.to_dict(),
            "scenario_description": self.config.get_strategy_description(),
        }

        return result_dict

    def get_rebalance_dates(self, all_dates: List) -> List:
        """获取调仓日期"""
        holding_period = self.config.get_holding_period()
        return [all_dates[i] for i in range(0, len(all_dates), holding_period)]


# ============================================================
# 便捷函数
# ============================================================

def quick_backtest(
    factor_values: pd.DataFrame,
    returns: pd.DataFrame,
    scenario: str = "weekly",
    n_groups: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """
    快速回测接口

    Args:
        factor_values: 因子值
        returns: 收益率
        scenario: 场景名称 ("daily", "weekly", "monthly", "long_short")
        n_groups: 分组数
        **kwargs: 其他参数

    Returns:
        回测结果
    """
    # 选择预设配置
    if scenario == "daily":
        config = ScenarioPresets.high_frequency_trading()
        config.n_groups = n_groups
    elif scenario == "weekly":
        config = ScenarioPresets.weekly_stock_selection()
        config.n_groups = n_groups
    elif scenario == "monthly":
        config = ScenarioPresets.monthly_stock_selection()
        config.n_groups = n_groups
    elif scenario == "long_short":
        config = ScenarioPresets.long_short_strategy()
        config.n_groups = n_groups
    else:
        config = ScenarioPresets.weekly_stock_selection()
        config.n_groups = n_groups

    # 应用额外参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # 执行
    executor = ScenarioExecutor(config)
    return executor.run(factor_values, returns)
