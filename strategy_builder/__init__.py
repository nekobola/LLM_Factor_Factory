"""
策略构建模块
"""

from .backtest import FactorBacktest, BacktestResult, BacktestConfig, compute_long_short, run_quick_backtest
from .scenario_config import (
    TradingScenario,
    RebalanceFrequency,
    StrategyType,
    TradingCostConfig,
    UniverseConfig,
    BacktestScenarioConfig,
    ScenarioPresets,
    ScenarioExecutor,
    quick_backtest,
)

__all__ = [
    # Backtest
    "FactorBacktest",
    "BacktestResult",
    "BacktestConfig",
    "compute_long_short",
    "run_quick_backtest",
    # Scenario Config
    "TradingScenario",
    "RebalanceFrequency",
    "StrategyType",
    "TradingCostConfig",
    "UniverseConfig",
    "BacktestScenarioConfig",
    "ScenarioPresets",
    "ScenarioExecutor",
    "quick_backtest",
]
