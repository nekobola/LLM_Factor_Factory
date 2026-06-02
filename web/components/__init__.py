"""
Web 组件模块
"""

from .charts import (
    plot_ic_timeseries,
    plot_ic_distribution,
    plot_ic_decay,
    plot_ic_cumulative,
    plot_group_returns,
    plot_long_short_nav,
    plot_validation_summary,
)
from .backtest_charts import (
    plot_nav_curve,
    plot_drawdown,
    plot_monthly_returns,
    plot_performance_table,
)

__all__ = [
    "plot_ic_timeseries",
    "plot_ic_distribution",
    "plot_ic_decay",
    "plot_ic_cumulative",
    "plot_group_returns",
    "plot_long_short_nav",
    "plot_validation_summary",
    "plot_nav_curve",
    "plot_drawdown",
    "plot_monthly_returns",
    "plot_performance_table",
]
