"""
因子回测引擎

实现基于因子的分组回测和多空策略回测。
支持持有期调仓和换仓成本计算。
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from loguru import logger


@dataclass
class BacktestResult:
    """
    回测结果

    Attributes:
        nav_curve: 策略净值曲线
        returns: 日收益率序列
        long_short_nav: 多空净值曲线 (仅做多模式下等于 nav_curve)
        long_short_returns: 多空日收益率
        group_navs: 各组净值曲线
        group_returns: 各组日收益率
        stats: 绩效统计指标
        n_groups: 分组数量
        holding_period: 持有期
        turnover_stats: 换仓统计
    """
    nav_curve: pd.Series
    returns: pd.Series
    long_short_nav: pd.Series
    long_short_returns: pd.Series
    group_navs: Dict[int, pd.Series] = field(default_factory=dict)
    group_returns: Dict[int, pd.Series] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)
    n_groups: int = 5
    holding_period: int = 1
    turnover_stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """
    回测配置

    Attributes:
        n_groups: 分组数量
        holding_period: 持有期（调仓频率，天）
        commission_rate: 佣金费率（单边）
        stamp_duty: 印花税率（卖出时，单边）
        slippage: 滑点率（单边）
        allow_short: 是否允许做空
        top_group_only: 仅做多因子最高组（不使用多空）
    """
    n_groups: int = 5
    holding_period: int = 5
    commission_rate: float = 0.0003  # 万三
    stamp_duty: float = 0.001  # 千一
    slippage: float = 0.0005  # 万五滑点
    allow_short: bool = False  # A股限制做空
    top_group_only: bool = True  # 仅做多因子最高组


class FactorBacktest:
    """
    因子回测引擎

    支持的功能：
    - 分组回测
    - 仅做多策略（适合A股）
    - 多空策略（可选）
    - 持有期调仓
    - 换仓成本计算（佣金+印花税+滑点）
    - 净值曲线计算
    - 绩效统计

    Example:
        backtest = FactorBacktest(
            factor_values=factor_df,
            returns=returns_df,
        )

        result = backtest.run(
            n_groups=5,
            holding_period=5,
            commission_rate=0.0003,
        )

        # 获取净值曲线
        nav = result.nav_curve

        # 获取绩效统计
        print(result.stats)
    """

    def __init__(
        self,
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
        prices: Optional[pd.DataFrame] = None,
    ):
        """
        初始化回测引擎

        Args:
            factor_values: 因子值 DataFrame (index: date, columns: assets)
            returns: 收益率 DataFrame (index: date, columns: assets)
            prices: 价格 DataFrame (可选，用于计算其他指标)
        """
        self.factor_values = factor_values
        self.returns = returns
        self.prices = prices

        # 对齐数据
        self._align_data()

        logger.info(f"FactorBacktest initialized: {len(self.factor_values)} dates, {self.factor_values.shape[1]} assets")

    def _align_data(self):
        """对齐因子值和收益率数据"""
        # 统一索引
        common_dates = self.factor_values.index.intersection(self.returns.index)
        common_assets = self.factor_values.columns.intersection(self.returns.columns)

        self.factor_values = self.factor_values.loc[common_dates, common_assets]
        self.returns = self.returns.loc[common_dates, common_assets]

        # 排序
        self.factor_values = self.factor_values.sort_index()
        self.returns = self.returns.sort_index()

        # 确保数据为数值类型
        self.factor_values = self.factor_values.apply(pd.to_numeric, errors='coerce')
        self.returns = self.returns.apply(pd.to_numeric, errors='coerce')

    def run(
        self,
        n_groups: int = 5,
        holding_period: int = 5,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.001,
        slippage: float = 0.0005,
        allow_short: bool = False,
    ) -> BacktestResult:
        """
        运行回测

        Args:
            n_groups: 分组数量
            holding_period: 持有期/调仓频率（天）
            commission_rate: 佣金费率（单边，默认万三）
            stamp_duty: 印花税率（卖出时，默认千一）
            slippage: 滑点率（单边，默认万五）
            allow_short: 是否允许做空

        Returns:
            BacktestResult 对象
        """
        logger.info(f"Running backtest: n_groups={n_groups}, holding_period={holding_period}")

        # 参数校验
        n_dates = len(self.factor_values)
        if holding_period > n_dates:
            logger.warning(f"holding_period ({holding_period}) > data length ({n_dates}), adjusting to {n_dates}")
            holding_period = n_dates

        if n_groups > self.factor_values.shape[1]:
            logger.warning(f"n_groups ({n_groups}) > n_assets ({self.factor_values.shape[1]}), adjusting")
            n_groups = min(n_groups, self.factor_values.shape[1])

        # 1. 确定调仓日
        rebalance_dates = self._get_rebalance_dates(holding_period)

        # 2. 按因子值分组（仅在调仓日分组）
        group_assignments = self._assign_groups_with_rebalance(n_groups, rebalance_dates)

        # 3. 计算各组收益（含换仓成本）
        group_returns, turnover_stats = self._compute_group_returns_with_turnover(
            group_assignments=group_assignments,
            rebalance_dates=rebalance_dates,
            commission_rate=commission_rate,
            stamp_duty=stamp_duty,
            slippage=slippage,
        )

        # 4. 计算策略收益（仅做多：买入因子最高组）
        if allow_short:
            # 多空策略
            strategy_returns = compute_long_short(
                group_returns=group_returns,
                top_group=n_groups - 1,
                bottom_group=0,
            )
        else:
            # 仅做多：买入因子最高组
            strategy_returns = group_returns.get(n_groups - 1, pd.Series(0, index=self.returns.index))

        # 5. 计算净值曲线
        group_navs = {}
        for group_id, ret in group_returns.items():
            ret_clean = pd.to_numeric(ret, errors='coerce').fillna(0)
            nav = (1 + ret_clean).cumprod()
            group_navs[group_id] = nav

        strategy_returns_clean = pd.to_numeric(strategy_returns, errors='coerce').fillna(0)
        strategy_nav = (1 + strategy_returns_clean).cumprod()

        # 6. 计算绩效统计
        stats = self._compute_performance_stats(strategy_returns)

        # 7. 构建结果
        result = BacktestResult(
            nav_curve=strategy_nav,
            returns=strategy_returns,
            long_short_nav=strategy_nav,
            long_short_returns=strategy_returns,
            group_navs=group_navs,
            group_returns=group_returns,
            stats=stats,
            n_groups=n_groups,
            holding_period=holding_period,
            turnover_stats=turnover_stats,
        )

        logger.info(f"Backtest complete: Sharpe={stats.get('sharpe_ratio', 0):.2f}, "
                   f"MaxDD={stats.get('max_drawdown', 0):.2%}, "
                   f"Turnover={turnover_stats.get('avg_turnover', 0):.2%}")

        return result

    def _get_rebalance_dates(self, holding_period: int) -> List:
        """
        获取调仓日期列表

        Args:
            holding_period: 调仓频率（天）

        Returns:
            调仓日期列表
        """
        all_dates = self.factor_values.index.tolist()
        rebalance_dates = []

        # 从第一个日期开始，每 holding_period 天调仓一次
        for i in range(0, len(all_dates), holding_period):
            rebalance_dates.append(all_dates[i])

        return rebalance_dates

    def _assign_groups_with_rebalance(
        self,
        n_groups: int,
        rebalance_dates: List,
    ) -> pd.DataFrame:
        """
        按因子值分配组别（仅在调仓日重新分组）

        Args:
            n_groups: 分组数量
            rebalance_dates: 调仓日期列表

        Returns:
            组别分配 DataFrame
        """
        group_assignments = pd.DataFrame(
            index=self.factor_values.index,
            columns=self.factor_values.columns,
            dtype=object,
        )

        last_assignment = None

        for date in self.factor_values.index:
            if date in rebalance_dates:
                # 调仓日：重新分组
                factor_row = self.factor_values.loc[date]

                try:
                    # 检查有效数据数量
                    valid_factors = factor_row.dropna()
                    if len(valid_factors) < n_groups:
                        logger.warning(f"Not enough valid factors on {date}: {len(valid_factors)} < {n_groups}")
                        if last_assignment is not None:
                            group_assignments.loc[date, last_assignment.index] = last_assignment.values
                        continue

                    groups = pd.qcut(
                        valid_factors,
                        q=n_groups,
                        labels=False,
                        duplicates='drop',
                    )
                    last_assignment = groups
                    group_assignments.loc[date, groups.index] = groups.values
                except ValueError as e:
                    # 数据不足或无法分组时保持上一期的分组
                    logger.warning(f"Failed to group on {date}: {e}")
                    if last_assignment is not None:
                        group_assignments.loc[date, last_assignment.index] = last_assignment.values
            else:
                # 非调仓日：保持上一期的分组
                if last_assignment is not None:
                    group_assignments.loc[date, last_assignment.index] = last_assignment.values

        return group_assignments

    def _compute_group_returns_with_turnover(
        self,
        group_assignments: pd.DataFrame,
        rebalance_dates: List,
        commission_rate: float,
        stamp_duty: float,
        slippage: float,
    ) -> Tuple[Dict[int, pd.Series], Dict[str, float]]:
        """
        计算各组收益率（含换仓成本）

        Args:
            group_assignments: 组别分配 DataFrame
            rebalance_dates: 调仓日期列表
            commission_rate: 佣金费率
            stamp_duty: 印花税
            slippage: 滑点

        Returns:
            (各组收益率字典, 换仓统计字典)
        """
        # 安全检查：处理全 NaN 或全相同因子值的情况
        valid_values = group_assignments.dropna().values
        if len(valid_values) == 0:
            logger.warning("No valid factor values for grouping, returning empty results")
            empty_series = pd.Series([0.0] * len(self.returns), index=self.returns.index)
            return {0: empty_series}, {'avg_turnover': 0, 'max_turnover': 0, 'total_cost_rate': 0}

        n_groups = int(valid_values.max()) + 1
        group_returns = {g: [] for g in range(n_groups)}

        # 记录每组上一期的持仓
        prev_holdings = {g: set() for g in range(n_groups)}
        turnover_records = []

        dates = self.returns.index.tolist()

        for i, date in enumerate(dates):
            if date not in group_assignments.index:
                for g in range(n_groups):
                    group_returns[g].append(0.0)
                continue

            # 判断是否为调仓日
            is_rebalance_day = date in rebalance_dates

            for group_id in range(n_groups):
                # 获取当前组的股票
                group_mask = group_assignments.loc[date]
                current_holdings = set(group_mask[group_mask == group_id].dropna().index.tolist())

                # 计算当日收益
                if len(current_holdings) == 0:
                    group_returns[group_id].append(0.0)
                    continue

                # 获取这些股票的收益
                ret_values = self.returns.loc[date, list(current_holdings)]
                if isinstance(ret_values, pd.Series):
                    daily_ret = float(ret_values.mean())
                else:
                    daily_ret = 0.0

                # 调仓日计算换仓成本
                if is_rebalance_day and i > 0:
                    prev_holdings_set = prev_holdings[group_id]

                    # 计算换仓比例
                    sell_set = prev_holdings_set - current_holdings  # 卖出的股票
                    buy_set = current_holdings - prev_holdings_set   # 买入的股票
                    hold_set = prev_holdings_set & current_holdings  # 持有的股票

                    total_value = len(prev_holdings_set) if len(prev_holdings_set) > 0 else len(current_holdings)
                    if total_value > 0:
                        sell_ratio = len(sell_set) / total_value
                        buy_ratio = len(buy_set) / total_value
                    else:
                        sell_ratio = 0
                        buy_ratio = 0

                    # 换仓成本 = 卖出成本 + 买入成本
                    # 卖出：佣金 + 印花税 + 滑点
                    sell_cost = sell_ratio * (commission_rate + stamp_duty + slippage)
                    # 买入：佣金 + 滑点
                    buy_cost = buy_ratio * (commission_rate + slippage)
                    total_cost = sell_cost + buy_cost

                    # 从收益中扣除成本
                    daily_ret -= total_cost

                    # 记录换手率
                    turnover_records.append(sell_ratio + buy_ratio)

                group_returns[group_id].append(daily_ret)
                prev_holdings[group_id] = current_holdings

        # 构建 Series
        for group_id in range(n_groups):
            group_returns[group_id] = pd.Series(
                group_returns[group_id],
                index=dates[:len(group_returns[group_id])],
                name=f"Group_{group_id}",
                dtype=float,
            )

        # 计算换仓统计
        turnover_stats = {
            "avg_turnover": np.mean(turnover_records) if turnover_records else 0,
            "max_turnover": np.max(turnover_records) if turnover_records else 0,
            "total_cost_rate": (commission_rate * 2 + stamp_duty + slippage * 2),  # 单次完整换仓成本
        }

        return group_returns, turnover_stats

    def _compute_performance_stats(self, returns: pd.Series) -> Dict[str, float]:
        """计算绩效统计"""
        from web.components.backtest_charts import compute_performance_stats
        return compute_performance_stats(returns)


def compute_long_short(
    group_returns: Dict[int, pd.Series],
    top_group: int,
    bottom_group: int,
) -> pd.Series:
    """
    计算多空收益

    Args:
        group_returns: 各组收益率字典
        top_group: 做多组别（因子值最高）
        bottom_group: 做空组别（因子值最低）

    Returns:
        多空收益序列
    """
    if top_group not in group_returns or bottom_group not in group_returns:
        raise ValueError(f"Group {top_group} or {bottom_group} not found")

    long_returns = group_returns[top_group]
    short_returns = group_returns[bottom_group]

    # 确保索引对齐
    common_index = long_returns.index.intersection(short_returns.index)
    long_returns = long_returns.loc[common_index]
    short_returns = short_returns.loc[common_index]

    # 做多 top 组，做空 bottom 组
    long_short = long_returns - short_returns

    # 确保返回数值类型
    long_short = pd.to_numeric(long_short, errors='coerce').fillna(0)

    return long_short


def run_quick_backtest(
    factor_values: pd.DataFrame,
    returns: pd.DataFrame,
    n_groups: int = 5,
    holding_period: int = 5,
) -> Dict[str, Any]:
    """
    快速回测接口

    Args:
        factor_values: 因子值
        returns: 收益率
        n_groups: 分组数量
        holding_period: 持有期

    Returns:
        回测结果字典
    """
    backtest = FactorBacktest(factor_values, returns)
    result = backtest.run(n_groups=n_groups, holding_period=holding_period)

    return {
        "stats": result.stats,
        "group_navs": {k: v.to_dict() for k, v in result.group_navs.items()},
        "nav_curve": result.nav_curve.to_dict(),
        "turnover_stats": result.turnover_stats,
    }
