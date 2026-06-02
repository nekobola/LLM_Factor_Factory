"""
IC/IR 分析器

计算因子的 Information Coefficient (IC) 和 Information Ratio (IR)。

IC 定义：因子值与未来收益率的相关系数
- IC_mean: IC 的时间序列均值
- IC_std: IC 的时间序列标准差
- ICIR (IR): IC_mean / IC_std，衡量因子预测能力的稳定性

行业标准：
- |IC_mean| > 0.02: 因子有一定预测能力
- ICIR > 0.5: 因子预测能力稳定
- ICIR > 1.0: 优秀因子
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


@dataclass
class ICResult:
    """
    IC 分析结果

    Attributes:
        ic_series: IC 时间序列
        ic_mean: IC 均值
        ic_std: IC 标准差
        ic_ir: IC 信息比率 (IC_mean / IC_std)
        ic_tstat: IC t 统计量
        ic_pvalue: IC p 值
        positive_ratio: IC > 0 的比例
        ic_decay: IC 衰减曲线（不同持有期）
    """
    ic_series: pd.Series
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_tstat: float
    ic_pvalue: float
    positive_ratio: float
    ic_decay: Optional[pd.Series] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ic_ir": self.ic_ir,
            "ic_tstat": self.ic_tstat,
            "ic_pvalue": self.ic_pvalue,
            "positive_ratio": self.positive_ratio,
            "ic_decay": self.ic_decay.to_dict() if self.ic_decay is not None else None,
        }


class ICAnalyzer:
    """
    IC/IR 分析器

    支持多种相关系数计算方法：
    - pearson: Pearson 相关系数（默认）
    - spearman: Spearman 秩相关系数（更稳健）
    - kendall: Kendall 秩相关系数

    Example:
        analyzer = ICAnalyzer(method="spearman")

        # 计算单期 IC
        ic_series = analyzer.compute_ic(factor_values, forward_returns)

        # 计算 IC 衰减
        ic_decay = analyzer.compute_ic_decay(factor_values, returns, periods=[1, 5, 10, 20])

        # 获取汇总统计
        result = analyzer.run(factor_values, forward_returns)
    """

    def __init__(
        self,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
        min_periods: int = 30,
        decay_periods: Optional[List[int]] = None,
    ):
        """
        初始化 IC 分析器

        Args:
            method: 相关系数计算方法
            min_periods: 计算相关系数的最小样本数
            decay_periods: IC 衰减分析的持有期列表
        """
        self.method = method
        self.min_periods = min_periods
        self.decay_periods = decay_periods or [1, 5, 10, 20]

        logger.info(f"ICAnalyzer initialized, method={method}")

    def compute_ic(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> pd.Series:
        """
        计算 IC 时间序列

        Args:
            factor_values: 因子值，index: date, columns: assets
            forward_returns: 未来收益率，index: date, columns: assets

        Returns:
            IC 时间序列，index: date
        """
        # 确保索引对齐
        common_dates = factor_values.index.intersection(forward_returns.index)
        factor_values = factor_values.loc[common_dates]
        forward_returns = forward_returns.loc[common_dates]

        ic_list = []

        for date in common_dates:
            factor = factor_values.loc[date].dropna()
            ret = forward_returns.loc[date].dropna()

            # 取交集
            common_assets = factor.index.intersection(ret.index)
            if len(common_assets) < self.min_periods:
                ic_list.append(np.nan)
                continue

            factor_aligned = factor.loc[common_assets]
            ret_aligned = ret.loc[common_assets]

            # 计算相关系数
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ic = self._compute_correlation(factor_aligned, ret_aligned)

            ic_list.append(ic)

        ic_series = pd.Series(ic_list, index=common_dates, name="IC")
        return ic_series

    def _compute_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """计算相关系数"""
        if self.method == "pearson":
            corr, _ = stats.pearsonr(x, y)
        elif self.method == "spearman":
            corr, _ = stats.spearmanr(x, y, nan_policy="omit")
        elif self.method == "kendall":
            corr, _ = stats.kendalltau(x, y, nan_policy="omit")
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return corr

    def compute_ic_decay(
        self,
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
        periods: Optional[List[int]] = None,
    ) -> pd.Series:
        """
        计算 IC 衰减曲线

        分析因子对不同持有期收益的预测能力衰减情况。

        Args:
            factor_values: 因子值
            returns: 收益率
            periods: 持有期列表（天数）

        Returns:
            IC 衰减曲线，index: period, value: IC_mean
        """
        periods = periods or self.decay_periods
        ic_values = []

        for period in periods:
            # 计算未来 period 天收益率
            forward_returns = returns.shift(-period) / returns.shift(-period + 1) - 1
            # 对于多日持有期，累计收益
            if period > 1:
                forward_returns = returns.pct_change(periods=period).shift(-period)

            ic_series = self.compute_ic(factor_values, forward_returns)
            ic_values.append(ic_series.mean())

        ic_decay = pd.Series(ic_values, index=periods, name="IC_decay")
        return ic_decay

    def run(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        compute_decay: bool = True,
    ) -> ICResult:
        """
        运行完整 IC 分析

        Args:
            factor_values: 因子值
            forward_returns: 未来收益率
            compute_decay: 是否计算 IC 衰减

        Returns:
            ICResult 对象
        """
        logger.info("Running IC analysis...")

        # 计算 IC 时间序列
        ic_series = self.compute_ic(factor_values, forward_returns)

        # 汇总统计
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()

        # 处理 ic_std 为 0 的情况
        if ic_std == 0 or np.isnan(ic_std):
            ic_ir = 0.0
        else:
            ic_ir = ic_mean / ic_std

        # t 检验
        valid_ic = ic_series.dropna()
        if len(valid_ic) > 1:
            ic_tstat, ic_pvalue = stats.ttest_1samp(valid_ic, 0)
        else:
            ic_tstat, ic_pvalue = 0.0, 1.0

        # IC > 0 比例
        positive_ratio = (valid_ic > 0).mean()

        # IC 衰减
        ic_decay = None
        if compute_decay:
            try:
                # 从 forward_returns 反推 returns
                # 这里简化处理，假设 forward_returns 是 1 日收益
                returns = forward_returns
                ic_decay = self.compute_ic_decay(factor_values, returns)
            except Exception as e:
                logger.warning(f"IC decay computation failed: {e}")

        result = ICResult(
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_tstat=ic_tstat,
            ic_pvalue=ic_pvalue,
            positive_ratio=positive_ratio,
            ic_decay=ic_decay,
            details={
                "method": self.method,
                "n_periods": len(valid_ic),
                "min_periods": self.min_periods,
            }
        )

        logger.info(f"IC analysis done: IC_mean={ic_mean:.4f}, ICIR={ic_ir:.4f}")
        return result


def compute_rank_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    快速计算 Rank IC (Spearman IC)

    Args:
        factor_values: 因子值
        forward_returns: 未来收益率

    Returns:
        IC 时间序列
    """
    analyzer = ICAnalyzer(method="spearman")
    return analyzer.compute_ic(factor_values, forward_returns)


def compute_normal_ic(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    快速计算 Normal IC (Pearson IC)

    Args:
        factor_values: 因子值
        forward_returns: 未来收益率

    Returns:
        IC 时间序列
    """
    analyzer = ICAnalyzer(method="pearson")
    return analyzer.compute_ic(factor_values, forward_returns)
