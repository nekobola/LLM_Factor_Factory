"""
分组回测/分层测试

将因子值分组，分析各组收益的分布和单调性。

核心指标：
- 分组收益：每组的多空收益
- 多空收益：Top 组 - Bottom 组
- 单调性得分：收益随组别递增的程度
- spread：组间收益差异
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


@dataclass
class GroupTestResult:
    """
    分组测试结果

    Attributes:
        group_returns: 各组平均收益
        group_ic: 各组 IC（可选）
        long_short_return: 多空收益 (Top - Bottom)
        long_short_tstat: 多空收益 t 统计量
        long_short_pvalue: 多空收益 p 值
        monotonicity_score: 单调性得分 (0-1)
        spread: 组间收益差异
        n_groups: 分组数量
        group_stats: 各组详细统计
    """
    group_returns: pd.Series
    long_short_return: float
    long_short_tstat: float
    long_short_pvalue: float
    monotonicity_score: float
    spread: float
    n_groups: int
    group_ic: Optional[pd.Series] = None
    group_stats: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_returns": self.group_returns.to_dict(),
            "long_short_return": self.long_short_return,
            "long_short_tstat": self.long_short_tstat,
            "long_short_pvalue": self.long_short_pvalue,
            "monotonicity_score": self.monotonicity_score,
            "spread": self.spread,
            "n_groups": self.n_groups,
        }


class GroupTester:
    """
    分组回测测试器

    支持两种分组方式：
    - quantile: 分位数分组（默认）
    - equal: 等数量分组

    Example:
        tester = GroupTester(n_groups=5)

        result = tester.run(factor_values, forward_returns)

        print(f"Top 组收益: {result.group_returns.iloc[-1]:.4f}")
        print(f"多空收益: {result.long_short_return:.4f}")
        print(f"单调性: {result.monotonicity_score:.2f}")
    """

    def __init__(
        self,
        n_groups: int = 5,
        method: str = "quantile",
        min_assets_per_group: int = 10,
    ):
        """
        初始化分组测试器

        Args:
            n_groups: 分组数量
            method: 分组方法 ("quantile" 或 "equal")
            min_assets_per_group: 每组最小资产数
        """
        self.n_groups = n_groups
        self.method = method
        self.min_assets_per_group = min_assets_per_group

        logger.info(f"GroupTester initialized, n_groups={n_groups}, method={method}")

    def assign_groups(
        self,
        factor_values: pd.Series,
    ) -> pd.Series:
        """
        将因子值分配到各组

        Args:
            factor_values: 单期因子值

        Returns:
            组别标签 Series
        """
        factor_clean = factor_values.dropna()

        if len(factor_clean) < self.n_groups * self.min_assets_per_group:
            return pd.Series(np.nan, index=factor_values.index)

        if self.method == "quantile":
            # 分位数分组
            labels = pd.qcut(
                factor_clean,
                q=self.n_groups,
                labels=False,
                duplicates="drop"
            )
        elif self.method == "equal":
            # 等数量分组
            rank = factor_clean.rank(method="first")
            labels = (rank / len(rank) * self.n_groups).astype(int).clip(0, self.n_groups - 1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 映射回原索引
        result = pd.Series(np.nan, index=factor_values.index)
        result.loc[factor_clean.index] = labels.values

        return result

    def compute_group_returns(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        计算各组收益率

        Args:
            factor_values: 因子值，index: date, columns: assets
            forward_returns: 未来收益率，index: date, columns: assets

        Returns:
            (各组收益 DataFrame, 组别分配 DataFrame)
        """
        # 确保索引对齐
        common_dates = factor_values.index.intersection(forward_returns.index)
        factor_values = factor_values.loc[common_dates]
        forward_returns = forward_returns.loc[common_dates]

        group_returns_list = []
        group_assignments = pd.DataFrame(index=common_dates, columns=factor_values.columns)

        for date in common_dates:
            factor = factor_values.loc[date]
            ret = forward_returns.loc[date]

            # 分配组别
            groups = self.assign_groups(factor)
            group_assignments.loc[date] = groups

            # 计算各组平均收益
            group_rets = {}
            for g in range(self.n_groups):
                assets_in_group = groups[groups == g].index
                if len(assets_in_group) > 0:
                    group_rets[g] = ret.loc[assets_in_group].mean()
                else:
                    group_rets[g] = np.nan

            group_returns_list.append(group_rets)

        group_returns = pd.DataFrame(group_returns_list, index=common_dates)
        return group_returns, group_assignments

    def compute_monotonicity_score(
        self,
        group_returns: pd.DataFrame,
    ) -> float:
        """
        计算单调性得分

        使用 Spearman 秩相关系数衡量收益随组别递增的程度。

        Args:
            group_returns: 各组收益时间序列

        Returns:
            单调性得分 (0-1)
        """
        # 计算平均组收益
        mean_returns = group_returns.mean()

        # 如果有缺失组，跳过
        mean_returns = mean_returns.dropna()

        if len(mean_returns) < 2:
            return 0.0

        # 计算与组别的相关性
        x = np.arange(len(mean_returns))
        y = mean_returns.values

        # Spearman 相关性
        corr, _ = stats.spearmanr(x, y)

        # 转换为 0-1 分数
        score = (corr + 1) / 2

        return score

    def compute_long_short(
        self,
        group_returns: pd.DataFrame,
    ) -> Tuple[float, float, float]:
        """
        计算多空收益

        Args:
            group_returns: 各组收益时间序列

        Returns:
            (多空收益均值, t 统计量, p 值)
        """
        # Top 组 - Bottom 组
        top_group = group_returns.iloc[:, -1]  # 最后一组
        bottom_group = group_returns.iloc[:, 0]  # 第一组

        long_short = top_group - bottom_group

        # t 检验
        mean_ls = long_short.mean()
        valid_ls = long_short.dropna()

        if len(valid_ls) > 1:
            tstat, pvalue = stats.ttest_1samp(valid_ls, 0)
        else:
            tstat, pvalue = 0.0, 1.0

        return mean_ls, tstat, pvalue

    def run(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> GroupTestResult:
        """
        运行分组测试

        Args:
            factor_values: 因子值
            forward_returns: 未来收益率

        Returns:
            GroupTestResult 对象
        """
        logger.info("Running group test...")

        # 计算各组收益
        group_returns, group_assignments = self.compute_group_returns(
            factor_values, forward_returns
        )

        # 计算多空收益
        ls_return, ls_tstat, ls_pvalue = self.compute_long_short(group_returns)

        # 计算单调性
        mono_score = self.compute_monotonicity_score(group_returns)

        # 计算组间差异 (spread)
        mean_returns = group_returns.mean()
        if len(mean_returns.dropna()) >= 2:
            spread = mean_returns.iloc[-1] - mean_returns.iloc[0]
        else:
            spread = 0.0

        # 各组统计
        group_stats = {
            f"group_{i}": {
                "mean_return": group_returns[i].mean(),
                "std_return": group_returns[i].std(),
                "count": (~group_returns[i].isna()).sum(),
            }
            for i in range(self.n_groups)
            if i in group_returns.columns
        }

        result = GroupTestResult(
            group_returns=mean_returns,
            long_short_return=ls_return,
            long_short_tstat=ls_tstat,
            long_short_pvalue=ls_pvalue,
            monotonicity_score=mono_score,
            spread=spread,
            n_groups=self.n_groups,
            group_stats=group_stats,
            details={
                "method": self.method,
                "min_assets_per_group": self.min_assets_per_group,
            }
        )

        logger.info(f"Group test done: LS={ls_return:.4f}, Monotonicity={mono_score:.2f}")
        return result


def create_alphalens_tearsheet(
    factor_values: pd.DataFrame,
    returns: pd.DataFrame,
    quantiles: int = 5,
):
    """
    创建 Alphalens 格式的 tearsheet

    需要安装 alphalens 库。

    Args:
        factor_values: 因子值
        returns: 收益率
        quantiles: 分位数

    Returns:
        Alphalens tearsheet 对象
    """
    try:
        import alphalens
        from alphalens.utils import get_clean_factor_and_forward_returns
    except ImportError:
        raise ImportError("Please install alphalens: pip install alphalens")

    # 转换为 Alphalens 格式
    factor_long = factor_values.stack()
    prices = (1 + returns).cumprod()

    # 清理数据
    factor_data = get_clean_factor_and_forward_returns(
        factor_long,
        prices,
        quantiles=quantiles,
        periods=(1, 5, 10, 20),
    )

    # 创建 tearsheet
    alphalens.tears.create_full_tear_sheet(factor_data)

    return factor_data
