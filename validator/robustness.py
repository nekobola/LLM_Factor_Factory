"""
Newey-West 稳健性检验

计算 Newey-West 调整后的标准误和 t 统计量，
用于处理异方差和自相关问题。

数学原理：
Newey-West 估计量对协方差矩阵进行 HAC
调整，公式如下：

Ω̂ = Σ₀ + Σ_{j=1}^{L} w_j × (Σ_j + Σ_j')

其中：
- Σ_j 是第 j 阶滞后协方差
- w_j 是 Bartlett 核权重: w_j = 1 - j/(L+1)
- L 是滞后阶数（使用 Schwert 准则自动选择）

参考文献：
Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from loguru import logger


@dataclass
class RobustnessResult:
    """
    稳健性检验结果

    Attributes:
        ic_mean: IC 均值
        ic_std: IC 标准差
        nw_std: Newey-West 调整后标准差
        nw_tstat: Newey-West t 统计量
        nw_pvalue: Newey-West p 值
        optimal_lags: 最优滞后阶数
        is_significant: 是否显著
        has_autocorr: 是否存在自相关
        lb_test: Ljung-Box 检验结果
    """
    ic_mean: float
    ic_std: float
    nw_std: float
    nw_tstat: float
    nw_pvalue: float
    optimal_lags: int
    is_significant: bool
    has_autocorr: bool
    lb_test: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "nw_std": self.nw_std,
            "nw_tstat": self.nw_tstat,
            "nw_pvalue": self.nw_pvalue,
            "optimal_lags": self.optimal_lags,
            "is_significant": self.is_significant,
            "has_autocorr": self.has_autocorr,
        }


class RobustnessTester:
    """
    Newey-West 稳健性检验器

    Example:
        tester = RobustnessTester(significance_level=0.05)

        result = tester.run(ic_series)

        if result.is_significant:
            print(f"因子 IC 显著，p={result.nw_pvalue:.4f}")
        else:
            print("因子 IC 不显著")
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        max_lags: int = 10,
        lag_selection: Literal["schwert", "aic", "bic", "fixed"] = "schwert",
    ):
        """
        初始化稳健性检验器

        Args:
            significance_level: 显著性水平
            max_lags: 最大滞后阶数
            lag_selection: 滞后阶数选择方法
        """
        self.significance_level = significance_level
        self.max_lags = max_lags
        self.lag_selection = lag_selection

        logger.info(f"RobustnessTester initialized, significance_level={significance_level}")

    def select_lags_schwert(self, n: int) -> int:
        """
        Schwert 准则选择滞后阶数

        L = int(4 × (n/100)^(2/9))

        这是处理金融时间序列常用的经验法则。

        Args:
            n: 样本量

        Returns:
            滞后阶数
        """
        lags = int(4 * (n / 100) ** (2 / 9))
        return min(lags, self.max_lags)

    def select_lags_aic(
        self,
        series: pd.Series,
        max_lags: Optional[int] = None,
    ) -> int:
        """
        AIC 准则选择滞后阶数

        Args:
            series: 时间序列
            max_lags: 最大滞后阶数

        Returns:
            最优滞后阶数
        """
        max_lags = max_lags or self.max_lags
        best_aic = float('inf')
        best_lag = 1

        for lag in range(1, max_lags + 1):
            try:
                model = OLS(series.values[lag:], self._create_lag_matrix(series, lag))
                results = model.fit()
                aic = results.aic

                if aic < best_aic:
                    best_aic = aic
                    best_lag = lag
            except Exception:
                continue

        return best_lag

    def _create_lag_matrix(self, series: pd.Series, lags: int) -> np.ndarray:
        """创建滞后矩阵（含截距）"""
        n = len(series)
        X = np.ones((n - lags, lags + 1))
        for i in range(lags):
            X[:, i + 1] = series.values[lags - i - 1 : n - i - 1]
        return X

    def select_lags(
        self,
        series: pd.Series,
        method: Optional[str] = None,
    ) -> int:
        """
        选择滞后阶数

        Args:
            series: 时间序列
            method: 选择方法

        Returns:
            滞后阶数
        """
        method = method or self.lag_selection
        n = len(series)

        if method == "schwert":
            return self.select_lags_schwert(n)
        elif method == "aic":
            return self.select_lags_aic(series)
        elif method == "bic":
            # BIC 倾向于选择更小的模型
            return self.select_lags_aic(series) // 2 + 1
        elif method == "fixed":
            return self.max_lags
        else:
            return self.select_lags_schwert(n)

    def compute_newey_west(
        self,
        series: pd.Series,
        lags: Optional[int] = None,
    ) -> tuple[float, float, float]:
        """
        计算 Newey-West 调整后的统计量

        Args:
            series: IC 时间序列
            lags: 滞后阶数

        Returns:
            (NW 标准误, NW t 统计量, NW p 值)
        """
        series_clean = series.dropna()
        n = len(series_clean)

        if n < 10:
            return np.nan, np.nan, np.nan

        # 选择滞后阶数
        if lags is None:
            lags = self.select_lags(series_clean)

        # 计算均值和普通标准误
        mean = series_clean.mean()
        std = series_clean.std()

        if std == 0 or np.isnan(std):
            return std, 0.0, 1.0

        # 计算 Newey-West 协方差
        # 使用 Bartlett 核
        cov_matrix = self._compute_hac_covariance(series_clean, lags)

        # Newey-West 标准误
        nw_var = cov_matrix / n
        nw_std = np.sqrt(nw_var) if nw_var > 0 else std

        # Newey-West t 统计量
        nw_tstat = mean / nw_std if nw_std > 0 else 0.0

        # 双边 t 检验 p 值
        nw_pvalue = 2 * (1 - stats.t.cdf(abs(nw_tstat), df=n - 1))

        return nw_std, nw_tstat, nw_pvalue

    def _compute_hac_covariance(
        self,
        series: pd.Series,
        lags: int,
    ) -> float:
        """
        计算 HAC (Heteroskedasticity and Autocorrelation Consistent) 协方差

        使用 Bartlett 核进行加权。

        Args:
            series: 时间序列
            lags: 滞后阶数

        Returns:
            HAC 协方差估计
        """
        n = len(series)
        mean = series.mean()
        residuals = series.values - mean

        # 零阶协方差（即方差）
        gamma_0 = np.sum(residuals ** 2) / n

        # 高阶协方差
        gamma_sum = 0.0
        for j in range(1, lags + 1):
            # Bartlett 核权重
            weight = 1 - j / (lags + 1)

            # 计算第 j 阶自协方差
            gamma_j = np.sum(residuals[j:] * residuals[:-j]) / n

            gamma_sum += weight * gamma_j

        # HAC 协方差
        hac_cov = gamma_0 + 2 * gamma_sum

        return hac_cov

    def ljung_box_test(
        self,
        series: pd.Series,
        lags: int = 10,
    ) -> Dict[str, Any]:
        """
        Ljung-Box 自相关检验

        检验序列是否存在自相关。

        Args:
            series: 时间序列
            lags: 检验滞后阶数

        Returns:
            检验结果字典
        """
        series_clean = series.dropna()

        if len(series_clean) < lags + 1:
            return {"statistic": np.nan, "pvalue": np.nan, "has_autocorr": False}

        result = acorr_ljungbox(series_clean, lags=lags, return_df=True)

        # 取最大滞后阶数的结果
        lb_stat = result.iloc[-1]["lb_stat"]
        lb_pvalue = result.iloc[-1]["lb_pvalue"]

        return {
            "statistic": lb_stat,
            "pvalue": lb_pvalue,
            "has_autocorr": lb_pvalue < 0.05,
        }

    def run(
        self,
        ic_series: pd.Series,
        lags: Optional[int] = None,
    ) -> RobustnessResult:
        """
        运行完整稳健性检验

        Args:
            ic_series: IC 时间序列
            lags: 滞后阶数（自动选择则为 None）

        Returns:
            RobustnessResult 对象
        """
        logger.info("Running robustness test...")

        ic_clean = ic_series.dropna()
        n = len(ic_clean)

        # 选择滞后阶数
        if lags is None:
            lags = self.select_lags(ic_clean)

        # 计算基本统计量
        ic_mean = ic_clean.mean()
        ic_std = ic_clean.std()

        # 计算 Newey-West 统计量
        nw_std, nw_tstat, nw_pvalue = self.compute_newey_west(ic_clean, lags)

        # Ljung-Box 检验
        lb_result = self.ljung_box_test(ic_clean, lags=min(lags, 10))

        # 判断显著性
        is_significant = nw_pvalue < self.significance_level

        result = RobustnessResult(
            ic_mean=ic_mean,
            ic_std=ic_std,
            nw_std=nw_std,
            nw_tstat=nw_tstat,
            nw_pvalue=nw_pvalue,
            optimal_lags=lags,
            is_significant=is_significant,
            has_autocorr=lb_result["has_autocorr"],
            lb_test=lb_result,
            details={
                "n_obs": n,
                "lag_selection": self.lag_selection,
            }
        )

        logger.info(f"Robustness test done: NW_t={nw_tstat:.2f}, NW_p={nw_pvalue:.4f}")
        return result


def newey_west_ttest(
    series: pd.Series,
    lags: Optional[int] = None,
) -> tuple[float, float]:
    """
    快速 Newey-West t 检验

    Args:
        series: 时间序列
        lags: 滞后阶数

    Returns:
        (t 统计量, p 值)
    """
    tester = RobustnessTester()
    _, tstat, pvalue = tester.compute_newey_west(series, lags)
    return tstat, pvalue
