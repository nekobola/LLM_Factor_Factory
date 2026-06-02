"""
正交化检验

对现有因子正交化，提取独立 Alpha。

方法：
1. 回归残差法：对现有因子回归，取残差作为正交化后的因子
2. Gram-Schmidt 正交化
3. 条件 IC：控制其他因子后的条件 IC

检验指标：
- 正交化 IC 保留率：正交化后 IC / 原始 IC
- 独立性检验：与其他因子的相关性
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
from loguru import logger


@dataclass
class OrthogonalityResult:
    """
    正交化检验结果

    Attributes:
        original_ic: 原始 IC
        orthogonal_ic: 正交化后 IC
        ic_retention: IC 保留率
        correlations: 与现有因子的相关性
        max_correlation: 最大相关系数（绝对值）
        vif: 方差膨胀因子
        is_orthogonal: 是否满足正交性要求
    """
    original_ic: float
    orthogonal_ic: float
    ic_retention: float
    correlations: pd.Series
    max_correlation: float
    vif: Optional[float] = None
    is_orthogonal: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_ic": self.original_ic,
            "orthogonal_ic": self.orthogonal_ic,
            "ic_retention": self.ic_retention,
            "max_correlation": self.max_correlation,
            "vif": self.vif,
            "is_orthogonal": self.is_orthogonal,
        }


class OrthogonalityTester:
    """
    正交化检验器

    Example:
        tester = OrthogonalityTester()

        # 设置现有因子
        tester.set_existing_factors({
            "momentum_20d": momentum_factor,
            "value_pe": pe_factor,
        })

        # 检验新因子
        result = tester.run(new_factor_values, forward_returns)

        if result.ic_retention > 0.5:
            print("因子具有独立 Alpha")
    """

    def __init__(
        self,
        method: Literal["regression", "gram_schmidt", "pca"] = "regression",
        correlation_threshold: float = 0.5,
        ic_retention_threshold: float = 0.5,
        use_ridge: bool = False,
        ridge_alpha: float = 1.0,
    ):
        """
        初始化正交化检验器

        Args:
            method: 正交化方法
            correlation_threshold: 相关性阈值（超过则认为不正交）
            ic_retention_threshold: IC 保留率阈值
            use_ridge: 是否使用 Ridge 回归（处理多重共线性）
            ridge_alpha: Ridge 正则化系数
        """
        self.method = method
        self.correlation_threshold = correlation_threshold
        self.ic_retention_threshold = ic_retention_threshold
        self.use_ridge = use_ridge
        self.ridge_alpha = ridge_alpha

        self._existing_factors: Dict[str, pd.DataFrame] = {}

        logger.info(f"OrthogonalityTester initialized, method={method}")

    def set_existing_factors(self, factors: Dict[str, pd.DataFrame]):
        """
        设置现有因子库

        Args:
            factors: 因子字典 {name: factor_values}
        """
        self._existing_factors = factors
        logger.info(f"Loaded {len(factors)} existing factors")

    def add_existing_factor(self, name: str, factor_values: pd.DataFrame):
        """添加现有因子"""
        self._existing_factors[name] = factor_values

    def compute_correlations(
        self,
        factor_values: pd.DataFrame,
    ) -> pd.Series:
        """
        计算与现有因子的相关性

        Args:
            factor_values: 新因子值

        Returns:
            相关系数 Series
        """
        if not self._existing_factors:
            return pd.Series(dtype=float)

        correlations = {}

        # 将 factor_values 转换为长格式
        factor_long = factor_values.stack()

        for name, existing_factor in self._existing_factors.items():
            existing_long = existing_factor.stack()

            # 对齐索引
            common_idx = factor_long.index.intersection(existing_long.index)
            if len(common_idx) < 30:
                correlations[name] = np.nan
                continue

            f1 = factor_long.loc[common_idx]
            f2 = existing_long.loc[common_idx]

            # 计算相关性
            corr, _ = stats.spearmanr(f1, f2)
            correlations[name] = corr

        return pd.Series(correlations)

    def orthogonalize_regression(
        self,
        factor_values: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        回归残差法正交化

        对现有因子回归，取残差作为正交化后的因子。

        Args:
            factor_values: 原始因子值

        Returns:
            正交化后的因子值
        """
        if not self._existing_factors:
            logger.warning("No existing factors to orthogonalize against")
            return factor_values

        # 准备回归数据
        factor_long = factor_values.stack().to_frame("target")
        existing_df = pd.DataFrame({
            name: f.stack() for name, f in self._existing_factors.items()
        })

        # 合并数据
        data = factor_long.join(existing_df, how="inner")

        if len(data) < 30:
            logger.warning("Insufficient data for orthogonalization")
            return factor_values

        # 特征和目标
        X = data.drop(columns=["target"])
        y = data["target"]

        # 处理缺失值
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # 回归
        if self.use_ridge:
            model = Ridge(alpha=self.ridge_alpha)
        else:
            model = LinearRegression()

        model.fit(X, y)

        # 残差
        residual = y - model.predict(X)

        # 转换回 DataFrame 格式
        residual_df = residual.unstack()
        return residual_df

    def orthogonalize_gram_schmidt(
        self,
        factor_values: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Gram-Schmidt 正交化

        Args:
            factor_values: 原始因子值

        Returns:
            正交化后的因子值
        """
        if not self._existing_factors:
            return factor_values

        # 将所有因子合并
        factor_long = factor_values.stack().to_frame("new")
        existing_df = pd.DataFrame({
            name: f.stack() for name, f in self._existing_factors.items()
        })

        data = factor_long.join(existing_df, how="inner")

        if len(data) < 30:
            return factor_values

        # Gram-Schmidt 过程
        # 将新因子对每个现有因子依次正交化
        orthogonalized = data["new"].copy()

        for col in existing_df.columns:
            existing = data[col].fillna(data[col].mean())

            # 投影到现有因子方向
            projection = (orthogonalized * existing).sum() / (existing ** 2).sum() * existing

            # 减去投影
            orthogonalized = orthogonalized - projection

        # 转换回 DataFrame
        result_df = orthogonalized.unstack()
        return result_df

    def compute_vif(
        self,
        factor_values: pd.DataFrame,
    ) -> Optional[float]:
        """
        计算方差膨胀因子 (VIF)

        VIF > 5 表示存在较强共线性，VIF > 10 表示严重共线性。

        Args:
            factor_values: 新因子值

        Returns:
            VIF 值
        """
        if not self._existing_factors:
            return None

        # 合并所有因子
        factor_long = factor_values.stack().to_frame("new")
        existing_df = pd.DataFrame({
            name: f.stack() for name, f in self._existing_factors.items()
        })

        data = factor_long.join(existing_df, how="inner")

        if len(data) < 30:
            return None

        # 计算 VIF
        # VIF = 1 / (1 - R²)
        X = data.drop(columns=["new"])

        if X.shape[1] == 0:
            return None

        X = X.fillna(X.mean())
        y = data["new"].fillna(data["new"].mean())

        model = LinearRegression()
        model.fit(X, y)

        r_squared = model.score(X, y)

        if r_squared >= 1.0:
            return float('inf')

        vif = 1 / (1 - r_squared)
        return vif

    def run(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> OrthogonalityResult:
        """
        运行正交化检验

        Args:
            factor_values: 因子值
            forward_returns: 未来收益率

        Returns:
            OrthogonalityResult 对象
        """
        logger.info("Running orthogonality test...")

        from .ic_analyzer import ICAnalyzer

        # 计算原始 IC
        ic_analyzer = ICAnalyzer(method="spearman")
        original_ic_result = ic_analyzer.run(factor_values, forward_returns)
        original_ic = original_ic_result.ic_mean

        # 正交化
        if self.method == "regression":
            orthogonal_factor = self.orthogonalize_regression(factor_values)
        elif self.method == "gram_schmidt":
            orthogonal_factor = self.orthogonalize_gram_schmidt(factor_values)
        else:
            orthogonal_factor = factor_values

        # 计算正交化后 IC
        ortho_ic_result = ic_analyzer.run(orthogonal_factor, forward_returns)
        orthogonal_ic = ortho_ic_result.ic_mean

        # IC 保留率
        if abs(original_ic) > 0.001:
            ic_retention = orthogonal_ic / original_ic
        else:
            ic_retention = 1.0

        # 相关性
        correlations = self.compute_correlations(factor_values)
        max_corr = correlations.abs().max() if len(correlations) > 0 else 0.0

        # VIF
        vif = self.compute_vif(factor_values)

        # 判断是否满足正交性
        is_orthogonal = (
            max_corr <= self.correlation_threshold and
            ic_retention >= self.ic_retention_threshold
        )

        result = OrthogonalityResult(
            original_ic=original_ic,
            orthogonal_ic=orthogonal_ic,
            ic_retention=ic_retention,
            correlations=correlations,
            max_correlation=max_corr,
            vif=vif,
            is_orthogonal=is_orthogonal,
            details={
                "method": self.method,
                "n_existing_factors": len(self._existing_factors),
            }
        )

        logger.info(f"Orthogonality test done: IC_retention={ic_retention:.2f}, Max_corr={max_corr:.2f}")
        return result
