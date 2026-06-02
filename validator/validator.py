"""
因子验证器

整合所有验证模块，提供统一的验证接口。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import asyncio

import pandas as pd
import numpy as np
from loguru import logger

from core.base_factor import ValidationResult as BaseValidationResult
from .ic_analyzer import ICAnalyzer, ICResult
from .group_test import GroupTester, GroupTestResult
from .orthogonality import OrthogonalityTester, OrthogonalityResult
from .robustness import RobustnessTester, RobustnessResult


@dataclass
class ValidationConfig:
    """
    验证配置

    Attributes:
        ic_mean_min: 最小 IC 均值阈值
        ic_ir_min: 最小 IC 信息比率阈值
        monotonicity_min: 最小单调性得分阈值
        ortho_retention_min: 最小正交化 IC 保留率阈值
        nw_pvalue_max: 最大 Newey-West p 值阈值
        n_groups: 分组数量
        ic_method: IC 计算方法
    """
    ic_mean_min: float = 0.02
    ic_ir_min: float = 0.5
    monotonicity_min: float = 0.8
    ortho_retention_min: float = 0.5
    nw_pvalue_max: float = 0.05
    n_groups: int = 5
    ic_method: str = "spearman"
    compute_decay: bool = True


@dataclass
class ValidationResult(BaseValidationResult):
    """
    完整验证结果

    继承 BaseValidationResult，包含所有验证模块的详细结果。
    """
    ic_result: Optional[ICResult] = None
    group_result: Optional[GroupTestResult] = None
    ortho_result: Optional[OrthogonalityResult] = None
    robust_result: Optional[RobustnessResult] = None


class FactorValidator:
    """
    因子验证器

    整合 IC 分析、分组测试、正交化检验和稳健性检验，
    提供统一的验证接口。

    Example:
        validator = FactorValidator()

        # 设置现有因子（用于正交化检验）
        validator.set_existing_factors(factor_dict)

        # 运行验证
        result = await validator.validate(
            factor_values=factor_values,
            forward_returns=forward_returns,
        )

        if result.passed:
            print("因子验证通过")
        else:
            print(f"验证失败: {result.errors}")
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
    ):
        """
        初始化验证器

        Args:
            config: 验证配置
        """
        self.config = config or ValidationConfig()

        # 初始化各子模块
        self.ic_analyzer = ICAnalyzer(
            method=self.config.ic_method,
            decay_periods=[1, 5, 10, 20],
        )
        self.group_tester = GroupTester(n_groups=self.config.n_groups)
        self.ortho_tester = OrthogonalityTester()
        self.robust_tester = RobustnessTester(
            significance_level=self.config.nw_pvalue_max,
        )

        logger.info(f"FactorValidator initialized with config: {self.config}")

    def set_existing_factors(self, factors: Dict[str, pd.DataFrame]):
        """
        设置现有因子库（用于正交化检验）

        Args:
            factors: 因子字典 {name: factor_values}
        """
        self.ortho_tester.set_existing_factors(factors)

    def add_existing_factor(self, name: str, factor_values: pd.DataFrame):
        """添加现有因子"""
        self.ortho_tester.add_existing_factor(name, factor_values)

    async def validate(
        self,
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
        factor_metadata: Optional[Dict[str, Any]] = None,
        skip_orthogonality: bool = False,
        skip_robustness: bool = False,
    ) -> ValidationResult:
        """
        运行完整验证

        Args:
            factor_values: 因子值，index: date, columns: assets
            returns: 收益率，index: date, columns: assets
            factor_metadata: 因子元数据（可选）
            skip_orthogonality: 是否跳过正交化检验
            skip_robustness: 是否跳过稳健性检验

        Returns:
            ValidationResult 对象
        """
        logger.info("Starting factor validation...")

        errors = []
        details = {}

        # 计算未来收益率
        forward_returns = returns.shift(-1)

        # Step 1: IC 分析
        logger.info("Running IC analysis...")
        ic_result = self.ic_analyzer.run(
            factor_values,
            forward_returns,
            compute_decay=self.config.compute_decay,
        )
        details["ic"] = ic_result.to_dict()

        # 检查 IC 阈值
        if abs(ic_result.ic_mean) < self.config.ic_mean_min:
            errors.append(
                f"IC 均值 {ic_result.ic_mean:.4f} 低于阈值 {self.config.ic_mean_min}"
            )

        if abs(ic_result.ic_ir) < self.config.ic_ir_min:
            errors.append(
                f"IC 信息比率 {ic_result.ic_ir:.4f} 低于阈值 {self.config.ic_ir_min}"
            )

        # Step 2: 分组测试
        logger.info("Running group test...")
        group_result = self.group_tester.run(factor_values, forward_returns)
        details["group"] = group_result.to_dict()

        # 检查单调性阈值
        if group_result.monotonicity_score < self.config.monotonicity_min:
            errors.append(
                f"单调性得分 {group_result.monotonicity_score:.2f} 低于阈值 {self.config.monotonicity_min}"
            )

        # Step 3: 正交化检验（可选）
        ortho_result = None
        if not skip_orthogonality and self.ortho_tester._existing_factors:
            logger.info("Running orthogonality test...")
            ortho_result = self.ortho_tester.run(factor_values, forward_returns)
            details["orthogonality"] = ortho_result.to_dict()

            # 检查 IC 保留率
            if ortho_result.ic_retention < self.config.ortho_retention_min:
                errors.append(
                    f"正交化 IC 保留率 {ortho_result.ic_retention:.2f} 低于阈值 {self.config.ortho_retention_min}"
                )

        # Step 4: 稳健性检验（可选）
        robust_result = None
        if not skip_robustness:
            logger.info("Running robustness test...")
            robust_result = self.robust_tester.run(ic_result.ic_series)
            details["robustness"] = robust_result.to_dict()

            # 检查显著性
            if robust_result.nw_pvalue > self.config.nw_pvalue_max:
                errors.append(
                    f"Newey-West p 值 {robust_result.nw_pvalue:.4f} 高于阈值 {self.config.nw_pvalue_max}"
                )

        # 判断是否通过
        passed = len(errors) == 0

        result = ValidationResult(
            passed=passed,
            ic_mean=ic_result.ic_mean,
            ic_ir=ic_result.ic_ir,
            ic_tstat=ic_result.ic_tstat,
            monotonicity_score=group_result.monotonicity_score,
            ortho_ic_retention=ortho_result.ic_retention if ortho_result else 1.0,
            nw_pvalue=robust_result.nw_pvalue if robust_result else 1.0,
            details=details,
            errors=errors,
            ic_result=ic_result,
            group_result=group_result,
            ortho_result=ortho_result,
            robust_result=robust_result,
        )

        logger.info(f"Validation complete: passed={passed}, errors={len(errors)}")
        return result

    def validate_sync(
        self,
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> ValidationResult:
        """
        同步验证接口

        Args:
            factor_values: 因子值
            returns: 收益率
            **kwargs: 其他参数

        Returns:
            ValidationResult 对象
        """
        return asyncio.run(self.validate(factor_values, returns, **kwargs))

    def quick_validate(
        self,
        factor_values: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        快速验证（仅返回关键指标）

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            关键指标字典
        """
        forward_returns = returns.shift(-1)

        # IC 分析
        ic_result = self.ic_analyzer.run(factor_values, forward_returns, compute_decay=False)

        # 分组测试
        group_result = self.group_tester.run(factor_values, forward_returns)

        return {
            "ic_mean": ic_result.ic_mean,
            "ic_ir": ic_result.ic_ir,
            "monotonicity": group_result.monotonicity_score,
            "long_short": group_result.long_short_return,
        }


def validate_factor(
    factor_values: pd.DataFrame,
    returns: pd.DataFrame,
    config: Optional[ValidationConfig] = None,
) -> ValidationResult:
    """
    便捷验证函数

    Args:
        factor_values: 因子值
        returns: 收益率
        config: 验证配置

    Returns:
        ValidationResult 对象
    """
    validator = FactorValidator(config=config)
    return validator.validate_sync(factor_values, returns)
