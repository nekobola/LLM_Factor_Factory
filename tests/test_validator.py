"""
测试验证器模块
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from validator.ic_analyzer import ICAnalyzer, ICResult
from validator.group_test import GroupTester, GroupTestResult
from validator.orthogonality import OrthogonalityTester, OrthogonalityResult
from validator.robustness import RobustnessTester, RobustnessResult
from validator.validator import FactorValidator, ValidationConfig


@pytest.fixture
def sample_factor_data():
    """生成测试因子数据"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    assets = [f"STOCK_{i}" for i in range(50)]

    # 创建因子值
    factor_values = pd.DataFrame(
        np.random.randn(252, 50),
        index=dates,
        columns=assets,
    )

    return factor_values


@pytest.fixture
def sample_returns():
    """生成测试收益率数据"""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252)
    assets = [f"STOCK_{i}" for i in range(50)]

    # 创建收益率（与因子有一定相关性）
    returns = pd.DataFrame(
        np.random.randn(252, 50) * 0.02,
        index=dates,
        columns=assets,
    )

    return returns


class TestICAnalyzer:
    """测试 IC 分析器"""

    def test_compute_ic(self, sample_factor_data, sample_returns):
        """测试 IC 计算"""
        analyzer = ICAnalyzer(method="spearman")

        ic_series = analyzer.compute_ic(sample_factor_data, sample_returns)

        assert isinstance(ic_series, pd.Series)
        assert len(ic_series) == len(sample_factor_data)

    def test_run_analysis(self, sample_factor_data, sample_returns):
        """测试完整分析"""
        analyzer = ICAnalyzer(method="spearman")

        result = analyzer.run(sample_factor_data, sample_returns)

        assert isinstance(result, ICResult)
        assert isinstance(result.ic_mean, float)
        assert isinstance(result.ic_ir, float)
        assert isinstance(result.ic_series, pd.Series)


class TestGroupTester:
    """测试分组测试器"""

    def test_assign_groups(self, sample_factor_data):
        """测试分组"""
        tester = GroupTester(n_groups=5)

        # 单期分组
        groups = tester.assign_groups(sample_factor_data.iloc[0])

        assert isinstance(groups, pd.Series)
        assert len(groups) == sample_factor_data.shape[1]

    def test_run_group_test(self, sample_factor_data, sample_returns):
        """测试分组测试"""
        tester = GroupTester(n_groups=5)

        result = tester.run(sample_factor_data, sample_returns)

        assert isinstance(result, GroupTestResult)
        assert result.n_groups == 5
        assert 0 <= result.monotonicity_score <= 1


class TestRobustnessTester:
    """测试稳健性检验"""

    def test_schwert_lags(self):
        """测试 Schwert 准则"""
        tester = RobustnessTester()

        # 不同样本量
        assert tester.select_lags_schwert(100) >= 1
        assert tester.select_lags_schwert(1000) >= 1

    def test_run_robustness(self, sample_factor_data, sample_returns):
        """测试稳健性检验"""
        analyzer = ICAnalyzer(method="spearman")
        ic_series = analyzer.compute_ic(sample_factor_data, sample_returns)

        tester = RobustnessTester()

        result = tester.run(ic_series)

        assert isinstance(result, RobustnessResult)
        assert isinstance(result.nw_tstat, float)
        assert isinstance(result.nw_pvalue, float)
        assert 0 <= result.nw_pvalue <= 1


class TestFactorValidator:
    """测试因子验证器"""

    def test_validator_initialization(self):
        """测试验证器初始化"""
        config = ValidationConfig(
            ic_mean_min=0.02,
            ic_ir_min=0.5,
        )

        validator = FactorValidator(config=config)

        assert validator.config.ic_mean_min == 0.02
        assert validator.config.ic_ir_min == 0.5

    def test_quick_validate(self, sample_factor_data, sample_returns):
        """测试快速验证"""
        validator = FactorValidator()

        result = validator.quick_validate(sample_factor_data, sample_returns)

        assert "ic_mean" in result
        assert "ic_ir" in result
        assert "monotonicity" in result
        assert "long_short" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
