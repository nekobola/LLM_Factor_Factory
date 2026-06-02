"""
测试 BaseFactor 核心功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date

from core.base_factor import (
    BaseFactor,
    FactorMetadata,
    FactorResult,
    FactorCategory,
    DataFrequency,
)


class TestFactorMetadata:
    """测试 FactorMetadata"""

    def test_create_metadata(self):
        """测试创建元数据"""
        metadata = FactorMetadata(
            name="test_factor",
            category=FactorCategory.MOMENTUM,
            description="测试因子",
            formula="close.pct_change(20)",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )

        assert metadata.name == "test_factor"
        assert metadata.category == FactorCategory.MOMENTUM
        assert metadata.lookback_period == 20

    def test_to_dict(self):
        """测试序列化"""
        metadata = FactorMetadata(
            name="test_factor",
            category=FactorCategory.MOMENTUM,
            description="测试因子",
            formula="close.pct_change(20)",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )

        data = metadata.to_dict()
        assert data["name"] == "test_factor"
        assert data["category"] == "动量"

    def test_from_dict(self):
        """测试反序列化"""
        data = {
            "name": "test_factor",
            "category": "动量",
            "description": "测试因子",
            "formula": "close.pct_change(20)",
            "lookback_period": 20,
            "frequency": "日线",
            "author": "LLM",
            "version": "1.0.0",
            "tags": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "params": {},
        }

        metadata = FactorMetadata.from_dict(data)
        assert metadata.name == "test_factor"
        assert metadata.category == FactorCategory.MOMENTUM


class TestFactorResult:
    """测试 FactorResult"""

    def test_create_result(self):
        """测试创建结果"""
        dates = pd.date_range("2023-01-01", periods=10)
        assets = ["A", "B", "C"]
        values = pd.DataFrame(
            np.random.randn(10, 3),
            index=dates,
            columns=assets,
        )

        metadata = FactorMetadata(
            name="test_factor",
            category=FactorCategory.MOMENTUM,
            description="测试因子",
            formula="test",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )

        result = FactorResult(
            values=values,
            metadata=metadata,
            computation_time=0.5,
        )

        assert result.values.shape == (10, 3)
        assert result.computation_time == 0.5
        assert 0 <= result.data_quality_score <= 1


class MomentumFactor(BaseFactor):
    """测试用动量因子"""

    def __init__(self, lookback: int = 20):
        metadata = FactorMetadata(
            name=f"momentum_{lookback}d",
            category=FactorCategory.MOMENTUM,
            description=f"{lookback}日动量因子",
            formula=f"close.pct_change({lookback})",
            lookback_period=lookback,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)
        self.lookback = lookback

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """计算动量"""
        if isinstance(data.index, pd.MultiIndex):
            close = data['close'].unstack(level='asset')
            momentum = close.pct_change(self.lookback)
            return momentum.stack()
        else:
            return data['close'].pct_change(self.lookback)

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """验证输入"""
        return 'close' in data.columns


class TestBaseFactor:
    """测试 BaseFactor"""

    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        dates = pd.date_range("2023-01-01", periods=100)
        assets = ["A", "B", "C"]

        # 创建 MultiIndex 数据
        index = pd.MultiIndex.from_product(
            [dates, assets],
            names=["date", "asset"]
        )

        data = pd.DataFrame({
            'open': np.random.randn(300) * 10 + 100,
            'high': np.random.randn(300) * 10 + 105,
            'low': np.random.randn(300) * 10 + 95,
            'close': np.random.randn(300) * 10 + 100,
            'volume': np.random.randn(300).abs() * 1000000,
        }, index=index)

        return data

    def test_factor_initialization(self):
        """测试因子初始化"""
        factor = MomentumFactor(lookback=20)

        assert factor.metadata.name == "momentum_20d"
        assert factor.metadata.category == FactorCategory.MOMENTUM

    def test_validate_inputs(self, sample_data):
        """测试输入验证"""
        factor = MomentumFactor(lookback=20)

        assert factor.validate_inputs(sample_data) is True

        # 缺少 close 列
        bad_data = sample_data.drop(columns=['close'])
        assert factor.validate_inputs(bad_data) is False

    def test_compute(self, sample_data):
        """测试计算"""
        factor = MomentumFactor(lookback=20)

        result = factor.compute(sample_data)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

    def test_call(self, sample_data):
        """测试调用"""
        factor = MomentumFactor(lookback=20)

        result = factor(sample_data)

        assert isinstance(result, FactorResult)
        assert result.values is not None
        assert result.computation_time > 0

    def test_winsorize(self):
        """测试去极值"""
        series = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])

        result = BaseFactor._winsorize(series, n_std=3.0)

        # 极值应该被截断
        assert result.max() < 100

    def test_zscore_standardize(self):
        """测试标准化"""
        dates = pd.date_range("2023-01-01", periods=10)
        assets = ["A", "B", "C"]
        index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])

        series = pd.Series(np.random.randn(30), index=index)

        result = BaseFactor._zscore_standardize(series)

        # 检查截面标准化后的均值接近 0
        for date in dates:
            cross_section = result.loc[date]
            assert abs(cross_section.mean()) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
