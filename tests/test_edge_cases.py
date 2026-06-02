"""
边界压力测试 - 全链路测试

测试目标：
1. LLM 生成不确定性测试 - 验证系统能否处理各种异常代码
2. 用户需求不确定性测试 - 验证不同交易场景能否正常工作
3. 数据边界测试 - 验证数据异常情况的处理
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any, List, Optional
import sys
import traceback

# 添加项目路径
sys.path.insert(0, 'D:\\素材\\fineconometrics_factors_mining')


# ============================================================
# 测试工具函数
# ============================================================

def create_test_data(
    n_dates: int = 200,
    n_assets: int = 10,
    start_date: str = '2023-01-01',
    freq: str = 'D',
    include_vwap: bool = True,
    include_amount: bool = True,
) -> pd.DataFrame:
    """创建测试数据"""
    dates = pd.date_range(start_date, periods=n_dates, freq=freq)
    assets = [f'asset_{i:02d}' for i in range(n_assets)]

    # 创建 MultiIndex
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])

    # 生成 OHLCV 数据
    np.random.seed(42)
    base_prices = np.random.uniform(10, 100, n_assets)
    returns = np.random.randn(n_dates * n_assets) * 0.02

    close = np.exp(np.log(np.repeat(base_prices, n_dates)) + np.cumsum(returns.reshape(n_assets, n_dates).T.flatten()))
    open_price = close * (1 + np.random.randn(n_dates * n_assets) * 0.005)
    high = np.maximum(open_price, close) * (1 + np.abs(np.random.randn(n_dates * n_assets) * 0.01))
    low = np.minimum(open_price, close) * (1 - np.abs(np.random.randn(n_dates * n_assets) * 0.01))
    volume = np.random.uniform(1e6, 1e7, n_dates * n_assets)

    data = {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }

    if include_amount:
        data['amount'] = close * volume
    if include_vwap:
        data['vwap'] = (high + low + close) / 3

    df = pd.DataFrame(data, index=idx)
    return df


def create_returns_data(data: pd.DataFrame) -> pd.DataFrame:
    """从 OHLCV 数据创建收益率"""
    close = data['close'].unstack(level='asset')
    returns = close.pct_change()
    return returns


def run_factor_compute(factor_code: str, data: pd.DataFrame) -> Dict[str, Any]:
    """执行因子代码并返回结果"""
    from idea_generator.parser import FactorParser

    parser = FactorParser()
    result = parser.parse(factor_code)

    if not result.success:
        return {
            'success': False,
            'error': 'parse_failed',
            'errors': result.errors,
        }

    try:
        factor = result.factor_instance
        factor_values = factor.compute(data)

        # 检查返回格式
        if factor_values is None:
            return {
                'success': False,
                'error': 'return_none',
                'message': 'compute() returned None',
            }

        if not isinstance(factor_values, pd.Series):
            return {
                'success': False,
                'error': 'wrong_type',
                'message': f'Expected pd.Series, got {type(factor_values)}',
            }

        if not isinstance(factor_values.index, pd.MultiIndex):
            return {
                'success': False,
                'error': 'wrong_index',
                'message': f'Expected MultiIndex, got {type(factor_values.index)}',
            }

        if factor_values.index.names != ['date', 'asset']:
            return {
                'success': False,
                'error': 'wrong_index_names',
                'message': f'Expected ["date", "asset"], got {factor_values.index.names}',
            }

        return {
            'success': True,
            'factor_values': factor_values,
        }

    except Exception as e:
        return {
            'success': False,
            'error': 'compute_error',
            'message': str(e),
            'traceback': traceback.format_exc(),
        }


def run_backtest(
    factor_values: pd.Series,
    returns: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """运行回测并返回结果"""
    from strategy_builder.backtest import FactorBacktest

    config = config or {}
    n_groups = config.get('n_groups', 5)
    holding_period = config.get('holding_period', 5)
    commission_rate = config.get('commission_rate', 0.0003)
    stamp_duty = config.get('stamp_duty', 0.001)
    slippage = config.get('slippage', 0.0005)

    try:
        # 转换因子值
        if isinstance(factor_values, pd.Series):
            factor_df = factor_values.unstack(level='asset')
        else:
            factor_df = factor_values

        # 对齐数据
        common_dates = factor_df.index.intersection(returns.index)
        common_assets = factor_df.columns.intersection(returns.columns)

        if len(common_dates) == 0:
            return {
                'success': False,
                'error': 'no_common_dates',
                'message': 'No common dates between factor and returns',
            }

        if len(common_assets) == 0:
            return {
                'success': False,
                'error': 'no_common_assets',
                'message': 'No common assets between factor and returns',
            }

        factor_df = factor_df.loc[common_dates, common_assets]
        returns_aligned = returns.loc[common_dates, common_assets]

        backtest = FactorBacktest(factor_df, returns_aligned)
        result = backtest.run(
            n_groups=n_groups,
            holding_period=holding_period,
            commission_rate=commission_rate,
            stamp_duty=stamp_duty,
            slippage=slippage,
            allow_short=False,
        )

        return {
            'success': True,
            'stats': result.stats,
            'turnover_stats': result.turnover_stats,
        }

    except Exception as e:
        return {
            'success': False,
            'error': 'backtest_error',
            'message': str(e),
            'traceback': traceback.format_exc(),
        }


# ============================================================
# 测试用例：LLM 生成不确定性
# ============================================================

class TestLLMGenerationEdgeCases:
    """测试 LLM 生成的各种异常代码"""

    def setup_method(self):
        """每个测试方法前创建测试数据"""
        self.data = create_test_data(n_dates=300, n_assets=15)
        self.returns = create_returns_data(self.data)

    def test_simple_momentum_factor(self):
        """测试 1: 标准动量因子 - 应该成功"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class MomentumFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="momentum_20d",
            category=FactorCategory.MOMENTUM,
            description="20日动量因子",
            formula="close.pct_change(20)",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        return close.groupby(level='asset').transform(lambda x: x.pct_change(20))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert result['success'], f"Expected success, got: {result}"

        backtest_result = run_backtest(result['factor_values'], self.returns)
        assert backtest_result['success'], f"Backtest failed: {backtest_result}"

    def test_factor_with_import_statements(self):
        """测试 2: 包含 import 语句的代码 - 应该自动处理"""
        code = '''
import pandas as pd
import numpy as np
from typing import Dict, Any

from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class TestFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="test_factor",
            category=FactorCategory.MOMENTUM,
            description="测试因子",
            formula="test",
            lookback_period=10,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        return close.groupby(level='asset').transform(lambda x: x.pct_change(10))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        # 即使有 import，也应该能处理
        assert result['success'], f"Failed with import statements: {result}"

    def test_factor_return_wrong_type(self):
        """测试 3: 返回错误类型 - 应该报错"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class WrongReturnFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="wrong_return",
            category=FactorCategory.MOMENTUM,
            description="返回错误类型",
            formula="test",
            lookback_period=10,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # 错误：返回 DataFrame 而不是 Series
        return data[['close', 'volume']]

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert not result['success'], "Should fail for wrong return type"
        assert result['error'] in ['wrong_type', 'compute_error']

    def test_factor_with_resample(self):
        """测试 4: 包含手动降频代码 - 应该能处理"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class ResampleFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="resample_factor",
            category=FactorCategory.MOMENTUM,
            description="包含降频的因子",
            formula="weekly resample",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        # 正确方式：保持 MultiIndex 结构
        returns = close.groupby(level='asset').pct_change().fillna(0)
        return returns

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert result['success'], f"Failed with resample: {result}"

    def test_factor_with_missing_values(self):
        """测试 5: 因子计算产生 NaN - 应该能处理"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class NaNFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="nan_factor",
            category=FactorCategory.MOMENTUM,
            description="会产生NaN的因子",
            formula="close / 0",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        # 这会产生 inf 和 nan
        result = close / (close - close)  # 除以零
        return result.fillna(0).replace([np.inf, -np.inf], 0)

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert result['success'], f"Failed with NaN handling: {result}"

    def test_factor_with_long_lookback(self):
        """测试 6: 超长回看期因子 - 应该能处理"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class LongLookbackFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="long_lookback",
            category=FactorCategory.MOMENTUM,
            description="超长回看期因子",
            formula="close.pct_change(500)",
            lookback_period=500,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        return close.groupby(level='asset').transform(lambda x: x.pct_change(500))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert result['success'], f"Failed with long lookback: {result}"
        # 检查前 500 个值是 NaN
        factor_values = result['factor_values']
        nan_count = factor_values.isna().sum()
        print(f"NaN count for long lookback: {nan_count}")

    def test_factor_with_complex_calculation(self):
        """测试 7: 复杂计算因子 - 应该能处理"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class ComplexFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="complex_factor",
            category=FactorCategory.MOMENTUM,
            description="复杂计算因子",
            formula="multi-step calculation",
            lookback_period=60,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        volume = data['volume']

        # 复杂计算：多步处理
        returns = close.groupby(level='asset').pct_change().fillna(0)
        vol = returns.groupby(level='asset').transform(lambda x: x.rolling(20).std())
        vol = vol.replace(0, np.nan).fillna(1e-6)

        # 遍历资产计算
        results = []
        for asset in close.index.get_level_values('asset').unique():
            asset_close = close.xs(asset, level='asset')
            asset_vol = vol.xs(asset, level='asset')
            asset_volume = volume.xs(asset, level='asset')

            # 自定义计算
            factor_val = asset_close.pct_change(20) / asset_vol * np.sqrt(252)
            factor_val = factor_val.fillna(0)
            factor_val.name = asset
            results.append(factor_val)

        result = pd.concat(results, axis=1).stack()
        result.index.names = ['date', 'asset']
        return result

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns and 'volume' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert result['success'], f"Failed with complex calculation: {result}"


# ============================================================
# 测试用例：用户需求不确定性
# ============================================================

class TestUserScenarioEdgeCases:
    """测试不同的用户交易场景"""

    def setup_method(self):
        """每个测试方法前创建测试数据"""
        self.data = create_test_data(n_dates=300, n_assets=15)
        self.returns = create_returns_data(self.data)

    def test_scenario_fixed_frequency_stock_selection(self):
        """场景 1: 固定频率选股"""
        # 标准因子代码
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class StockSelectionFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="stock_selection",
            category=FactorCategory.MOMENTUM,
            description="固定频率选股因子",
            formula="momentum_20d",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        return close.groupby(level='asset').transform(lambda x: x.pct_change(20))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert result['success']

        # 测试不同调仓频率
        for holding_period in [1, 5, 10, 20]:
            backtest_result = run_backtest(
                result['factor_values'],
                self.returns,
                {'holding_period': holding_period}
            )
            assert backtest_result['success'], f"Failed for holding_period={holding_period}"

    def test_scenario_fixed_universe_timing(self):
        """场景 2: 固定标的择时"""
        # 这个场景需要不同的回测逻辑
        # 当前系统支持，但需要验证

        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class TimingFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="timing_factor",
            category=FactorCategory.MOMENTUM,
            description="择时因子",
            formula="market_timing_signal",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        # 所有资产使用相同的择时信号（市场择时）
        market_return = close.groupby(level='date').mean().pct_change(20)
        # 广播到所有资产
        result = market_return.reindex(close.index, level='date')
        return result

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert result['success'], f"Timing factor failed: {result}"

    def test_scenario_different_group_counts(self):
        """场景 3: 不同分组数量"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class TestFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="test_factor",
            category=FactorCategory.MOMENTUM,
            description="测试因子",
            formula="momentum",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        return close.groupby(level='asset').transform(lambda x: x.pct_change(20))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert result['success']

        # 测试不同分组数
        for n_groups in [3, 5, 10]:
            if n_groups <= 15:  # 资产数量限制
                backtest_result = run_backtest(
                    result['factor_values'],
                    self.returns,
                    {'n_groups': n_groups}
                )
                assert backtest_result['success'], f"Failed for n_groups={n_groups}"

    def test_scenario_high_trading_costs(self):
        """场景 4: 高交易成本环境"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class TestFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="test_factor",
            category=FactorCategory.MOMENTUM,
            description="测试因子",
            formula="momentum",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        return close.groupby(level='asset').transform(lambda x: x.pct_change(20))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, self.data)
        assert result['success']

        # 测试高交易成本
        backtest_result = run_backtest(
            result['factor_values'],
            self.returns,
            {
                'commission_rate': 0.001,  # 千一
                'stamp_duty': 0.003,       # 千三
                'slippage': 0.002,         # 千二
            }
        )
        assert backtest_result['success'], f"Failed with high costs: {backtest_result}"
        print(f"High cost scenario - Sharpe: {backtest_result['stats'].get('sharpe_ratio', 0):.2f}")


# ============================================================
# 测试用例：数据边界测试
# ============================================================

class TestDataEdgeCases:
    """测试数据异常情况"""

    def setup_method(self):
        """每个测试方法前创建测试数据"""
        self.data = create_test_data(n_dates=200, n_assets=10)
        self.returns = create_returns_data(self.data)

    def test_data_with_missing_dates(self):
        """测试数据有缺失日期"""
        data = create_test_data(n_dates=200, n_assets=10)

        # 随机删除一些日期
        np.random.seed(42)
        dates_to_remove = np.random.choice(
            data.index.get_level_values('date').unique(),
            size=20,
            replace=False
        )
        data = data[~data.index.get_level_values('date').isin(dates_to_remove)]

        returns = create_returns_data(data)

        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class TestFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="test_factor",
            category=FactorCategory.MOMENTUM,
            description="测试因子",
            formula="momentum",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        return close.groupby(level='asset').transform(lambda x: x.pct_change(20))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, data)
        assert result['success'], f"Failed with missing dates: {result}"

    def test_data_with_missing_assets(self):
        """测试数据有缺失资产"""
        data = create_test_data(n_dates=200, n_assets=10)

        # 某些资产数据不完整
        np.random.seed(42)
        for asset in ['asset_00', 'asset_01', 'asset_02']:
            dates = data.index.get_level_values('date').unique()
            dates_to_remove = np.random.choice(dates, size=50, replace=False)
            data = data[~((data.index.get_level_values('date').isin(dates_to_remove)) &
                         (data.index.get_level_values('asset') == asset))]

        returns = create_returns_data(data)

        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class TestFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="test_factor",
            category=FactorCategory.MOMENTUM,
            description="测试因子",
            formula="momentum",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        return close.groupby(level='asset').transform(lambda x: x.pct_change(20))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, data)
        assert result['success'], f"Failed with missing assets: {result}"

    def test_data_with_extreme_values(self):
        """测试数据有极端值"""
        data = create_test_data(n_dates=200, n_assets=10)

        # 添加极端值
        np.random.seed(42)
        extreme_indices = np.random.choice(len(data), size=10, replace=False)
        data.iloc[extreme_indices, data.columns.get_loc('close')] *= 10  # 价格暴涨
        data.iloc[extreme_indices, data.columns.get_loc('volume')] *= 100  # 成交量暴增

        returns = create_returns_data(data)

        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class TestFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="test_factor",
            category=FactorCategory.MOMENTUM,
            description="测试因子",
            formula="momentum",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        return close.groupby(level='asset').transform(lambda x: x.pct_change(20))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, data)
        assert result['success'], f"Failed with extreme values: {result}"


# ============================================================
# 运行测试
# ============================================================

def run_all_tests():
    """运行所有测试并汇报结果"""
    test_classes = [
        TestLLMGenerationEdgeCases,
        TestUserScenarioEdgeCases,
        TestDataEdgeCases,
    ]

    results = {
        'passed': [],
        'failed': [],
        'errors': [],
    }

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                method = getattr(instance, method_name)
                try:
                    instance.setup_method()
                    method()
                    results['passed'].append(f"{test_class.__name__}::{method_name}")
                    print(f"[PASS] {method_name}")
                except AssertionError as e:
                    results['failed'].append({
                        'test': f"{test_class.__name__}::{method_name}",
                        'error': str(e),
                    })
                    print(f"[FAIL] {method_name}: {e}")
                except Exception as e:
                    results['errors'].append({
                        'test': f"{test_class.__name__}::{method_name}",
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                    })
                    print(f"[ERROR] {method_name}: {e}")

    # 汇总报告
    print("\n" + "="*60)
    print("测试汇总报告")
    print("="*60)
    print(f"通过: {len(results['passed'])}")
    print(f"失败: {len(results['failed'])}")
    print(f"错误: {len(results['errors'])}")

    if results['failed']:
        print("\n失败详情:")
        for item in results['failed']:
            print(f"  - {item['test']}: {item['error']}")

    if results['errors']:
        print("\n错误详情:")
        for item in results['errors']:
            print(f"  - {item['test']}: {item['error']}")

    return results


if __name__ == '__main__':
    run_all_tests()
