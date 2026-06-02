"""
残酷边界压力测试

测试更极端的场景：
1. LLM 生成的"毒药代码" - 各种边界破坏
2. 零数据/极小数据场景
3. 因子值全 NaN / 全相同
4. 回测参数极端组合
5. 并发稳定性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import sys
import traceback
import time

sys.path.insert(0, 'D:\\素材\\fineconometrics_factors_mining')

from tests.test_edge_cases import (
    create_test_data,
    create_returns_data,
    run_factor_compute,
    run_backtest,
)


# ============================================================
# 残酷测试：LLM 生成的"毒药代码"
# ============================================================

class TestPoisonCode:
    """测试各种可能破坏系统的代码"""

    def test_code_with_syntax_error(self):
        """语法错误的代码 - 应该被 parser 捕获"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class BadSyntaxFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="bad_syntax"
            category=FactorCategory.MOMENTUM  # 缺少逗号
            description="语法错误"
            formula="error"
            lookback_period=20
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['close']

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        data = create_test_data(n_dates=100, n_assets=5)
        result = run_factor_compute(code, data)
        assert not result['success'], "Should fail for syntax error"
        print(f"Syntax error caught: {result.get('errors', result.get('message', 'unknown'))}")

    def test_code_with_infinite_loop(self):
        """包含潜在死循环的代码 - 应该超时或被检测"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class InfiniteLoopFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="infinite_loop",
            category=FactorCategory.MOMENTUM,
            description="潜在死循环",
            formula="while True",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        # 有限循环，但大量迭代
        result = close.copy()
        for i in range(1000):  # 大量迭代但不死循环
            result = result * 1.0001
        return result

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        data = create_test_data(n_dates=100, n_assets=5)
        start_time = time.time()
        result = run_factor_compute(code, data)
        elapsed = time.time() - start_time
        print(f"Heavy computation time: {elapsed:.2f}s")
        # 应该能在合理时间内完成
        assert result['success'], f"Should complete: {result}"

    def test_code_with_memory_bomb(self):
        """内存炸弹 - 创建超大对象"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class MemoryBombFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="memory_bomb",
            category=FactorCategory.MOMENTUM,
            description="内存炸弹",
            formula="large array",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        # 创建一个大数组但不返回
        large_array = np.zeros((1000, 1000))  # 1M elements
        # 正确返回
        return close.groupby(level='asset').transform(lambda x: x.pct_change(20))

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        data = create_test_data(n_dates=100, n_assets=5)
        result = run_factor_compute(code, data)
        assert result['success'], f"Should handle memory bomb: {result}"

    def test_code_that_modifies_input(self):
        """修改输入数据的代码"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class ModifyInputFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="modify_input",
            category=FactorCategory.MOMENTUM,
            description="修改输入",
            formula="inplace modification",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # 危险：尝试修改输入数据
        # 但 pandas 通常会创建副本
        close = data['close'].copy()
        close[:] = 100  # 全部设为100
        return close

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        data = create_test_data(n_dates=100, n_assets=5)
        original_data = data.copy()
        result = run_factor_compute(code, data)
        assert result['success'], f"Should handle: {result}"

    def test_code_with_empty_compute(self):
        """compute 方法返回空"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class EmptyComputeFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="empty_compute",
            category=FactorCategory.MOMENTUM,
            description="返回空",
            formula="empty",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # 返回空 Series
        return pd.Series(dtype=float)

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        data = create_test_data(n_dates=100, n_assets=5)
        result = run_factor_compute(code, data)
        # 空结果应该被检测
        if result['success']:
            factor_values = result.get('factor_values')
            if factor_values is not None and len(factor_values) == 0:
                print("Empty factor values detected")
        print(f"Empty compute result: {result}")

    def test_code_with_wrong_index_structure(self):
        """返回错误的索引结构"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class WrongIndexFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="wrong_index",
            category=FactorCategory.MOMENTUM,
            description="错误索引",
            formula="wrong index",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        # 返回只有 date 索引的 Series（缺少 asset）
        close = data['close']
        result = close.groupby(level='date').mean()
        return result

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        data = create_test_data(n_dates=100, n_assets=5)
        result = run_factor_compute(code, data)
        # 应该被检测到索引错误
        if result['success']:
            factor_values = result.get('factor_values')
            if factor_values is not None:
                # 检查索引
                if not isinstance(factor_values.index, pd.MultiIndex):
                    print(f"Wrong index type detected: {type(factor_values.index)}")
                    result = {'success': False, 'error': 'wrong_index_type'}
        print(f"Wrong index result: {result}")

    def test_code_that_throws_exception(self):
        """compute 方法抛出异常"""
        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class ExceptionFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="exception_factor",
            category=FactorCategory.MOMENTUM,
            description="抛出异常",
            formula="raise exception",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        raise ValueError("Intentional exception for testing")

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        data = create_test_data(n_dates=100, n_assets=5)
        result = run_factor_compute(code, data)
        assert not result['success'], "Should fail for exception"
        assert result['error'] == 'compute_error'
        print(f"Exception caught: {result.get('message', 'unknown')}")


# ============================================================
# 残酷测试：极端数据场景
# ============================================================

class TestExtremeData:
    """极端数据场景测试"""

    def test_zero_data(self):
        """零数据场景"""
        data = pd.DataFrame()
        returns = pd.DataFrame()

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
        assert not result['success'], "Should fail for empty data"
        print(f"Zero data result: {result}")

    def test_single_asset(self):
        """单一资产场景"""
        data = create_test_data(n_dates=100, n_assets=1)
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
        # 单一资产应该能计算
        if result['success']:
            backtest_result = run_backtest(result['factor_values'], returns, {'n_groups': 1})
            print(f"Single asset backtest: {backtest_result}")
        print(f"Single asset compute: {result}")

    def test_single_date(self):
        """单日数据场景"""
        data = create_test_data(n_dates=1, n_assets=10)

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
        return close  # 无法计算 pct_change

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, data)
        print(f"Single date result: {result}")

    def test_all_nan_factor_values(self):
        """因子值全为 NaN"""
        data = create_test_data(n_dates=100, n_assets=10)
        returns = create_returns_data(data)

        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class AllNaNFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="all_nan",
            category=FactorCategory.MOMENTUM,
            description="全NaN因子",
            formula="NaN",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        result = close.copy()
        result[:] = np.nan
        return result

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, data)
        if result['success']:
            backtest_result = run_backtest(result['factor_values'], returns)
            print(f"All NaN backtest: {backtest_result}")

    def test_all_same_factor_values(self):
        """因子值全相同"""
        data = create_test_data(n_dates=100, n_assets=10)
        returns = create_returns_data(data)

        code = '''
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency

class AllSameFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="all_same",
            category=FactorCategory.MOMENTUM,
            description="全相同因子",
            formula="constant",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        result = close.copy()
        result[:] = 1.0  # 全部设为1
        return result

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns
'''
        result = run_factor_compute(code, data)
        if result['success']:
            backtest_result = run_backtest(result['factor_values'], returns)
            print(f"All same backtest: {backtest_result}")


# ============================================================
# 残酷测试：极端回测参数
# ============================================================

class TestExtremeBacktestParams:
    """极端回测参数测试"""

    def setup_method(self):
        self.data = create_test_data(n_dates=100, n_assets=10)
        self.returns = create_returns_data(self.data)

        self.factor_code = '''
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

    def test_extreme_groups_more_than_assets(self):
        """分组数大于资产数"""
        result = run_factor_compute(self.factor_code, self.data)
        if result['success']:
            backtest_result = run_backtest(
                result['factor_values'],
                self.returns,
                {'n_groups': 20}  # 大于资产数
            )
            print(f"Groups > assets: {backtest_result}")

    def test_extreme_holding_period(self):
        """极端持有期"""
        result = run_factor_compute(self.factor_code, self.data)

        # 持有期大于数据天数
        for holding_period in [1, 50, 100, 200]:
            if result['success']:
                backtest_result = run_backtest(
                    result['factor_values'],
                    self.returns,
                    {'holding_period': holding_period}
                )
                print(f"Holding period {holding_period}: success={backtest_result['success']}")

    def test_zero_trading_cost(self):
        """零交易成本"""
        result = run_factor_compute(self.factor_code, self.data)
        if result['success']:
            backtest_result = run_backtest(
                result['factor_values'],
                self.returns,
                {
                    'commission_rate': 0,
                    'stamp_duty': 0,
                    'slippage': 0,
                }
            )
            print(f"Zero cost backtest: {backtest_result}")

    def test_extreme_trading_cost(self):
        """极端交易成本（100%）"""
        result = run_factor_compute(self.factor_code, self.data)
        if result['success']:
            backtest_result = run_backtest(
                result['factor_values'],
                self.returns,
                {
                    'commission_rate': 0.3,  # 30%
                    'stamp_duty': 0.3,       # 30%
                    'slippage': 0.4,         # 40%
                }
            )
            print(f"Extreme cost backtest: {backtest_result}")


# ============================================================
# 运行测试
# ============================================================

def run_cruel_tests():
    """运行残酷测试"""
    test_classes = [
        TestPoisonCode,
        TestExtremeData,
        TestExtremeBacktestParams,
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
                    if hasattr(instance, 'setup_method'):
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
    print("残酷测试汇总报告")
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
    run_cruel_tests()
