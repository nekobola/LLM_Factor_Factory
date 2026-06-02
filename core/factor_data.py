"""
因子数据访问工具类

提供统一的数据访问接口，简化 MultiIndex 操作。
LLM 生成的代码可以使用这些工具函数，避免直接操作 MultiIndex。

Example:
    # 在因子 compute 方法中使用
    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = FactorData.get_column(data, 'close')
        returns = FactorData.pct_change_by_asset(close, 20)
        return returns
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Callable


class FactorData:
    """
    因子数据访问工具类

    封装 MultiIndex DataFrame 的常见操作，提供简单易用的 API。

    所有方法都保证：
    1. 输入是 MultiIndex (date, asset) 的 DataFrame 或 Series
    2. 输出保持相同的 MultiIndex 结构
    """

    @staticmethod
    def get_column(data: pd.DataFrame, column: str) -> pd.Series:
        """
        获取单列数据

        Args:
            data: MultiIndex DataFrame
            column: 列名

        Returns:
            MultiIndex Series
        """
        return data[column]

    @staticmethod
    def get_assets(data: Union[pd.DataFrame, pd.Series]) -> List:
        """获取所有资产列表"""
        return data.index.get_level_values('asset').unique().tolist()

    @staticmethod
    def get_dates(data: Union[pd.DataFrame, pd.Series]) -> List:
        """获取所有日期列表"""
        return data.index.get_level_values('date').unique().tolist()

    @staticmethod
    def get_asset_data(data: Union[pd.DataFrame, pd.Series], asset: str) -> Union[pd.DataFrame, pd.Series]:
        """
        获取单个资产的数据

        Args:
            data: MultiIndex DataFrame 或 Series
            asset: 资产代码

        Returns:
            单资产的 DataFrame 或 Series（索引为 date）
        """
        return data.xs(asset, level='asset')

    @staticmethod
    def pct_change_by_asset(series: pd.Series, periods: int = 1) -> pd.Series:
        """
        按资产计算百分比变化

        Args:
            series: MultiIndex Series
            periods: 周期数

        Returns:
            MultiIndex Series
        """
        return series.groupby(level='asset').transform(
            lambda x: x.pct_change(periods)
        )

    @staticmethod
    def shift_by_asset(series: pd.Series, periods: int = 1) -> pd.Series:
        """
        按资产平移数据

        Args:
            series: MultiIndex Series
            periods: 平移周期数

        Returns:
            MultiIndex Series
        """
        return series.groupby(level='asset').shift(periods)

    @staticmethod
    def rolling_by_asset(
        series: pd.Series,
        window: int,
        func: Union[str, Callable],
        min_periods: Optional[int] = None,
    ) -> pd.Series:
        """
        按资产滚动计算

        Args:
            series: MultiIndex Series
            window: 窗口大小
            func: 计算函数 ('mean', 'std', 'sum', 'max', 'min') 或自定义函数
            min_periods: 最小周期数

        Returns:
            MultiIndex Series
        """
        if isinstance(func, str):
            return series.groupby(level='asset').transform(
                lambda x: getattr(x.rolling(window, min_periods=min_periods), func)()
            )
        else:
            return series.groupby(level='asset').transform(
                lambda x: x.rolling(window, min_periods=min_periods).apply(func)
            )

    @staticmethod
    def groupby_date(series: pd.Series) -> pd.Series:
        """
        按日期分组（用于截面操作）

        Args:
            series: MultiIndex Series

        Returns:
            分组对象，可用于 transform
        """
        return series.groupby(level='date')

    @staticmethod
    def groupby_asset(series: pd.Series) -> pd.Series:
        """
        按资产分组（用于时序操作）

        Args:
            series: MultiIndex Series

        Returns:
            分组对象，可用于 transform
        """
        return series.groupby(level='asset')

    @staticmethod
    def cross_sectional_rank(series: pd.Series) -> pd.Series:
        """
        截面排名（每日对资产排名）

        Args:
            series: MultiIndex Series

        Returns:
            排名 Series（每日排名 0 到 N-1）
        """
        return series.groupby(level='date').rank()

    @staticmethod
    def cross_sectional_zscore(series: pd.Series) -> pd.Series:
        """
        截面标准化（每日 Z-score）

        Args:
            series: MultiIndex Series

        Returns:
            标准化后的 Series
        """
        return series.groupby(level='date').transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )

    @staticmethod
    def cross_sectional_quantile(series: pd.Series, q: int = 5) -> pd.Series:
        """
        截面分位数分组

        Args:
            series: MultiIndex Series
            q: 分组数

        Returns:
            分组标签 Series（0 到 q-1）
        """
        return series.groupby(level='date').transform(
            lambda x: pd.qcut(x, q=q, labels=False, duplicates='drop')
        )

    @staticmethod
    def combine_results(
        results: List[pd.Series],
        method: str = 'stack',
    ) -> pd.Series:
        """
        合并多个资产的结果

        Args:
            results: 各资产的 Series 列表（每个 Series 索引为 date，name 为 asset）
            method: 合并方法 ('stack', 'concat')

        Returns:
            MultiIndex Series
        """
        if method == 'stack':
            result = pd.concat(results, axis=1).stack()
            result.index.names = ['date', 'asset']
            return result
        else:
            return pd.concat(results)

    @staticmethod
    def safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
        fill_value: float = 0.0,
    ) -> pd.Series:
        """
        安全除法（处理除零）

        Args:
            numerator: 分子
            denominator: 分母
            fill_value: 除零时的填充值

        Returns:
            结果 Series
        """
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(fill_value)

    @staticmethod
    def winsorize(
        series: pd.Series,
        limits: tuple = (0.01, 0.01),
    ) -> pd.Series:
        """
        缩尾处理

        Args:
            series: MultiIndex Series
            limits: (下限, 上限) 分位数

        Returns:
            处理后的 Series
        """
        lower_limit = series.quantile(limits[0])
        upper_limit = series.quantile(1 - limits[1])
        return series.clip(lower=lower_limit, upper=upper_limit)

    @staticmethod
    def normalize(series: pd.Series) -> pd.Series:
        """
        标准化 (Z-score)

        Args:
            series: MultiIndex Series

        Returns:
            标准化后的 Series
        """
        return (series - series.mean()) / (series.std() + 1e-8)


# ============================================================
# 便捷函数（可直接在因子代码中使用）
# ============================================================

def get_close(data: pd.DataFrame) -> pd.Series:
    """获取收盘价"""
    return FactorData.get_column(data, 'close')

def get_open(data: pd.DataFrame) -> pd.Series:
    """获取开盘价"""
    return FactorData.get_column(data, 'open')

def get_high(data: pd.DataFrame) -> pd.Series:
    """获取最高价"""
    return FactorData.get_column(data, 'high')

def get_low(data: pd.DataFrame) -> pd.Series:
    """获取最低价"""
    return FactorData.get_column(data, 'low')

def get_volume(data: pd.DataFrame) -> pd.Series:
    """获取成交量"""
    return FactorData.get_column(data, 'volume')

def get_amount(data: pd.DataFrame) -> pd.Series:
    """获取成交额"""
    return FactorData.get_column(data, 'amount')

def _ensure_series(series: Union[pd.Series, np.ndarray], reference: Optional[pd.Series] = None) -> pd.Series:
    """
    确保输入是 pd.Series，如果是 numpy array 则转换

    Args:
        series: 输入数据
        reference: 参考 Series，用于获取索引

    Returns:
        pd.Series
    """
    if isinstance(series, pd.Series):
        return series
    if isinstance(series, np.ndarray):
        if reference is not None and isinstance(reference, pd.Series):
            return pd.Series(series, index=reference.index)
        raise ValueError("Cannot convert numpy array to Series without reference index")
    raise ValueError(f"Expected pd.Series or np.ndarray, got {type(series)}")


def pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """按资产计算收益率"""
    series = _ensure_series(series)
    return FactorData.pct_change_by_asset(series, periods)

def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """按资产滚动均值"""
    series = _ensure_series(series)
    return FactorData.rolling_by_asset(series, window, 'mean')

def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """按资产滚动标准差"""
    series = _ensure_series(series)
    return FactorData.rolling_by_asset(series, window, 'std')

def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """按资产滚动求和"""
    series = _ensure_series(series)
    return FactorData.rolling_by_asset(series, window, 'sum')

def shift(series: pd.Series, periods: int = 1) -> pd.Series:
    """按资产平移"""
    series = _ensure_series(series)
    return FactorData.shift_by_asset(series, periods)

def cross_rank(series: pd.Series) -> pd.Series:
    """截面排名"""
    return FactorData.cross_sectional_rank(series)

def cross_zscore(series: pd.Series) -> pd.Series:
    """截面标准化"""
    return FactorData.cross_sectional_zscore(series)

def where(condition: pd.Series, x: Union[pd.Series, float, int], y: Union[pd.Series, float, int]) -> pd.Series:
    """
    条件选择，保持 MultiIndex 结构

    类似于 np.where，但返回 pd.Series 保持索引

    Args:
        condition: 布尔条件 Series
        x: 条件为 True 时的值
        y: 条件为 False 时的值

    Returns:
        pd.Series，保持 condition 的索引

    Example:
        # 正向资金流
        mf_pos = where(tp > tp_lag, mf, 0)
    """
    if isinstance(condition, np.ndarray):
        raise TypeError("condition must be pd.Series, not numpy array. Use: where(tp > tp_lag, mf, 0)")

    result = condition.copy()
    result = result.astype(object)

    if isinstance(x, pd.Series):
        result[condition] = x[condition]
    else:
        result[condition] = x

    if isinstance(y, pd.Series):
        result[~condition] = y[~condition]
    else:
        result[~condition] = y

    return result
