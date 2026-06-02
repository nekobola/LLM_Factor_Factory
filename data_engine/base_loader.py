"""
数据加载器基类

定义统一的数据加载接口，支持多种数据源。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, Dict, Any, List, Union
import asyncio

import pandas as pd
import numpy as np
from loguru import logger


class DataType(str, Enum):
    """数据类型"""
    DAILY = "日线"
    MINUTE = "分钟线"
    TICK = "Tick"
    ADJ_FACTOR = "复权因子"
    FINANCIAL = "财务数据"
    INDUSTRY = "行业分类"


class AdjustType(str, Enum):
    """复权类型"""
    NONE = "不复权"
    FRONT = "前复权"
    BACK = "后复权"


@dataclass
class DataLoaderConfig:
    """
    数据加载器配置

    Attributes:
        start_date: 开始日期
        end_date: 结束日期
        symbols: 股票代码列表
        data_type: 数据类型
        adjust_type: 复权类型
        fields: 字段列表
        cache_dir: 缓存目录
        use_cache: 是否使用缓存
    """
    start_date: Union[str, date]
    end_date: Union[str, date]
    symbols: Optional[List[str]] = None
    data_type: DataType = DataType.DAILY
    adjust_type: AdjustType = AdjustType.FRONT
    fields: Optional[List[str]] = None
    cache_dir: str = ".cache/data"
    use_cache: bool = True

    def __post_init__(self):
        if isinstance(self.start_date, str):
            self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()
        if isinstance(self.end_date, str):
            self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d").date()


class DataLoader(ABC):
    """
    数据加载器抽象基类

    所有数据源加载器必须实现此接口。

    Example:
        class MyLoader(DataLoader):
            def load_ohlcv(self, symbols, start, end):
                # 实现具体加载逻辑
                return df

            def load_returns(self, symbols, start, end):
                # 实现收益率加载
                return returns_df
    """

    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """
        初始化加载器

        Args:
            config: 加载器配置
        """
        self.config = config or DataLoaderConfig(
            start_date="2020-01-01",
            end_date=date.today(),
        )
        self._cache: Dict[str, pd.DataFrame] = {}
        self._last_load_time: Optional[datetime] = None

        logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    async def load_ohlcv(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        加载 OHLCV 数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表

        Returns:
            DataFrame with MultiIndex (date, asset), columns: OHLCV
        """
        pass

    @abstractmethod
    async def load_returns(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        """
        加载收益率数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with MultiIndex (date, asset), values: returns
        """
        pass

    async def load(
        self,
        lookback_period: int = 252,
        frequency: str = "日线",
        symbols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        通用加载接口

        根据回看期和频率加载数据。

        Args:
            lookback_period: 回看期（天数）
            frequency: 数据频率
            symbols: 股票代码列表

        Returns:
            OHLCV DataFrame
        """
        from datetime import timedelta

        end_date = self.config.end_date
        start_date = end_date - timedelta(days=lookback_period * 2)  # 预留更多数据

        return await self.load_ohlcv(
            symbols=symbols or self.config.symbols,
            start_date=start_date,
            end_date=end_date,
        )

    def get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        import hashlib
        key_str = f"{args}_{kwargs}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        if not self.config.use_cache:
            return None
        return self._cache.get(key)

    def save_to_cache(self, key: str, data: pd.DataFrame):
        """保存数据到缓存"""
        if self.config.use_cache:
            self._cache[key] = data

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

    @staticmethod
    def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化 OHLCV 数据格式

        确保输出格式为 MultiIndex (date, asset), columns: open, high, low, close, volume

        Args:
            df: 原始数据

        Returns:
            标准化后的数据
        """
        # 标准化列名
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            "Amount": "amount",
            "VWAP": "vwap",
        }

        df = df.rename(columns=column_mapping)

        # 确保有必要的列
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing column: {col}")

        return df

    @staticmethod
    def compute_returns(
        prices: pd.DataFrame,
        method: str = "simple",
    ) -> pd.DataFrame:
        """
        计算收益率

        Args:
            prices: 价格数据 (index: date, columns: assets)
            method: 计算方法 ("simple" 或 "log")

        Returns:
            收益率 DataFrame
        """
        if method == "simple":
            returns = prices.pct_change()
        elif method == "log":
            returns = np.log(prices) - np.log(prices.shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")

        return returns

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
