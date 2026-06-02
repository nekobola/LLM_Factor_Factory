"""
Tushare 数据加载器

使用 Tushare API 加载 A 股市场数据。

需要设置环境变量: TUSHARE_TOKEN
"""

import os
from datetime import date, datetime, timedelta
from typing import Optional, List, Union
import asyncio

import pandas as pd
import numpy as np
from loguru import logger

from ..base_loader import DataLoader, DataLoaderConfig, AdjustType


class TushareLoader(DataLoader):
    """
    Tushare 数据加载器

    支持的数据:
    - 日线行情 (daily)
    - 分钟线 (需积分)
    - 复权因子
    - 财务数据

    Example:
        loader = TushareLoader(token="your_token")

        # 加载日线数据
        df = await loader.load_ohlcv(
            symbols=["000001.SZ", "600000.SH"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
    """

    def __init__(
        self,
        token: Optional[str] = None,
        config: Optional[DataLoaderConfig] = None,
        disable_proxy: bool = True,
    ):
        """
        初始化 Tushare 加载器

        Args:
            token: Tushare API Token，默认从环境变量 TUSHARE_TOKEN 读取
            config: 加载器配置
            disable_proxy: 是否禁用代理 (默认 True，避免代理导致的连接问题)
        """
        super().__init__(config)

        # 禁用代理
        if disable_proxy:
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)
            os.environ.pop('http_proxy', None)
            os.environ.pop('https_proxy', None)

        self.token = token or os.getenv("TUSHARE_TOKEN")
        if not self.token:
            logger.warning("Tushare token not provided, some features may not work")

        self._pro = None
        self._initialized = False

    def _init_api(self):
        """初始化 Tushare API"""
        if self._initialized:
            return

        try:
            import tushare as ts
            ts.set_token(self.token)
            self._pro = ts.pro_api()
            self._initialized = True
            logger.info("Tushare API initialized")
        except ImportError:
            raise ImportError("Please install tushare: pip install tushare")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Tushare API: {e}")

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
            symbols: 股票代码列表 (如 ["000001.SZ", "600000.SH"])
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表

        Returns:
            DataFrame with MultiIndex (date, asset)
        """
        self._init_api()

        # 参数处理
        start_date = self._format_date(start_date or self.config.start_date)
        end_date = self._format_date(end_date or self.config.end_date)
        symbols = symbols or self.config.symbols

        # 检查缓存
        cache_key = self.get_cache_key("ohlcv", symbols, start_date, end_date)
        cached = self.get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Loading OHLCV data from Tushare: {start_date} to {end_date}")

        # 加载数据
        if symbols:
            # 按股票代码逐个加载
            df_list = []
            for symbol in symbols:
                try:
                    df_symbol = await self._load_single_symbol(symbol, start_date, end_date)
                    df_list.append(df_symbol)
                except Exception as e:
                    logger.warning(f"Failed to load {symbol}: {e}")
                    continue

                # 添加延迟避免频率限制
                await asyncio.sleep(0.1)

            if df_list:
                df = pd.concat(df_list, ignore_index=True)
            else:
                df = pd.DataFrame()
        else:
            # 加载全市场数据
            df = await self._load_all_symbols(start_date, end_date)

        # 标准化格式
        if not df.empty:
            df = self._format_ohlcv(df)

        # 保存缓存
        self.save_to_cache(cache_key, df)

        logger.info(f"Loaded {len(df)} rows of OHLCV data")
        return df

    async def _load_single_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """加载单只股票数据"""
        loop = asyncio.get_event_loop()

        def _load():
            return self._pro.daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date,
            )

        df = await loop.run_in_executor(None, _load)

        # 加载复权因子
        if self.config.adjust_type != AdjustType.NONE:
            adj_df = await self._load_adj_factor(symbol, start_date, end_date)
            if not adj_df.empty:
                df = df.merge(adj_df, on=["ts_code", "trade_date"], how="left")

        return df

    async def _load_all_symbols(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """加载全市场数据"""
        loop = asyncio.get_event_loop()

        def _load():
            return self._pro.daily(
                start_date=start_date,
                end_date=end_date,
            )

        df = await loop.run_in_executor(None, _load)
        return df

    async def _load_adj_factor(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """加载复权因子"""
        loop = asyncio.get_event_loop()

        def _load():
            return self._pro.adj_factor(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date,
            )

        return await loop.run_in_executor(None, _load)

    def _format_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化 OHLCV 数据"""
        if df.empty:
            return df

        # 重命名列
        column_mapping = {
            "ts_code": "asset",
            "trade_date": "date",
            "vol": "volume",
            "amount": "amount",
        }
        df = df.rename(columns=column_mapping)

        # 转换日期
        df["date"] = pd.to_datetime(df["date"])

        # 应用复权
        if self.config.adjust_type != AdjustType.NONE and "adj_factor" in df.columns:
            df = self._apply_adjust(df)

        # 创建 MultiIndex
        df = df.set_index(["date", "asset"])

        # 标准化列名
        df = self.standardize_ohlcv(df.reset_index()).set_index(["date", "asset"])

        return df

    def _apply_adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用复权因子"""
        if "adj_factor" not in df.columns:
            return df

        adj_factor = df["adj_factor"]

        if self.config.adjust_type == AdjustType.FRONT:
            # 前复权：以最新价格为基准
            latest_factor = adj_factor.iloc[0]
            scale = latest_factor / adj_factor

            df["open"] = df["open"] * scale.values
            df["high"] = df["high"] * scale.values
            df["low"] = df["low"] * scale.values
            df["close"] = df["close"] * scale.values

        elif self.config.adjust_type == AdjustType.BACK:
            # 后复权：以上市价格为基准
            earliest_factor = adj_factor.iloc[-1]
            scale = adj_factor / earliest_factor

            df["open"] = df["open"] * scale.values
            df["high"] = df["high"] * scale.values
            df["low"] = df["low"] * scale.values
            df["close"] = df["close"] * scale.values

        return df

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
            收益率 DataFrame
        """
        # 加载 OHLCV 数据
        ohlcv = await self.load_ohlcv(symbols, start_date, end_date)

        if ohlcv.empty:
            return pd.DataFrame()

        # 提取收盘价
        close = ohlcv["close"].unstack(level="asset")

        # 计算收益率
        returns = self.compute_returns(close, method="simple")

        # 转换回 MultiIndex 格式
        returns = returns.stack().to_frame("return")

        return returns

    async def load_index_constituents(
        self,
        index_code: str = "000300.SH",
    ) -> List[str]:
        """
        加载指数成分股

        Args:
            index_code: 指数代码

        Returns:
            成分股代码列表
        """
        self._init_api()

        loop = asyncio.get_event_loop()

        def _load():
            return self._pro.index_weight(
                index_code=index_code,
                start_date=(date.today() - timedelta(days=30)).strftime("%Y%m%d"),
            )

        df = await loop.run_in_executor(None, _load)

        if df.empty:
            return []

        # 取最新成分股
        latest_date = df["trade_date"].max()
        constituents = df[df["trade_date"] == latest_date]["con_code"].tolist()

        logger.info(f"Loaded {len(constituents)} constituents for {index_code}")
        return constituents

    async def load_hs300(
        self,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        """
        加载沪深300成分股数据

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with MultiIndex (date, asset)
        """
        logger.info("Loading HS300 constituents...")

        # 1. 获取沪深300成分股
        constituents = await self.load_index_constituents("000300.SH")

        if not constituents:
            logger.warning("No constituents found, using default list")
            # 使用一些常见的大盘股作为默认
            constituents = [
                "600519.SH", "600036.SH", "601318.SH", "600276.SH", "600030.SH",
                "601166.SH", "600000.SH", "600887.SH", "601328.SH", "600016.SH",
            ]

        logger.info(f"Loading data for {len(constituents)} constituents...")

        # 2. 加载成分股数据
        df = await self.load_ohlcv(
            symbols=constituents[:50],  # 限制数量避免 API 限制
            start_date=start_date,
            end_date=end_date,
        )

        return df

    @staticmethod
    def _format_date(d: Union[str, date]) -> str:
        """格式化日期为 Tushare 格式 (YYYYMMDD)"""
        if isinstance(d, str):
            return d.replace("-", "")
        return d.strftime("%Y%m%d")
