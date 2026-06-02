"""
AkShare 数据加载器

使用 AkShare 加载 A 股市场数据（免费，无需 Token）。
"""

import os
from datetime import date, datetime, timedelta
from typing import Optional, List, Union
import asyncio

import pandas as pd
import numpy as np
from loguru import logger

from ..base_loader import DataLoader, DataLoaderConfig, AdjustType


class AkShareLoader(DataLoader):
    """
    AkShare 数据加载器

    AkShare 是免费的开源财经数据接口，无需 Token。

    支持的数据:
    - A 股日线行情
    - A 股分钟线
    - 指数数据
    - 财务数据

    Example:
        loader = AkShareLoader()

        # 加载日线数据
        df = await loader.load_ohlcv(
            symbols=["000001", "600000"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
    """

    def __init__(self, config: Optional[DataLoaderConfig] = None, disable_proxy: bool = True):
        """
        初始化 AkShare 加载器

        Args:
            config: 加载器配置
            disable_proxy: 是否禁用代理 (默认 True，避免代理导致的连接问题)
        """
        super().__init__(config)

        # 禁用代理
        if disable_proxy:
            for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                os.environ.pop(k, None)

        self._ak = None

    def _init_api(self):
        """初始化 AkShare"""
        if self._ak is not None:
            return

        # 彻底清除代理设置
        for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
            os.environ.pop(k, None)
        # 设置不使用代理
        os.environ['NO_PROXY'] = '*'
        os.environ['no_proxy'] = '*'

        try:
            import akshare as ak
            self._ak = ak
            logger.info("AkShare initialized (proxy disabled)")
        except ImportError:
            raise ImportError("Please install akshare: pip install akshare")

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
            symbols: 股票代码列表 (如 ["000001", "600000"])
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表

        Returns:
            DataFrame with MultiIndex (date, asset)
        """
        self._init_api()

        start_date = self._format_date(start_date or self.config.start_date)
        end_date = self._format_date(end_date or self.config.end_date)
        symbols = symbols or self.config.symbols

        # 检查缓存
        cache_key = self.get_cache_key("ohlcv", symbols, start_date, end_date)
        cached = self.get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Loading OHLCV data from AkShare: {start_date} to {end_date}")

        df_list = []

        if symbols:
            for symbol in symbols:
                try:
                    df_symbol = await self._load_single_symbol(
                        symbol, start_date, end_date
                    )
                    df_list.append(df_symbol)
                except Exception as e:
                    logger.warning(f"Failed to load {symbol}: {e}")
                    continue

                await asyncio.sleep(0.1)
        else:
            # 加载全市场数据
            df = await self._load_all_symbols(start_date, end_date)
            if not df.empty:
                df_list.append(df)

        if df_list:
            df = pd.concat(df_list, ignore_index=True)
        else:
            df = pd.DataFrame()

        # 格式化
        if not df.empty:
            df = self._format_ohlcv(df, start_date, end_date)

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
            # 尝试新浪接口 (更稳定)
            try:
                # 新浪接口需要带市场前缀
                if symbol.startswith('6'):
                    sina_symbol = f"sh{symbol}"
                else:
                    sina_symbol = f"sz{symbol}"
                df = self._ak.stock_zh_a_daily(
                    symbol=sina_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=self._get_adjust_param(),
                )
                df["asset"] = symbol
                return df
            except Exception as e:
                logger.warning(f"Sina API failed for {symbol}: {e}")
                # 备用东方财富接口
                try:
                    df = self._ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust=self._get_adjust_param(),
                    )
                    df["asset"] = symbol
                    return df
                except Exception as e2:
                    logger.warning(f"Eastmoney API also failed for {symbol}: {e2}")
                    return pd.DataFrame()

        df = await loop.run_in_executor(None, _load)
        return df

    async def _load_all_symbols(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """加载全市场数据"""
        # AkShare 不直接支持全市场加载，需要逐个加载
        logger.warning("AkShare does not support loading all symbols at once. Use symbols parameter.")
        return pd.DataFrame()

    def _get_adjust_param(self) -> str:
        """获取复权参数"""
        if self.config.adjust_type == AdjustType.FRONT:
            return "qfq"  # 前复权
        elif self.config.adjust_type == AdjustType.BACK:
            return "hfq"  # 后复权
        else:
            return ""  # 不复权 (新浪接口用空字符串)

    def _format_ohlcv(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """格式化 OHLCV 数据"""
        if df.empty:
            return df

        # AkShare 列名映射
        column_mapping = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "change",
            "换手率": "turnover",
        }

        df = df.rename(columns=column_mapping)

        # 转换日期
        df["date"] = pd.to_datetime(df["date"])

        # 过滤日期范围
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        # 创建 MultiIndex
        df = df.set_index(["date", "asset"])

        return df

    async def load_returns(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        """加载收益率数据"""
        ohlcv = await self.load_ohlcv(symbols, start_date, end_date)

        if ohlcv.empty:
            return pd.DataFrame()

        close = ohlcv["close"].unstack(level="asset")
        returns = self.compute_returns(close, method="simple")
        returns = returns.stack().to_frame("return")

        return returns

    async def get_stock_list(self) -> pd.DataFrame:
        """
        获取 A 股股票列表

        Returns:
            股票列表 DataFrame
        """
        self._init_api()

        loop = asyncio.get_event_loop()

        def _load():
            return self._ak.stock_info_a_code_name()

        df = await loop.run_in_executor(None, _load)
        logger.info(f"Loaded {len(df)} stocks")
        return df

    async def load_index_data(
        self,
        index_code: str = "sh000300",
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        """
        加载指数数据

        Args:
            index_code: 指数代码 (如 "sh000300", "sz399006")
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            指数数据 DataFrame
        """
        self._init_api()

        start_date = self._format_date(start_date or self.config.start_date)
        end_date = self._format_date(end_date or self.config.end_date)

        loop = asyncio.get_event_loop()

        def _load():
            return self._ak.stock_zh_index_daily(symbol=index_code)

        df = await loop.run_in_executor(None, _load)

        if df.empty:
            return df

        # 格式化
        df = df.rename(columns={
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])

        # 过滤日期
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

        return df.set_index("date")

    @staticmethod
    def _format_date(d: Union[str, date]) -> str:
        """格式化日期 (YYYYMMDD)"""
        if isinstance(d, str):
            return d.replace("-", "")
        return d.strftime("%Y%m%d")

    async def load_hs300_constituents(self) -> List[str]:
        """
        获取沪深300成分股列表

        Returns:
            成分股代码列表
        """
        self._init_api()
        loop = asyncio.get_event_loop()

        def _load():
            # 获取沪深300成分股
            df = self._ak.index_stock_cons_weight_csindex(symbol="000300")
            return df

        try:
            df = await loop.run_in_executor(None, _load)
            if df.empty:
                # 备用：返回常见大盘股
                logger.warning("Failed to load HS300 constituents, using default list")
                return [
                    "600519", "600036", "601318", "600276", "600030",
                    "601166", "600000", "600887", "601328", "600016",
                    "601398", "601288", "600837", "600009", "600010",
                    "600011", "600015", "600018", "600019", "600025",
                ]

            # 提取股票代码
            codes = df["成分券代码"].tolist() if "成分券代码" in df.columns else df.iloc[:, 0].tolist()
            logger.info(f"Loaded {len(codes)} HS300 constituents")
            return codes
        except Exception as e:
            logger.warning(f"Failed to load HS300 constituents: {e}, using default list")
            return [
                "600519", "600036", "601318", "600276", "600030",
                "601166", "600000", "600887", "601328", "600016",
                "601398", "601288", "600837", "600009", "600010",
                "600011", "600015", "600018", "600019", "600025",
            ]

    async def load_hs300(
        self,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        n_stocks: int = 50,
    ) -> pd.DataFrame:
        """
        加载沪深300成分股数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            n_stocks: 加载股票数量限制

        Returns:
            DataFrame with MultiIndex (date, asset)
        """
        logger.info("Loading HS300 data from AkShare...")

        # 获取成分股列表
        constituents = await self.load_hs300_constituents()
        constituents = constituents[:n_stocks]

        logger.info(f"Loading data for {len(constituents)} stocks...")

        # 加载数据
        df = await self.load_ohlcv(
            symbols=constituents,
            start_date=start_date,
            end_date=end_date,
        )

        return df
