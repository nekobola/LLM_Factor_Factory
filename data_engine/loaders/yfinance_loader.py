"""
YFinance 数据加载器

使用 Yahoo Finance API 加载全球市场数据。
"""

from datetime import date, datetime, timedelta
from typing import Optional, List, Union
import asyncio

import pandas as pd
import numpy as np
from loguru import logger

from ..base_loader import DataLoader, DataLoaderConfig, AdjustType


class YFinanceLoader(DataLoader):
    """
    YFinance 数据加载器

    支持全球市场股票、ETF、指数数据。

    Example:
        loader = YFinanceLoader()

        # 加载美股数据
        df = await loader.load_ohlcv(
            symbols=["AAPL", "MSFT", "GOOGL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # 加载 A 股数据 (需要添加后缀)
        df = await loader.load_ohlcv(
            symbols=["000001.SZ", "600000.SS"],
        )
    """

    # Yahoo Finance 股票代码后缀映射
    MARKET_SUFFIX = {
        "US": "",        # 美股无后缀
        "SH": ".SS",     # 上交所
        "SZ": ".SZ",     # 深交所
        "HK": ".HK",     # 港股
        "JP": ".T",      # 日本
        "UK": ".L",      # 英国
        "DE": ".DE",     # 德国
    }

    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """
        初始化 YFinance 加载器

        Args:
            config: 加载器配置
        """
        super().__init__(config)
        self._yf = None

    def _init_api(self):
        """初始化 YFinance"""
        if self._yf is not None:
            return

        try:
            import yfinance as yf
            self._yf = yf
            logger.info("YFinance initialized")
        except ImportError:
            raise ImportError("Please install yfinance: pip install yfinance")

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
            DataFrame with MultiIndex (date, asset)
        """
        self._init_api()

        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        symbols = symbols or self.config.symbols

        if not symbols:
            raise ValueError("Symbols must be provided for YFinance")

        # 检查缓存
        cache_key = self.get_cache_key("ohlcv", symbols, str(start_date), str(end_date))
        cached = self.get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.info(f"Loading OHLCV data from YFinance: {start_date} to {end_date}")

        # YFinance 支持批量下载
        df = await self._load_batch(symbols, start_date, end_date)

        # 格式化
        if not df.empty:
            df = self._format_ohlcv(df)

        self.save_to_cache(cache_key, df)

        logger.info(f"Loaded {len(df)} rows of OHLCV data")
        return df

    async def _load_batch(
        self,
        symbols: List[str],
        start_date: Union[str, date],
        end_date: Union[str, date],
    ) -> pd.DataFrame:
        """批量加载数据"""
        loop = asyncio.get_event_loop()

        def _load():
            # YFinance 下载
            data = self._yf.download(
                tickers=" ".join(symbols),
                start=start_date,
                end=end_date,
                group_by="ticker",
                auto_adjust=self.config.adjust_type != AdjustType.NONE,
                progress=False,
            )
            return data

        df = await loop.run_in_executor(None, _load)
        return df

    def _format_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化 OHLCV 数据"""
        if df.empty:
            return df

        # 处理单股票和多股票的不同格式
        if isinstance(df.columns, pd.MultiIndex):
            # 多股票格式
            records = []
            for date in df.index:
                for symbol in df.columns.levels[1]:
                    try:
                        row = df.loc[date, (slice(None), symbol)]
                        if isinstance(row, pd.Series):
                            record = {
                                "date": date,
                                "asset": symbol,
                                "open": row.get("Open", np.nan),
                                "high": row.get("High", np.nan),
                                "low": row.get("Low", np.nan),
                                "close": row.get("Close", np.nan),
                                "volume": row.get("Volume", np.nan),
                                "adj_close": row.get("Adj Close", np.nan),
                            }
                            records.append(record)
                    except Exception:
                        continue

            result = pd.DataFrame(records)
        else:
            # 单股票格式
            result = df.reset_index()
            result["asset"] = df.columns.name or "SINGLE"
            result = result.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adj_close",
            })

        # 转换日期
        result["date"] = pd.to_datetime(result["date"])

        # 创建 MultiIndex
        result = result.set_index(["date", "asset"])

        return result

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

        # 使用收盘价或复权价
        price_col = "adj_close" if "adj_close" in ohlcv.columns else "close"
        close = ohlcv[price_col].unstack(level="asset")

        returns = self.compute_returns(close, method="simple")
        returns = returns.stack().to_frame("return")

        return returns

    async def get_stock_info(self, symbol: str) -> dict:
        """
        获取股票信息

        Args:
            symbol: 股票代码

        Returns:
            股票信息字典
        """
        self._init_api()

        loop = asyncio.get_event_loop()

        def _load():
            ticker = self._yf.Ticker(symbol)
            return ticker.info

        try:
            info = await loop.run_in_executor(None, _load)
            return info
        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
            return {}

    async def get_dividends(
        self,
        symbol: str,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
    ) -> pd.Series:
        """
        获取股息数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股息时间序列
        """
        self._init_api()

        loop = asyncio.get_event_loop()

        def _load():
            ticker = self._yf.Ticker(symbol)
            return ticker.dividends

        dividends = await loop.run_in_executor(None, _load)

        if dividends.empty:
            return dividends

        # 过滤日期
        if start_date:
            dividends = dividends[dividends.index >= pd.to_datetime(start_date)]
        if end_date:
            dividends = dividends[dividends.index <= pd.to_datetime(end_date)]

        return dividends

    @classmethod
    def convert_symbol(
        cls,
        code: str,
        from_market: str = "SH",
    ) -> str:
        """
        转换股票代码格式

        Args:
            code: 原始代码 (如 "000001")
            from_market: 市场 ("SH", "SZ", "US")

        Returns:
            YFinance 格式代码

        Example:
            >>> YFinanceLoader.convert_symbol("000001", "SZ")
            "000001.SZ"
        """
        suffix = cls.MARKET_SUFFIX.get(from_market, "")
        return f"{code}{suffix}"
