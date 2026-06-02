"""
因子抽象基类与数据结构

所有因子必须继承 BaseFactor，实现统一的输入输出接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
import time

import pandas as pd
import numpy as np


class FactorCategory(str, Enum):
    """因子分类"""
    MOMENTUM = "动量"
    VALUE = "价值"
    QUALITY = "质量"
    GROWTH = "成长"
    VOLATILITY = "波动率"
    LIQUIDITY = "流动性"
    SENTIMENT = "情绪"
    TECHNICAL = "技术"
    FUNDAMENTAL = "基本面"
    ALTERNATIVE = "另类"
    CUSTOM = "自定义"


class DataFrequency(str, Enum):
    """数据频率"""
    DAILY = "日线"
    MINUTE_1 = "1分钟"
    MINUTE_5 = "5分钟"
    MINUTE_15 = "15分钟"
    MINUTE_30 = "30分钟"
    MINUTE_60 = "60分钟"
    WEEKLY = "周线"
    MONTHLY = "月线"


@dataclass
class FactorMetadata:
    """
    因子元数据

    Attributes:
        name: 因子名称（唯一标识）
        category: 因子分类
        description: 因子逻辑描述
        formula: 数学表达式（可执行字符串）
        lookback_period: 回看窗口期
        frequency: 数据频率
        author: 作者
        version: 版本号
        tags: 标签列表
        created_at: 创建时间
        updated_at: 更新时间
        params: 可配置参数
    """
    name: str
    category: FactorCategory
    description: str
    formula: str
    lookback_period: int
    frequency: DataFrequency
    author: str = "LLM"
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "formula": self.formula,
            "lookback_period": self.lookback_period,
            "frequency": self.frequency.value,
            "author": self.author,
            "version": self.version,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorMetadata":
        """从字典创建"""
        data["category"] = FactorCategory(data["category"])
        data["frequency"] = DataFrequency(data["frequency"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class FactorResult:
    """
    因子计算结果

    Attributes:
        values: 因子值 DataFrame，index: (date, asset) MultiIndex
        metadata: 因子元数据
        computation_time: 计算耗时（秒）
        data_quality_score: 数据质量评分 (0-1)
        missing_ratio: 缺失值比例
        warnings: 警告信息列表
    """
    values: pd.DataFrame
    metadata: FactorMetadata
    computation_time: float
    data_quality_score: float = 1.0
    missing_ratio: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """计算数据质量指标"""
        if self.values is not None and not self.values.empty:
            self.missing_ratio = self.values.isna().mean().mean()
            # 数据质量评分：基于缺失率和有效数据量
            self.data_quality_score = max(0, 1 - self.missing_ratio * 2)

    def get_factor_series(self) -> pd.Series:
        """获取因子值 Series（长格式）"""
        if isinstance(self.values.index, pd.MultiIndex):
            return self.values.stack()
        return self.values.squeeze()


@dataclass
class ValidationResult:
    """
    因子验证结果

    Attributes:
        passed: 是否通过验证
        ic_mean: IC 均值
        ic_ir: IC 信息比率 (IC_mean / IC_std)
        ic_tstat: IC t 统计量
        monotonicity_score: 分组单调性得分
        ortho_ic_retention: 正交化后 IC 保留率
        nw_pvalue: Newey-West 调整后 p 值
        details: 详细验证结果
        errors: 错误信息列表
    """
    passed: bool
    ic_mean: float = 0.0
    ic_ir: float = 0.0
    ic_tstat: float = 0.0
    monotonicity_score: float = 0.0
    ortho_ic_retention: float = 1.0
    nw_pvalue: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class BaseFactor(ABC):
    """
    因子抽象基类

    所有因子必须继承此类，实现 compute() 和 validate_inputs() 方法。
    因子计算流程：validate_inputs → preprocess → compute → postprocess

    Example:
        class MomentumFactor(BaseFactor):
            def __init__(self, lookback: int = 20):
                metadata = FactorMetadata(
                    name=f"momentum_{lookback}d",
                    category=FactorCategory.MOMENTUM,
                    description=f"{lookback}日动量因子",
                    formula=f"close / close.shift({lookback}) - 1",
                    lookback_period=lookback,
                    frequency=DataFrequency.DAILY,
                )
                super().__init__(metadata)
                self.lookback = lookback

            def compute(self, data: pd.DataFrame) -> pd.Series:
                close = data['close'].unstack(level='asset')
                momentum = close / close.shift(self.lookback) - 1
                return momentum.stack()

            def validate_inputs(self, data: pd.DataFrame) -> bool:
                return 'close' in data.columns
    """

    # 验证阈值（类级别配置）
    VALIDATION_THRESHOLDS = {
        "ic_mean_min": 0.02,
        "ic_ir_min": 0.5,
        "monotonicity_min": 0.8,
        "ortho_retention_min": 0.5,
        "nw_pvalue_max": 0.05,
    }

    def __init__(self, metadata: FactorMetadata):
        """
        初始化因子

        Args:
            metadata: 因子元数据
        """
        self.metadata = metadata
        self._cache: Dict[str, Any] = {}
        self._preprocessors: List[Callable] = []
        self._postprocessors: List[Callable] = []

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        核心计算逻辑

        Args:
            data: 输入数据 DataFrame
                  - index: (date, asset) MultiIndex
                  - columns: OHLCV + 其他特征

        Returns:
            因子值 Series，index: (date, asset) MultiIndex
        """
        pass

    @abstractmethod
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据完整性

        Args:
            data: 输入数据 DataFrame

        Returns:
            是否验证通过
        """
        pass

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理

        包括：去极值、填充缺失值、标准化等

        Args:
            data: 原始数据

        Returns:
            预处理后的数据
        """
        # 应用预处理链
        processed = data.copy()
        for preprocessor in self._preprocessors:
            processed = preprocessor(processed)
        return processed

    def postprocess(self, factor_values: pd.Series) -> pd.Series:
        """
        因子值后处理

        包括：Winsorize、Z-Score 标准化、中性化等

        Args:
            factor_values: 原始因子值

        Returns:
            处理后的因子值
        """
        processed = factor_values.copy()

        # Winsorize (3σ)
        processed = self._winsorize(processed, n_std=3.0)

        # Z-Score 标准化（截面）
        processed = self._zscore_standardize(processed)

        # 应用后处理链
        for postprocessor in self._postprocessors:
            processed = postprocessor(processed)

        return processed

    @staticmethod
    def _winsorize(series: pd.Series, n_std: float = 3.0) -> pd.Series:
        """
        去极值（MAD 方法）

        Args:
            series: 输入序列
            n_std: 标准差倍数

        Returns:
            去极值后的序列
        """
        if series.empty:
            return series

        # 使用 MAD 方法更稳健
        median = series.median()
        mad = (series - median).abs().median()
        lower = median - n_std * mad * 1.4826  # MAD → std 转换系数
        upper = median + n_std * mad * 1.4826

        return series.clip(lower=lower, upper=upper)

    @staticmethod
    def _zscore_standardize(series: pd.Series) -> pd.Series:
        """
        Z-Score 标准化（截面）

        Args:
            series: 输入序列（MultiIndex: date, asset）

        Returns:
            标准化后的序列
        """
        if not isinstance(series.index, pd.MultiIndex):
            return series

        # 按日期截面标准化
        def standardize_cross_section(group):
            mean = group.mean()
            std = group.std()
            if std == 0 or pd.isna(std):
                return group * 0
            return (group - mean) / std

        return series.groupby(level='date').transform(standardize_cross_section)

    def add_preprocessor(self, func: Callable):
        """添加预处理器"""
        self._preprocessors.append(func)

    def add_postprocessor(self, func: Callable):
        """添加后处理器"""
        self._postprocessors.append(func)

    def __call__(self, data: pd.DataFrame, skip_validation: bool = False) -> FactorResult:
        """
        统一调用入口

        Args:
            data: 输入数据
            skip_validation: 是否跳过输入验证

        Returns:
            FactorResult 对象

        Raises:
            ValueError: 输入验证失败
        """
        start_time = time.time()
        warnings = []

        # Step 1: 验证输入
        if not skip_validation and not self.validate_inputs(data):
            raise ValueError(f"输入数据验证失败，缺少必要字段")

        # Step 2: 预处理
        processed_data = self.preprocess(data)

        # Step 3: 计算
        try:
            factor_values = self.compute(processed_data)
        except Exception as e:
            raise RuntimeError(f"因子计算失败: {e}") from e

        # Step 4: 后处理
        factor_values = self.postprocess(factor_values)

        # Step 5: 构建结果
        computation_time = time.time() - start_time

        # 转换为 DataFrame 格式
        if isinstance(factor_values.index, pd.MultiIndex):
            values_df = factor_values.unstack(level='asset')
        else:
            values_df = factor_values.to_frame(self.metadata.name)

        return FactorResult(
            values=values_df,
            metadata=self.metadata,
            computation_time=computation_time,
            warnings=warnings,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.metadata.name}', category={self.metadata.category.value})"

    def __str__(self) -> str:
        return f"[{self.metadata.category.value}] {self.metadata.name}: {self.metadata.description}"
