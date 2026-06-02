"""
多模型 LLM 客户端

支持 Claude (Anthropic) 和 OpenAI 模型切换。
"""

import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Literal, Union
import asyncio

from loguru import logger


class LLMProvider(str, Enum):
    """LLM 提供商"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    """
    LLM 配置

    Attributes:
        provider: 提供商
        model: 模型名称
        api_key: API Key
        base_url: API 基础 URL (用于 OpenAI 兼容接口)
        temperature: 温度参数
        max_tokens: 最大输出 token
        timeout: 请求超时时间（秒）
        system_prompt: 系统提示
    """
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # 自定义 API endpoint
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0  # 超时时间
    system_prompt: str = ""

    # Claude 模型映射
    CLAUDE_MODELS = {
        "claude-4-sonnet": "claude-sonnet-4-20250514",
        "claude-4-opus": "claude-opus-4-20250514",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    }

    # OpenAI 模型映射
    OPENAI_MODELS = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4-turbo",
    }

    def __post_init__(self):
        # 设置默认 API Key
        if self.api_key is None:
            if self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                self.api_key = os.getenv("OPENAI_API_KEY")

        # 设置默认 base_url
        if self.base_url is None and self.provider == LLMProvider.OPENAI:
            self.base_url = os.getenv("OPENAI_BASE_URL")

        # 标准化模型名称 (仅对预定义模型映射)
        if self.provider == LLMProvider.ANTHROPIC:
            self.model = self.CLAUDE_MODELS.get(self.model, self.model)
        # OpenAI 兼容接口不映射模型名称，直接使用传入的值


class LLMClient:
    """
    多模型 LLM 客户端

    支持异步调用，自动重试，成本追踪。

    Example:
        client = LLMClient(config=LLMConfig(provider="anthropic"))

        # 生成因子思路
        idea = await client.generate_idea("生成一个动量因子")

        # 生成代码
        code = await client.generate_code(idea)

        # 修正代码
        fixed_code = await client.fix_code(code, error_message)
    """

    # 因子生成系统提示
    FACTOR_SYSTEM_PROMPT = """你是一个专业的量化因子研究员。你的任务是设计新的金融因子。

你需要遵循以下原则：
1. 因子必须基于可观察的市场数据（价格、成交量、财务指标等）
2. 因子逻辑要有经济学含义
3. 考虑数据的可获得性和时效性
4. 注意处理缺失值和异常值
5. 避免未来函数和数据泄露

你的输出格式：
- 因子名称
- 因子分类（必须使用以下之一：MOMENTUM/VALUE/QUALITY/GROWTH/VOLATILITY/LIQUIDITY/SENTIMENT/TECHNICAL/FUNDAMENTAL/ALTERNATIVE/CUSTOM）
- 因子描述（100字以内）
- 数学公式
- Python 实现代码"""

    CODE_SYSTEM_PROMPT = """你是一个专业的量化开发工程师。根据因子思路生成 Python 代码。

【重要】执行环境已预加载以下模块，无需 import：
- 数据分析：pd (pandas), np (numpy)
- 科学计算：linregress (线性回归，来自 scipy.stats)
- 类型注解：Dict, Any, List, Optional, Tuple, Union, Callable
- 项目基类：BaseFactor, FactorMetadata, FactorCategory, DataFrequency
- 数据工具：FactorData, get_close, get_open, get_high, get_low, get_volume, get_amount
- 计算工具：pct_change, rolling_mean, rolling_std, rolling_sum, shift, cross_rank, cross_zscore, where

【代码规范 - 必须遵守】
1. **禁止 import 语句** - 所有模块已预加载
2. 继承 BaseFactor 类，在 __init__ 中创建 FactorMetadata 并传给 super().__init__()
3. 实现 compute() 方法，返回 pd.Series，索引必须为 (date, asset)
4. 实现 validate_inputs() 方法，检查必要列是否存在

【数据格式 - 关键】
- 输入 data: MultiIndex DataFrame，索引为 (date, asset)
- 可用字段：open, high, low, close, volume, amount, vwap
- 输出: pd.Series，索引必须与输入 data 相同 (date, asset)

【推荐使用工具函数 - 简化 MultiIndex 操作】
✅ 推荐方式（使用工具函数）：
- 获取价格：close = get_close(data)  或  close = data['close']
- 计算收益率：returns = pct_change(close, 20)  # 20日收益率
- 滚动均值：ma20 = rolling_mean(close, 20)
- 滚动标准差：std20 = rolling_std(close, 20)
- 平移数据：lag_close = shift(close, 1)  # 前一日收盘价
- 截面排名：rank = cross_rank(factor_values)  # 每日对资产排名
- 截面标准化：zscore = cross_zscore(factor_values)  # 每日 Z-score
- 条件选择：mf_pos = where(tp > tp_lag, mf, 0)  # 保持 MultiIndex 结构
- 线性回归：slope, intercept, r_value, p_value, std_err = linregress(x, y)

✅ 也可以使用 FactorData 类方法：
- FactorData.get_column(data, 'close')
- FactorData.pct_change_by_asset(series, periods)
- FactorData.rolling_by_asset(series, window, 'mean')
- FactorData.shift_by_asset(series, periods)
- FactorData.cross_sectional_rank(series)
- FactorData.cross_sectional_zscore(series)

❌ 避免的错误操作：
- np.where(condition, x, y)  # 返回 numpy array，丢失 MultiIndex！使用 where() 函数代替
- returns.loc[pd.IndexSlice[:, asset], :]  # 会返回 DataFrame，导致索引错误
- series.groupby(level='asset').apply(...)  # 可能改变索引结构

【重要：频率调整由回测引擎处理】
- 因子 compute() 方法应返回日频数据
- 不要在因子代码中手动降频（如 resample、weekly sampling）
- 回测引擎会根据持有期参数自动处理调仓频率

【示例代码 - 简单动量因子】
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
        close = get_close(data)
        return pct_change(close, 20)

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns

【示例代码 - 波动率因子】
class VolatilityFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="volatility_20d",
            category=FactorCategory.VOLATILITY,
            description="20日波动率因子",
            formula="returns.rolling(20).std()",
            lookback_period=21,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = get_close(data)
        returns = pct_change(close, 1)
        return rolling_std(returns, 20)

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns

【示例代码 - 截面标准化因子】
class CrossSectionalMomentum(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="cs_momentum_20d",
            category=FactorCategory.MOMENTUM,
            description="截面标准化动量因子",
            formula="zscore(pct_change(close, 20))",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = get_close(data)
        momentum = pct_change(close, 20)
        return cross_zscore(momentum)

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns

【示例代码 - 复杂因子（需要遍历资产）】
class ComplexFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="complex_factor",
            category=FactorCategory.MOMENTUM,
            description="复杂因子示例",
            formula="...",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        close = get_close(data)
        volume = get_volume(data)

        results = []
        for asset in close.index.get_level_values('asset').unique():
            # 使用 xs 获取单个资产的 Series
            asset_close = close.xs(asset, level='asset')
            asset_volume = volume.xs(asset, level='asset')

            # 计算因子值
            factor_values = self._calc_factor(asset_close, asset_volume)
            factor_values.name = asset
            results.append(factor_values)

        # 合并结果，确保索引为 (date, asset)
        result = pd.concat(results, axis=1).stack()
        result.index.names = ['date', 'asset']
        return result

    def _calc_factor(self, close, volume):
        # 具体计算逻辑
        returns = close.pct_change()
        return returns.rolling(20).mean()

    def validate_inputs(self, data: pd.DataFrame) -> bool:
        return 'close' in data.columns and 'volume' in data.columns
"""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化 LLM 客户端

        Args:
            config: LLM 配置
        """
        self.config = config or LLMConfig()
        self._client = None
        self._initialized = False

        # 统计
        self.total_tokens = 0
        self.total_cost = 0.0

        logger.info(f"LLMClient initialized, provider={self.config.provider}")

    def _init_client(self):
        """初始化底层客户端"""
        if self._initialized:
            return

        if self.config.provider == LLMProvider.ANTHROPIC:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        else:
            try:
                from openai import AsyncOpenAI
                # 支持自定义 base_url (OpenAI 兼容接口)
                client_kwargs = {
                    "api_key": self.config.api_key,
                    "timeout": self.config.timeout,
                    "max_retries": 2,
                }
                if self.config.base_url:
                    client_kwargs["base_url"] = self.config.base_url
                    logger.info(f"Using custom OpenAI-compatible endpoint: {self.config.base_url}")
                self._client = AsyncOpenAI(**client_kwargs)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")

        self._initialized = True
        logger.info(f"Initialized {self.config.provider} client with model: {self.config.model}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            system: 系统提示
            temperature: 温度参数
            max_tokens: 最大 token

        Returns:
            模型响应文本
        """
        self._init_client()

        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        system = system or self.config.system_prompt

        if self.config.provider == LLMProvider.ANTHROPIC:
            response = await self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            content = response.content[0].text
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens

        else:  # OpenAI
            full_messages = []
            if system:
                full_messages.append({"role": "system", "content": system})
            full_messages.extend(messages)

            response = await self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=full_messages,
            )
            content = response.choices[0].message.content
            self.total_tokens += response.usage.total_tokens

        return content

    async def generate_idea(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成因子思路

        Args:
            prompt: 用户提示
            context: 上下文信息

        Returns:
            因子思路描述
        """
        messages = [{"role": "user", "content": prompt}]

        if context:
            context_str = f"\n\n上下文信息：\n{json.dumps(context, ensure_ascii=False, indent=2)}"
            messages[0]["content"] += context_str

        idea = await self.chat(
            messages=messages,
            system=self.FACTOR_SYSTEM_PROMPT,
        )

        logger.info(f"Generated factor idea ({len(idea)} chars)")
        return idea

    async def generate_code(
        self,
        idea: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        根据思路生成代码

        Args:
            idea: 因子思路
            context: 上下文信息

        Returns:
            Python 代码
        """
        prompt = f"""根据以下因子思路生成 Python 代码：

{idea}

请生成完整的 BaseFactor 子类代码。"""

        if context:
            prompt += f"\n\n上下文：\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        messages = [{"role": "user", "content": prompt}]

        code = await self.chat(
            messages=messages,
            system=self.CODE_SYSTEM_PROMPT,
            temperature=0.3,  # 代码生成使用较低温度
        )

        # 提取代码块
        code = self._extract_code(code)

        # 自动修正代码（新增）
        from .code_fixer import CodeFixer
        fixer = CodeFixer()
        code, fixes = fixer.fix(code)
        if fixes:
            logger.info(f"Code fixes applied: {fixes}")

        logger.info(f"Generated factor code ({len(code)} chars)")
        return code

    async def fix_code(
        self,
        code: str,
        error: str,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        修正代码错误

        Args:
            code: 原始代码
            error: 错误信息
            validation_result: 验证结果

        Returns:
            修正后的代码
        """
        prompt = f"""以下代码存在错误，请修正：

原始代码：
```python
{code}
```

错误信息：
{error}
"""

        if validation_result:
            prompt += f"\n验证结果：\n{json.dumps(validation_result, ensure_ascii=False, indent=2)}"

        messages = [{"role": "user", "content": prompt}]

        fixed_code = await self.chat(
            messages=messages,
            system=self.CODE_SYSTEM_PROMPT,
            temperature=0.3,
        )

        fixed_code = self._extract_code(fixed_code)

        logger.info(f"Fixed code ({len(fixed_code)} chars)")
        return fixed_code

    def _extract_code(self, text: str) -> str:
        """从响应中提取代码块"""
        import re

        # 匹配 ```python ... ``` 或 ``` ... ```
        pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # 如果没有代码块，返回原文本
        return text.strip()

    async def batch_generate(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
    ) -> List[str]:
        """
        批量生成

        Args:
            prompts: 提示列表
            temperature: 温度参数

        Returns:
            响应列表
        """
        tasks = [
            self.chat([{"role": "user", "content": p}], temperature=temperature)
            for p in prompts
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    def get_usage_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }
