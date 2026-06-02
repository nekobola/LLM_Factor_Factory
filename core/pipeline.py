"""
因子生成与验证工作流 Pipeline

LLM 驱动的自动化因子挖掘流程：
1. LLM 生成因子思路
2. 解析为可执行代码
3. 自动运行验证
4. 若失败，LLM 修正逻辑
5. 若通过，注册入库
"""

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
import time
import json

from loguru import logger

from .base_factor import BaseFactor, FactorMetadata, FactorResult, ValidationResult
from .registry import FactorRegistry


class PipelineStage(str, Enum):
    """Pipeline 阶段"""
    IDEA_GENERATION = "想法生成"
    CODE_GENERATION = "代码生成"
    COMPUTATION = "因子计算"
    VALIDATION = "因子验证"
    FIX_ITERATION = "错误修正"
    REGISTRATION = "注册入库"
    COMPLETED = "完成"
    FAILED = "失败"


class PipelineError(Exception):
    """Pipeline 错误"""
    pass


@dataclass
class PipelineState:
    """Pipeline 状态"""
    stage: PipelineStage
    iteration: int = 0
    max_iterations: int = 5
    start_time: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 中间结果
    idea: Optional[str] = None
    generated_code: Optional[str] = None
    factor: Optional[BaseFactor] = None
    factor_result: Optional[FactorResult] = None
    validation_result: Optional[ValidationResult] = None

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


@dataclass
class PipelineResult:
    """Pipeline 最终结果"""
    success: bool
    factor_name: Optional[str] = None
    factor: Optional[BaseFactor] = None
    validation: Optional[ValidationResult] = None
    elapsed_time: float = 0.0
    iterations: int = 0
    error: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


class FactorPipeline:
    """
    因子生成与验证工作流

    Example:
        from idea_generator import LLMClient
        from validator import FactorValidator
        from data_engine import DataLoader

        pipeline = FactorPipeline(
            llm_client=LLMClient(provider="anthropic"),
            validator=FactorValidator(),
            data_loader=DataLoader(),
        )

        result = await pipeline.run("生成一个基于成交量的动量因子")
        if result.success:
            print(f"因子 {result.factor_name} 验证通过!")
    """

    def __init__(
        self,
        llm_client: "LLMClient",
        validator: "FactorValidator",
        data_loader: "DataLoader",
        registry: Optional[FactorRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 Pipeline

        Args:
            llm_client: LLM 客户端
            validator: 因子验证器
            data_loader: 数据加载器
            registry: 因子注册中心
            config: 配置参数
        """
        self.llm = llm_client
        self.validator = validator
        self.data = data_loader
        self.registry = registry or FactorRegistry()
        self.config = config or {}

        # 回调钩子
        self._hooks: Dict[str, List[Callable]] = {
            "on_idea_generated": [],
            "on_code_generated": [],
            "on_computation_done": [],
            "on_validation_done": [],
            "on_factor_registered": [],
            "on_error": [],
        }

        logger.info("FactorPipeline initialized")

    def add_hook(self, event: str, callback: Callable):
        """添加事件回调钩子"""
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def _trigger_hooks(self, event: str, **kwargs):
        """触发事件钩子"""
        for callback in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(**kwargs)
                else:
                    callback(**kwargs)
            except Exception as e:
                logger.warning(f"Hook callback error: {e}")

    async def run(
        self,
        prompt: str,
        max_iterations: int = 5,
        auto_fix: bool = True,
        skip_validation: bool = False,
    ) -> PipelineResult:
        """
        运行完整的因子生成流程

        Args:
            prompt: 因子生成提示（自然语言描述）
            max_iterations: 最大迭代次数
            auto_fix: 是否自动修正错误
            skip_validation: 是否跳过验证

        Returns:
            PipelineResult 对象
        """
        state = PipelineState(
            stage=PipelineStage.IDEA_GENERATION,
            max_iterations=max_iterations,
        )

        history = []

        try:
            while state.iteration < max_iterations:
                state.iteration += 1
                logger.info(f"=== Iteration {state.iteration}/{max_iterations} ===")

                # Stage 1: 生成因子思路
                state.stage = PipelineStage.IDEA_GENERATION
                idea = await self._generate_idea(prompt, state)
                state.idea = idea
                await self._trigger_hooks("on_idea_generated", idea=idea, state=state)

                # Stage 2: 生成代码
                state.stage = PipelineStage.CODE_GENERATION
                code = await self._generate_code(idea, state)
                state.generated_code = code
                await self._trigger_hooks("on_code_generated", code=code, state=state)

                # Stage 3: 计算因子
                state.stage = PipelineStage.COMPUTATION
                try:
                    factor, factor_result = await self._compute_factor(code, state)
                    state.factor = factor
                    state.factor_result = factor_result
                    await self._trigger_hooks("on_computation_done", result=factor_result, state=state)
                except Exception as e:
                    error_msg = f"因子计算失败: {e}\n{traceback.format_exc()}"
                    state.errors.append(error_msg)
                    history.append({
                        "iteration": state.iteration,
                        "stage": "computation",
                        "error": error_msg,
                    })

                    if auto_fix:
                        prompt = self._build_fix_prompt(error_msg, code)
                        continue
                    else:
                        raise PipelineError(error_msg)

                # Stage 4: 验证因子
                if not skip_validation:
                    state.stage = PipelineStage.VALIDATION
                    try:
                        validation = await self._validate_factor(factor, factor_result, state)
                        state.validation_result = validation
                        await self._trigger_hooks("on_validation_done", validation=validation, state=state)

                        if validation.passed:
                            # 验证通过，注册入库
                            state.stage = PipelineStage.REGISTRATION
                            await self._register_factor(factor, validation, code, state)
                            state.stage = PipelineStage.COMPLETED

                            return PipelineResult(
                                success=True,
                                factor_name=factor.metadata.name,
                                factor=factor,
                                validation=validation,
                                elapsed_time=state.elapsed_time,
                                iterations=state.iteration,
                                history=history,
                            )
                        else:
                            # 验证未通过
                            error_msg = f"验证未通过: {validation.errors}"
                            state.errors.append(error_msg)
                            history.append({
                                "iteration": state.iteration,
                                "stage": "validation",
                                "validation": validation.__dict__,
                                "error": error_msg,
                            })

                            if auto_fix:
                                prompt = self._build_validation_fix_prompt(validation, code)
                                continue
                            else:
                                raise PipelineError(error_msg)

                    except Exception as e:
                        error_msg = f"验证过程失败: {e}\n{traceback.format_exc()}"
                        state.errors.append(error_msg)
                        history.append({
                            "iteration": state.iteration,
                            "stage": "validation",
                            "error": error_msg,
                        })

                        if auto_fix:
                            prompt = self._build_fix_prompt(error_msg, code)
                            continue
                        else:
                            raise PipelineError(error_msg)
                else:
                    # 跳过验证，直接注册
                    state.stage = PipelineStage.REGISTRATION
                    await self._register_factor(factor, None, code, state)
                    state.stage = PipelineStage.COMPLETED

                    return PipelineResult(
                        success=True,
                        factor_name=factor.metadata.name,
                        factor=factor,
                        elapsed_time=state.elapsed_time,
                        iterations=state.iteration,
                        history=history,
                    )

            # 达到最大迭代次数
            state.stage = PipelineStage.FAILED
            error_msg = f"达到最大迭代次数 ({max_iterations})，仍未生成有效因子"
            return PipelineResult(
                success=False,
                error=error_msg,
                elapsed_time=state.elapsed_time,
                iterations=state.iteration,
                history=history,
            )

        except Exception as e:
            state.stage = PipelineStage.FAILED
            error_msg = f"Pipeline 失败: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return PipelineResult(
                success=False,
                error=error_msg,
                elapsed_time=state.elapsed_time,
                iterations=state.iteration,
                history=history,
            )

    async def _generate_idea(self, prompt: str, state: PipelineState) -> str:
        """生成因子思路"""
        logger.info(f"Generating idea from prompt: {prompt[:100]}...")

        idea = await self.llm.generate_idea(
            prompt=prompt,
            context={
                "iteration": state.iteration,
                "previous_errors": state.errors[-3:] if state.errors else [],
            }
        )

        logger.info(f"Generated idea: {idea[:200]}...")
        return idea

    async def _generate_code(self, idea: str, state: PipelineState) -> str:
        """生成因子代码"""
        logger.info("Generating factor code...")

        code = await self.llm.generate_code(
            idea=idea,
            context={
                "iteration": state.iteration,
                "base_class": "BaseFactor",
                "available_fields": ["open", "high", "low", "close", "volume", "amount", "vwap"],
            }
        )

        logger.info(f"Generated code ({len(code)} chars)")
        return code

    async def _compute_factor(
        self,
        code: str,
        state: PipelineState,
    ) -> tuple[BaseFactor, FactorResult]:
        """计算因子值"""
        logger.info("Computing factor values...")

        # 解析并执行代码
        factor = await self._parse_and_create_factor(code)

        # 加载数据
        data = await self.data.load(
            lookback_period=factor.metadata.lookback_period,
            frequency=factor.metadata.frequency,
        )

        # 计算因子
        result = factor(data)

        logger.info(f"Computed factor values: shape={result.values.shape}, quality={result.data_quality_score:.2f}")
        return factor, result

    async def _parse_and_create_factor(self, code: str) -> BaseFactor:
        """
        解析代码并创建因子实例

        使用安全沙箱执行代码
        """
        # 创建安全的执行命名空间
        safe_globals = {
            "__builtins__": {
                "abs": abs, "min": min, "max": max, "sum": sum,
                "len": len, "range": range, "enumerate": enumerate,
                "zip": zip, "list": list, "dict": dict, "tuple": tuple,
                "True": True, "False": False, "None": None,
            },
            "pd": __import__("pandas"),
            "np": __import__("numpy"),
            "BaseFactor": BaseFactor,
            "FactorMetadata": FactorMetadata,
            "FactorCategory": __import__("core.base_factor", fromlist=["FactorCategory"]).FactorCategory,
            "DataFrequency": __import__("core.base_factor", fromlist=["DataFrequency"]).DataFrequency,
        }

        # 执行代码
        local_vars = {}
        try:
            exec(code, safe_globals, local_vars)
        except Exception as e:
            raise PipelineError(f"代码执行错误: {e}")

        # 查找 BaseFactor 子类
        factor_class = None
        for name, obj in local_vars.items():
            if isinstance(obj, type) and issubclass(obj, BaseFactor) and obj is not BaseFactor:
                factor_class = obj
                break

        if factor_class is None:
            raise PipelineError("代码中未找到 BaseFactor 子类")

        # 创建实例
        factor_instance = factor_class()
        return factor_instance

    async def _validate_factor(
        self,
        factor: BaseFactor,
        factor_result: FactorResult,
        state: PipelineState,
    ) -> ValidationResult:
        """验证因子"""
        logger.info("Validating factor...")

        # 加载收益率数据
        returns = await self.data.load_returns()

        # 运行验证
        validation = await self.validator.validate(
            factor_values=factor_result.values,
            returns=returns,
            factor_metadata=factor.metadata,
        )

        logger.info(f"Validation result: passed={validation.passed}, IC={validation.ic_mean:.4f}, IR={validation.ic_ir:.4f}")
        return validation

    async def _register_factor(
        self,
        factor: BaseFactor,
        validation: Optional[ValidationResult],
        code: str,
        state: PipelineState,
    ):
        """注册因子"""
        logger.info(f"Registering factor: {factor.metadata.name}")

        factor_id = self.registry.register(
            factor=factor,
            validation_result=validation,
            code=code,
        )

        await self._trigger_hooks(
            "on_factor_registered",
            factor_id=factor_id,
            factor=factor,
            validation=validation,
        )

        logger.info(f"Factor registered with id={factor_id}")

    def _build_fix_prompt(self, error: str, code: str) -> str:
        """构建错误修正提示"""
        return f"""
上一个因子的代码执行失败，请根据错误信息修正代码。

错误信息:
{error}

原始代码:
```python
{code}
```

请分析错误原因并生成修正后的代码。确保:
1. 处理好缺失值和异常情况
2. 遵循 BaseFactor 接口规范
3. 返回正确格式的因子值
"""

    def _build_validation_fix_prompt(self, validation: ValidationResult, code: str) -> str:
        """构建验证修正提示"""
        errors_str = "\n".join(validation.errors) if validation.errors else "无具体错误"

        return f"""
因子的验证结果不理想，请尝试优化代码以提高验证指标。

当前验证结果:
- IC 均值: {validation.ic_mean:.4f} (目标 > 0.02)
- IC 信息比率: {validation.ic_ir:.4f} (目标 > 0.5)
- 单调性得分: {validation.monotonicity_score:.4f} (目标 > 0.8)
- 正交化保留率: {validation.ortho_ic_retention:.4f} (目标 > 0.5)

问题:
{errors_str}

原始代码:
```python
{code}
```

请分析原因并尝试以下优化策略:
1. 调整计算窗口期
2. 增加数据预处理（去极值、平滑）
3. 修改因子逻辑
4. 考虑与其他因子的交互
"""

    async def run_batch(
        self,
        prompts: List[str],
        max_iterations: int = 5,
        parallel: bool = False,
    ) -> List[PipelineResult]:
        """
        批量运行多个因子生成任务

        Args:
            prompts: 提示列表
            max_iterations: 每个任务的最大迭代次数
            parallel: 是否并行执行

        Returns:
            结果列表
        """
        if parallel:
            tasks = [
                self.run(prompt, max_iterations=max_iterations)
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [
                r if isinstance(r, PipelineResult) else PipelineResult(success=False, error=str(r))
                for r in results
            ]
        else:
            results = []
            for prompt in prompts:
                result = await self.run(prompt, max_iterations=max_iterations)
                results.append(result)
            return results


# 延迟导入的类型提示
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from idea_generator.llm_client import LLMClient
    from validator import FactorValidator
    from data_engine import DataLoader
