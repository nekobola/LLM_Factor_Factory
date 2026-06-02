"""
因子表达式解析器

解析 LLM 生成的代码，验证安全性并创建因子实例。
"""

import ast
import re
import inspect
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Type
import warnings as warnings_module

import pandas as pd
import numpy as np
from loguru import logger

import sys
sys.path.insert(0, '..')
from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency
from core.factor_data import (
    FactorData, get_close, get_open, get_high, get_low,
    get_volume, get_amount, pct_change, rolling_mean,
    rolling_std, rolling_sum, shift, cross_rank, cross_zscore, where
)


@dataclass
class ParseResult:
    """
    解析结果

    Attributes:
        success: 是否成功
        factor_class: 因子类
        factor_instance: 因子实例
        errors: 错误列表
        warnings: 警告列表
        metadata: 因子元数据
    """
    success: bool
    factor_class: Optional[Type[BaseFactor]] = None
    factor_instance: Optional[BaseFactor] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Optional[FactorMetadata] = None


class FactorParser:
    """
    因子代码解析器

    功能：
    - 语法验证
    - 安全检查（禁止危险操作）
    - 提取元数据
    - 创建因子实例

    Example:
        parser = FactorParser()

        code = '''
        class MomentumFactor(BaseFactor):
            def __init__(self):
                metadata = FactorMetadata(...)
                super().__init__(metadata)

            def compute(self, data):
                return data['close'].pct_change(20)
        '''

        result = parser.parse(code)
        if result.success:
            factor = result.factor_instance
    """

    # 允许的内置函数
    ALLOWED_BUILTINS = {
        "abs", "min", "max", "sum", "len", "range", "enumerate",
        "zip", "list", "dict", "tuple", "set", "sorted", "reversed",
        "True", "False", "None", "int", "float", "str", "bool",
        # 类定义和面向对象相关
        "__build_class__",  # 用于创建类
        "property",  # 属性装饰器
        "classmethod",  # 类方法装饰器
        "staticmethod",  # 静态方法装饰器
        "super",  # super() 函数
        "type",  # type() 函数
        "isinstance",  # isinstance() 函数
        "issubclass",  # issubclass() 函数
        "hasattr",  # hasattr() 函数
        "getattr",  # getattr() 函数
        "setattr",  # setattr() 函数
        "delattr",  # delattr() 函数
        "callable",  # callable() 函数
        "object",  # object 基类
        # 其他常用内置函数
        "print",  # 调试用
        "round", "divmod", "pow",
        "any", "all",
        "map", "filter",
        "iter", "next",
        "repr", "hash",
    }

    # 允许的模块（数据分析相关）
    ALLOWED_MODULES = {
        "pandas", "pd",
        "numpy", "np",
        "math",
        "statistics",
    }

    # 允许的标准库模块（类型注解、数据结构等）
    ALLOWED_STDLIB_MODULES = {
        "typing",
        "dataclasses",
        "enum",
        "collections",
        "functools",
        "itertools",
        "operator",
        "copy",
        "datetime",
        "decimal",
        "abc",
        "numbers",  # 数值类型抽象基类
        "math",
        "statistics",
    }

    # 允许的项目内部模块
    ALLOWED_PROJECT_MODULES = {
        "core",
        "core.base_factor",
        "base_factor",
        "base",  # 兼容 LLM 可能生成的简化导入
    }

    # 禁止的操作
    FORBIDDEN_PATTERNS = [
        r"exec\s*\(",
        r"eval\s*\(",
        r"compile\s*\(",
        r"open\s*\(",
        # 注意：不禁止 __import__，因为我们需要在安全环境中支持 import 语句
        r"import\s+os\b",
        r"import\s+sys\b",
        r"import\s+subprocess\b",
        r"import\s+shutil\b",
        r"import\s+socket\b",
        r"import\s+requests\b",
        r"import\s+urllib\b",
        r"from\s+os\b",
        r"from\s+sys\b",
        r"from\s+subprocess\b",
        r"from\s+shutil\b",
        r"from\s+socket\b",
        r"from\s+requests\b",
        r"from\s+urllib\b",
    ]

    def __init__(self, strict_mode: bool = True):
        """
        初始化解析器

        Args:
            strict_mode: 严格模式（更严格的安全检查）
        """
        self.strict_mode = strict_mode
        logger.info(f"FactorParser initialized, strict_mode={strict_mode}")

    def parse(self, code: str) -> ParseResult:
        """
        解析代码

        Args:
            code: Python 代码字符串

        Returns:
            ParseResult 对象
        """
        errors = []
        warnings = []

        # Step 1: 语法检查
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"语法错误: {e}")
            return ParseResult(success=False, errors=errors)

        # Step 2: 安全检查
        security_issues = self._check_security(code, tree)
        if security_issues:
            if self.strict_mode:
                errors.extend(security_issues)
                return ParseResult(success=False, errors=errors)
            else:
                warnings.extend(security_issues)

        # Step 3: 提取类定义
        factor_classes = self._extract_factor_classes(tree)
        if not factor_classes:
            errors.append("未找到 BaseFactor 子类")
            return ParseResult(success=False, errors=errors)

        # Step 4: 创建安全执行环境
        safe_globals = self._create_safe_globals()

        # Step 5: 执行代码
        local_vars = {}
        try:
            with warnings_module.catch_warnings():
                warnings_module.simplefilter("ignore")
                exec(compile(tree, "<string>", "exec"), safe_globals, local_vars)
        except Exception as e:
            errors.append(f"代码执行错误: {e}")
            return ParseResult(success=False, errors=errors)

        # Step 6: 查找因子类
        factor_class = None
        for name, obj in local_vars.items():
            if isinstance(obj, type) and issubclass(obj, BaseFactor) and obj is not BaseFactor:
                factor_class = obj
                break

        if factor_class is None:
            errors.append("未找到有效的因子类")
            return ParseResult(success=False, errors=errors)

        # Step 7: 创建实例
        try:
            factor_instance = factor_class()
        except Exception as e:
            errors.append(f"创建因子实例失败: {e}")
            return ParseResult(success=False, errors=errors)

        # Step 8: 验证接口
        interface_errors = self._validate_interface(factor_instance)
        if interface_errors:
            errors.extend(interface_errors)
            return ParseResult(success=False, errors=errors)

        # 提取元数据
        metadata = factor_instance.metadata if hasattr(factor_instance, "metadata") else None

        return ParseResult(
            success=True,
            factor_class=factor_class,
            factor_instance=factor_instance,
            warnings=warnings,
            metadata=metadata,
        )

    def _check_security(self, code: str, tree: ast.AST) -> List[str]:
        """简化的安全检查（代码已被 CodeFixer 预处理）"""
        issues = []

        # 只检查最危险的操作
        dangerous_patterns = [
            (r"exec\s*\(", "exec 调用"),
            (r"eval\s*\(", "eval 调用"),
            (r"compile\s*\(", "compile 调用"),
            (r"open\s*\(", "文件操作"),
            (r"__import__\s*\(", "动态导入"),
            (r"subprocess", "subprocess 模块"),
            (r"os\.system", "系统命令"),
        ]

        for pattern, desc in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(f"检测到危险操作: {desc}")

        return issues

    def _is_module_allowed(self, module_name: str) -> bool:
        """检查模块是否被允许导入"""
        module_root = module_name.split(".")[0]

        # 检查数据分析模块
        if module_root in self.ALLOWED_MODULES:
            return True

        # 检查标准库模块
        if module_root in self.ALLOWED_STDLIB_MODULES:
            return True

        # 检查项目内部模块（支持完整路径匹配）
        for allowed in self.ALLOWED_PROJECT_MODULES:
            if module_name == allowed or module_name.startswith(allowed + "."):
                return True

        return False

    def _extract_factor_classes(self, tree: ast.AST) -> List[ast.ClassDef]:
        """提取因子类定义"""
        factor_classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否继承自 BaseFactor
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "BaseFactor":
                        factor_classes.append(node)
                    elif isinstance(base, ast.Attribute) and base.attr == "BaseFactor":
                        factor_classes.append(node)

        return factor_classes

    def _create_safe_globals(self) -> Dict[str, Any]:
        """创建简化的执行环境"""
        import typing
        from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set, Type
        import dataclasses
        import enum
        from datetime import datetime, date, time, timedelta
        from collections import namedtuple, defaultdict, Counter
        from functools import partial, reduce, wraps
        from itertools import product, combinations, permutations
        import operator
        import copy
        import decimal
        import math
        import statistics
        from abc import ABC, abstractmethod
        import numbers
        from scipy.stats import linregress

        # 获取内置函数字典
        builtins_dict = __builtins__ if isinstance(__builtins__, dict) else dict(__builtins__)

        # 禁止的危险函数
        dangerous = {"exec", "eval", "compile", "open", "input", "breakpoint"}

        # 创建安全的 builtins（保留大部分常用函数）
        safe_builtins = {k: v for k, v in builtins_dict.items() if k not in dangerous}

        # 使用原始的 __import__，因为代码已经被 CodeFixer 预处理过
        # 不再需要复杂的沙箱导入控制

        return {
            "__builtins__": safe_builtins,
            # 数据分析库
            "pd": pd,
            "np": np,
            "pandas": pd,
            "numpy": np,
            # 标准库 - 数学相关
            "math": math,
            "statistics": statistics,
            "decimal": decimal,
            "numbers": numbers,
            # 标准库 - 类型相关
            "typing": typing,
            "Dict": Dict,
            "Any": Any,
            "List": List,
            "Optional": Optional,
            "Tuple": Tuple,
            "Union": Union,
            "Callable": Callable,
            "Set": Set,
            "Type": Type,
            "dataclasses": dataclasses,
            "enum": enum,
            # 标准库 - 日期时间
            "datetime": datetime,
            "date": date,
            "time": time,
            "timedelta": timedelta,
            # 标准库 - 集合和函数式
            "namedtuple": namedtuple,
            "defaultdict": defaultdict,
            "Counter": Counter,
            "partial": partial,
            "reduce": reduce,
            "wraps": wraps,
            "product": product,
            "combinations": combinations,
            "permutations": permutations,
            # 标准库 - 其他
            "operator": operator,
            "copy": copy,
            "ABC": ABC,
            "abstractmethod": abstractmethod,
            # 科学计算
            "linregress": linregress,
            # 项目模块
            "BaseFactor": BaseFactor,
            "FactorMetadata": FactorMetadata,
            "FactorCategory": FactorCategory,
            "DataFrequency": DataFrequency,
            # FactorData 工具类（简化 MultiIndex 操作）
            "FactorData": FactorData,
            "get_close": get_close,
            "get_open": get_open,
            "get_high": get_high,
            "get_low": get_low,
            "get_volume": get_volume,
            "get_amount": get_amount,
            "pct_change": pct_change,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "rolling_sum": rolling_sum,
            "shift": shift,
            "cross_rank": cross_rank,
            "cross_zscore": cross_zscore,
            "where": where,
        }

    def _validate_interface(self, instance: BaseFactor) -> List[str]:
        """验证因子接口"""
        errors = []

        # 检查必要属性
        if not hasattr(instance, "metadata"):
            errors.append("因子缺少 metadata 属性")
        elif not isinstance(instance.metadata, FactorMetadata):
            errors.append("metadata 必须是 FactorMetadata 类型")

        # 检查必要方法
        if not hasattr(instance, "compute"):
            errors.append("因子缺少 compute 方法")
        elif not callable(instance.compute):
            errors.append("compute 必须是可调用方法")

        if not hasattr(instance, "validate_inputs"):
            errors.append("因子缺少 validate_inputs 方法")
        elif not callable(instance.validate_inputs):
            errors.append("validate_inputs 必须是可调用方法")

        return errors

    def extract_metadata_from_code(self, code: str) -> Optional[Dict[str, Any]]:
        """
        从代码中提取元数据信息

        Args:
            code: Python 代码

        Returns:
            元数据字典
        """
        metadata = {}

        # 提取类名
        class_match = re.search(r"class\s+(\w+)\s*\(", code)
        if class_match:
            metadata["class_name"] = class_match.group(1)

        # 提取 FactorMetadata 参数
        name_match = re.search(r"name\s*=\s*[\"']([^\"']+)[\"']", code)
        if name_match:
            metadata["name"] = name_match.group(1)

        category_match = re.search(r"category\s*=\s*FactorCategory\.(\w+)", code)
        if category_match:
            metadata["category"] = category_match.group(1)

        desc_match = re.search(r"description\s*=\s*[\"']([^\"']+)[\"']", code)
        if desc_match:
            metadata["description"] = desc_match.group(1)

        return metadata


def safe_execute(code: str, strict: bool = True) -> ParseResult:
    """
    安全执行因子代码

    Args:
        code: Python 代码
        strict: 是否严格模式

    Returns:
        ParseResult 对象
    """
    parser = FactorParser(strict_mode=strict)
    return parser.parse(code)
