"""
错误处理分层机制

实现分层错误处理：
1. 自动修复层 - 处理简单问题（import、语法小问题）
2. 验证层 - 检查代码结构和接口
3. 执行层 - 捕获运行时错误
4. 用户报告层 - 清晰的错误信息
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import re
import traceback

from loguru import logger


class ErrorSeverity(str, Enum):
    """错误严重程度"""
    AUTO_FIXABLE = "auto_fixable"      # 可自动修复
    WARNING = "warning"                 # 警告，可继续
    ERROR = "error"                     # 错误，需用户介入
    CRITICAL = "critical"               # 严重错误，无法继续


class ErrorCategory(str, Enum):
    """错误类别"""
    SYNTAX = "syntax"                   # 语法错误
    IMPORT = "import"                   # 导入错误
    INTERFACE = "interface"             # 接口错误
    DATA_FORMAT = "data_format"         # 数据格式错误
    RUNTIME = "runtime"                 # 运行时错误
    VALIDATION = "validation"           # 验证错误
    UNKNOWN = "unknown"                 # 未知错误


@dataclass
class FactorError:
    """
    因子错误信息

    Attributes:
        severity: 严重程度
        category: 错误类别
        message: 错误消息
        suggestion: 修复建议
        auto_fix: 是否可自动修复
        fixed: 是否已修复
        original_code: 原始代码
        fixed_code: 修复后代码
    """
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    suggestion: Optional[str] = None
    auto_fix: bool = False
    fixed: bool = False
    original_code: Optional[str] = None
    fixed_code: Optional[str] = None
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "auto_fix": self.auto_fix,
            "fixed": self.fixed,
        }


@dataclass
class ErrorReport:
    """
    错误报告

    包含多个错误和修复结果。
    """
    errors: List[FactorError] = field(default_factory=list)
    fixed_count: int = 0
    remaining_errors: int = 0

    def add_error(self, error: FactorError):
        self.errors.append(error)

    def has_critical_errors(self) -> bool:
        return any(e.severity == ErrorSeverity.CRITICAL for e in self.errors)

    def has_fixable_errors(self) -> bool:
        return any(e.auto_fix and not e.fixed for e in self.errors)

    def get_user_message(self) -> str:
        """生成用户友好的错误消息"""
        if not self.errors:
            return "无错误"

        # 按严重程度分组
        critical = [e for e in self.errors if e.severity == ErrorSeverity.CRITICAL]
        errors = [e for e in self.errors if e.severity == ErrorSeverity.ERROR]
        warnings = [e for e in self.errors if e.severity == ErrorSeverity.WARNING]
        fixed = [e for e in self.errors if e.fixed]

        lines = []

        if fixed:
            lines.append(f"[OK] 已自动修复 {len(fixed)} 个问题")

        if critical:
            lines.append(f"[ERROR] 发现 {len(critical)} 个严重错误：")
            for e in critical[:3]:  # 最多显示3个
                lines.append(f"   - {e.message}")
                if e.suggestion:
                    lines.append(f"     建议：{e.suggestion}")

        if errors:
            lines.append(f"[WARN] 发现 {len(errors)} 个错误需要处理：")
            for e in errors[:3]:
                lines.append(f"   - {e.message}")
                if e.suggestion:
                    lines.append(f"     建议：{e.suggestion}")

        if warnings:
            lines.append(f"[INFO] {len(warnings)} 个警告（可继续执行）")

        return "\n".join(lines)


# ============================================================
# 自动修复层
# ============================================================

class AutoFixer:
    """
    自动修复器

    处理简单的代码问题。
    """

    def fix(self, code: str) -> Tuple[str, List[FactorError]]:
        """
        自动修复代码

        Args:
            code: 原始代码

        Returns:
            (修复后代码, 修复的错误列表)
        """
        fixes = []
        fixed_code = code

        # 1. 移除不必要的 import
        fixed_code, import_fixes = self._fix_imports(fixed_code)
        fixes.extend(import_fixes)

        # 2. 修复常见语法问题
        fixed_code, syntax_fixes = self._fix_common_syntax(fixed_code)
        fixes.extend(syntax_fixes)

        # 3. 修复缩进问题
        fixed_code, indent_fixes = self._fix_indentation(fixed_code)
        fixes.extend(indent_fixes)

        return fixed_code, fixes

    def _fix_imports(self, code: str) -> Tuple[str, List[FactorError]]:
        """修复 import 语句"""
        fixes = []

        # 移除标准库 import（环境已预加载）
        patterns_to_remove = [
            r'^import pandas as pd\s*$',
            r'^import numpy as np\s*$',
            r'^from typing import.*$',
            r'^import math\s*$',
            r'^from datetime import.*$',
        ]

        lines = code.split('\n')
        new_lines = []

        for line in lines:
            stripped = line.strip()
            removed = False

            for pattern in patterns_to_remove:
                if re.match(pattern, stripped):
                    fixes.append(FactorError(
                        severity=ErrorSeverity.AUTO_FIXABLE,
                        category=ErrorCategory.IMPORT,
                        message=f"移除冗余的 import: {stripped}",
                        auto_fix=True,
                        fixed=True,
                        original_code=stripped,
                        fixed_code="# 已移除（环境已预加载）",
                    ))
                    removed = True
                    break

            if not removed:
                new_lines.append(line)

        return '\n'.join(new_lines), fixes

    def _fix_common_syntax(self, code: str) -> Tuple[str, List[FactorError]]:
        """修复常见语法问题"""
        fixes = []

        # 修复中文注释格式
        # 确保中文注释后有空格
        lines = code.split('\n')
        new_lines = []

        for line in lines:
            new_line = line

            # 修复 #后缺少空格的中文注释
            if '#' in line:
                match = re.search(r'#([^#\s])', line)
                if match and '一' <= match.group(1) <= '鿿':
                    new_line = line.replace('#' + match.group(1), '# ' + match.group(1))
                    if new_line != line:
                        fixes.append(FactorError(
                            severity=ErrorSeverity.AUTO_FIXABLE,
                            category=ErrorCategory.SYNTAX,
                            message="修复注释格式：#后添加空格",
                            auto_fix=True,
                            fixed=True,
                        ))

            new_lines.append(new_line)

        return '\n'.join(new_lines), fixes

    def _fix_indentation(self, code: str) -> Tuple[str, List[FactorError]]:
        """修复缩进问题"""
        fixes = []

        # 检查是否有混用 tab 和空格
        has_tab = '\t' in code
        has_spaces = '    ' in code

        if has_tab and has_spaces:
            # 统一使用空格
            fixed_code = code.replace('\t', '    ')
            fixes.append(FactorError(
                severity=ErrorSeverity.AUTO_FIXABLE,
                category=ErrorCategory.SYNTAX,
                message="修复缩进：统一使用空格",
                auto_fix=True,
                fixed=True,
            ))
            return fixed_code, fixes

        return code, fixes


# ============================================================
# 错误诊断层
# ============================================================

class ErrorDiagnoser:
    """
    错误诊断器

    分析错误并分类。
    """

    def diagnose(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> FactorError:
        """
        诊断错误

        Args:
            error: 异常对象
            context: 上下文信息

        Returns:
            FactorError 对象
        """
        error_str = str(error)
        error_type = type(error).__name__
        tb = traceback.format_exc()

        # 语法错误
        if isinstance(error, SyntaxError):
            return FactorError(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYNTAX,
                message=f"语法错误: {error.msg} (行 {error.lineno})",
                suggestion="检查代码语法，确保括号、引号匹配",
                traceback=tb,
            )

        # 导入错误
        if isinstance(error, (ImportError, ModuleNotFoundError)):
            return FactorError(
                severity=ErrorSeverity.AUTO_FIXABLE,
                category=ErrorCategory.IMPORT,
                message=f"导入错误: {error_str}",
                suggestion="移除 import 语句（环境已预加载必要模块）",
                auto_fix=True,
                traceback=tb,
            )

        # 名称错误（变量未定义）
        if isinstance(error, NameError):
            return FactorError(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.RUNTIME,
                message=f"名称错误: {error_str}",
                suggestion=f"检查变量名是否正确，可能需要导入或定义",
                traceback=tb,
            )

        # 类型错误
        if isinstance(error, TypeError):
            # 检查是否是数据格式问题
            if "MultiIndex" in error_str or "index" in error_str.lower():
                return FactorError(
                    severity=ErrorSeverity.ERROR,
                    category=ErrorCategory.DATA_FORMAT,
                    message=f"数据格式错误: {error_str}",
                    suggestion="确保返回 pd.Series，索引为 (date, asset) MultiIndex",
                    traceback=tb,
                )
            return FactorError(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.RUNTIME,
                message=f"类型错误: {error_str}",
                suggestion="检查数据类型是否正确",
                traceback=tb,
            )

        # 值错误
        if isinstance(error, ValueError):
            if "zero-size" in error_str:
                return FactorError(
                    severity=ErrorSeverity.WARNING,
                    category=ErrorCategory.DATA_FORMAT,
                    message="数据不足：无法进行分组",
                    suggestion="检查因子值是否全为 NaN 或数据量是否足够",
                    traceback=tb,
                )
            return FactorError(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.RUNTIME,
                message=f"值错误: {error_str}",
                suggestion="检查输入值是否在有效范围内",
                traceback=tb,
            )

        # 键错误
        if isinstance(error, KeyError):
            return FactorError(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.DATA_FORMAT,
                message=f"键错误: {error_str}",
                suggestion="检查数据列名是否正确（open, high, low, close, volume）",
                traceback=tb,
            )

        # 其他错误
        return FactorError(
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.UNKNOWN,
            message=f"{error_type}: {error_str}",
            suggestion="查看详细错误信息进行调试",
            traceback=tb,
        )


# ============================================================
# 验证层
# ============================================================

class CodeValidator:
    """
    代码验证器

    验证代码结构和接口。
    """

    def validate(self, code: str) -> List[FactorError]:
        """
        验证代码

        Args:
            code: 因子代码

        Returns:
            错误列表
        """
        errors = []

        # 1. 检查是否包含 BaseFactor 继承
        if "BaseFactor" not in code:
            errors.append(FactorError(
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.INTERFACE,
                message="代码未继承 BaseFactor 类",
                suggestion="确保类定义包含 class YourFactor(BaseFactor):",
            ))

        # 2. 检查是否实现 compute 方法
        if "def compute" not in code:
            errors.append(FactorError(
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.INTERFACE,
                message="未实现 compute() 方法",
                suggestion="必须实现 compute(self, data: pd.DataFrame) -> pd.Series",
            ))

        # 3. 检查是否实现 validate_inputs 方法
        if "def validate_inputs" not in code:
            errors.append(FactorError(
                severity=ErrorSeverity.WARNING,
                category=ErrorCategory.INTERFACE,
                message="未实现 validate_inputs() 方法",
                suggestion="建议实现 validate_inputs(self, data: pd.DataFrame) -> bool",
            ))

        # 4. 检查是否创建 FactorMetadata
        if "FactorMetadata" not in code:
            errors.append(FactorError(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.INTERFACE,
                message="未创建 FactorMetadata",
                suggestion="在 __init__ 中创建 metadata 并传给 super().__init__()",
            ))

        # 5. 检查返回类型注解
        if "pd.Series" not in code:
            errors.append(FactorError(
                severity=ErrorSeverity.WARNING,
                category=ErrorCategory.INTERFACE,
                message="compute() 方法缺少返回类型注解",
                suggestion="添加返回类型注解: -> pd.Series",
            ))

        return errors


# ============================================================
# 主错误处理器
# ============================================================

class ErrorHandler:
    """
    主错误处理器

    整合所有错误处理层。
    """

    def __init__(self):
        self.auto_fixer = AutoFixer()
        self.diagnoser = ErrorDiagnoser()
        self.validator = CodeValidator()

    def process_code(self, code: str) -> Tuple[str, ErrorReport]:
        """
        处理代码

        Args:
            code: 原始代码

        Returns:
            (处理后的代码, 错误报告)
        """
        report = ErrorReport()

        # 1. 自动修复
        fixed_code, fixes = self.auto_fixer.fix(code)
        for fix in fixes:
            report.add_error(fix)
            if fix.fixed:
                report.fixed_count += 1

        # 2. 验证代码结构
        validation_errors = self.validator.validate(fixed_code)
        for error in validation_errors:
            report.add_error(error)
            if error.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
                report.remaining_errors += 1

        return fixed_code, report

    def handle_exception(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> FactorError:
        """
        处理异常

        Args:
            error: 异常对象
            context: 上下文信息

        Returns:
            FactorError 对象
        """
        return self.diagnoser.diagnose(error, context)

    def get_user_report(self, report: ErrorReport) -> str:
        """获取用户报告"""
        return report.get_user_message()


# ============================================================
# 便捷函数
# ============================================================

def safe_execute(func, *args, **kwargs) -> Tuple[Any, Optional[FactorError]]:
    """
    安全执行函数

    Args:
        func: 要执行的函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        (结果, 错误)
    """
    handler = ErrorHandler()

    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error = handler.handle_exception(e)
        return None, error
