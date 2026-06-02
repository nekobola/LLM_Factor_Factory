"""
统计验证层

包含:
- IC/IR 分析
- 分组回测/分层测试
- 正交化检验
- Newey-West 稳健性检验
"""

from .ic_analyzer import ICAnalyzer, ICResult
from .group_test import GroupTester, GroupTestResult
from .orthogonality import OrthogonalityTester, OrthogonalityResult
from .robustness import RobustnessTester, RobustnessResult
from .validator import FactorValidator, ValidationResult

__all__ = [
    "ICAnalyzer", "ICResult",
    "GroupTester", "GroupTestResult",
    "OrthogonalityTester", "OrthogonalityResult",
    "RobustnessTester", "RobustnessResult",
    "FactorValidator", "ValidationResult",
]
