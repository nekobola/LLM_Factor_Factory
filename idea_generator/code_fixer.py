"""
LLM 生成代码的自动修正器

在代码执行前自动修正常见问题，确保代码符合 BaseFactor 接口规范。
"""

import re
from typing import Tuple, List, Optional
from loguru import logger


class CodeFixer:
    """
    LLM 生成代码的自动修正器

    修正内容：
    1. 错误的导入路径（.base, base, base_factor 等）
    2. 缺少 metadata 参数的 super().__init__() 调用
    3. 注释格式问题
    4. 相对导入语句

    Example:
        fixer = CodeFixer()
        fixed_code, fixes = fixer.fix(llm_generated_code)
        print(f"Applied fixes: {fixes}")
    """

    # 常见的导入修正映射
    IMPORT_PATTERNS = [
        # 错误的相对导入
        (r"from\s+\.base\s+import\s+BaseFactor", "from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency"),
        (r"from\s+\.base_factor\s+import\s+BaseFactor", "from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency"),
        # 错误的模块名
        (r"from\s+base\s+import\s+BaseFactor", "from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency"),
        (r"from\s+base_factor\s+import\s+BaseFactor", "from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency"),
        (r"import\s+base_factor", "from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency"),
        # 部分正确但缺少其他导入
        (r"from\s+core\.base_factor\s+import\s+BaseFactor\s*$", "from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency"),
        (r"from\s+core\.base_factor\s+import\s+BaseFactor\s*,\s*FactorMetadata\s*$", "from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency"),
    ]

    # 必需的导入（如果不存在则添加）
    REQUIRED_IMPORT = "from core.base_factor import BaseFactor, FactorMetadata, FactorCategory, DataFrequency"

    # 常用的额外导入
    COMMON_IMPORTS = [
        "import pandas as pd",
        "import numpy as np",
    ]

    def __init__(self):
        self.fixes_applied: List[str] = []

    def fix(self, code: str) -> Tuple[str, List[str]]:
        """
        修正代码

        Args:
            code: LLM 生成的原始代码

        Returns:
            (修正后的代码, 修正列表)
        """
        self.fixes_applied = []
        original_code = code

        # 1. 修正导入语句
        code = self._fix_imports(code)

        # 2. 确保必需的导入存在
        code = self._ensure_imports(code)

        # 3. 修正 super().__init__() 调用
        code = self._fix_super_init(code)

        # 4. 修正注释格式（中文注释后需要空格）
        code = self._fix_comments(code)

        # 5. 移除剩余的相对导入
        code = self._remove_relative_imports(code)

        # 6. 修正无效的 FactorCategory
        code = self._fix_factor_category(code)

        # 7. 清理多余的空行
        code = self._cleanup_whitespace(code)

        if self.fixes_applied:
            logger.info(f"CodeFixer applied fixes: {self.fixes_applied}")

        return code, self.fixes_applied

    def _fix_imports(self, code: str) -> str:
        """修正导入语句"""
        for pattern, replacement in self.IMPORT_PATTERNS:
            if re.search(pattern, code, re.MULTILINE):
                code = re.sub(pattern, replacement, code, flags=re.MULTILINE)
                self.fixes_applied.append("修正了导入语句")
                break  # 只修正一次，避免重复
        return code

    def _ensure_imports(self, code: str) -> str:
        """确保必需的导入存在"""
        # 检查是否有正确的 core.base_factor 导入
        if "from core.base_factor import" not in code:
            # 在文件开头添加导入
            code = self.REQUIRED_IMPORT + "\n\n" + code
            self.fixes_applied.append("添加了必需的导入")
            return code

        # 检查是否导入了所有必需的类
        import_line = None
        for line in code.split("\n"):
            if line.startswith("from core.base_factor import"):
                import_line = line
                break

        if import_line:
            # 检查是否缺少某些类
            required_classes = {"BaseFactor", "FactorMetadata", "FactorCategory", "DataFrequency"}
            imported = set(re.findall(r"\b(\w+)\b", import_line.split("import")[1]))

            missing = required_classes - imported
            if missing:
                # 重新构建导入行
                new_import = f"from core.base_factor import {', '.join(required_classes)}"
                code = code.replace(import_line, new_import)
                self.fixes_applied.append(f"补充了缺失的导入: {missing}")

        return code

    def _fix_super_init(self, code: str) -> str:
        """修正 super().__init__() 调用，确保传递 metadata 参数"""
        # 检查是否有 super().__init__() 调用（无参数）
        pattern = r"super\(\)\.__init__\(\s*\)"

        if not re.search(pattern, code):
            # 检查是否有带参数的调用
            if "super().__init__(metadata" in code or "super().__init__(FactorMetadata" in code:
                return code
            # 可能是其他形式的 super 调用，或者没有 super 调用
            return code

        # 提取类名和可能的参数
        class_match = re.search(r"class\s+(\w+)\s*\(\s*(?:BaseFactor|[\w.]+BaseFactor)\s*\)", code)
        class_name = class_match.group(1) if class_match else "GeneratedFactor"

        # 尝试从代码中提取可能的 metadata 参数
        metadata = self._extract_or_create_metadata(code, class_name)

        # 替换 super().__init__() 调用
        replacement = f"""super().__init__(FactorMetadata(
                name="{metadata['name']}",
                category=FactorCategory.{metadata['category']},
                description="{metadata['description']}",
                formula="{metadata['formula']}",
                lookback_period={metadata['lookback_period']},
                frequency=DataFrequency.{metadata['frequency']},
            ))"""

        code = re.sub(pattern, replacement, code)
        self.fixes_applied.append("修正了 super().__init__() 调用，添加了 metadata 参数")

        return code

    def _extract_or_create_metadata(self, code: str, class_name: str) -> dict:
        """从代码中提取或创建 metadata"""
        # 尝试从现有代码中提取信息
        metadata = {
            'name': self._to_snake_case(class_name),
            'category': 'CUSTOM',
            'description': 'Auto-generated factor',
            'formula': 'N/A',
            'lookback_period': 20,
            'frequency': 'DAILY',
        }

        # 尝试提取窗口参数
        window_match = re.search(r"window\s*[=:]\s*(\d+)", code)
        if window_match:
            metadata['lookback_period'] = int(window_match.group(1))
            metadata['name'] = f"{metadata['name']}_{window_match.group(1)}d"

        # 尝试提取描述
        desc_match = re.search(r'"""([^"]+)"""', code, re.DOTALL)
        if desc_match:
            desc = desc_match.group(1).strip().split('\n')[0][:100]
            metadata['description'] = desc.replace('"', "'")

        return metadata

    def _to_snake_case(self, name: str) -> str:
        """将类名转换为蛇形命名"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _fix_comments(self, code: str) -> str:
        """修正注释格式（中文注释后需要空格）"""
        # 修正 #后面直接跟中文字符的情况
        def fix_comment(match):
            comment = match.group(0)
            if len(comment) > 1 and comment[1] not in ' \t':
                # 检查是否是中文字符
                if '一' <= comment[1] <= '鿿':
                    return '# ' + comment[1:]
            return comment

        code = re.sub(r'#[^\n]*', fix_comment, code)
        return code

    def _remove_relative_imports(self, code: str) -> str:
        """移除剩余的相对导入"""
        # 移除 from .xxx import 形式的导入
        code = re.sub(r'^from\s+\.\w+\s+import.*$\n?', '', code, flags=re.MULTILINE)
        # 移除 from ..xxx import 形式的导入
        code = re.sub(r'^from\s+\.\.+\w*\s*import.*$\n?', '', code, flags=re.MULTILINE)
        return code

    def _fix_factor_category(self, code: str) -> str:
        """修正无效的 FactorCategory"""
        # 有效的类别列表
        valid_categories = {
            'MOMENTUM', 'VALUE', 'QUALITY', 'GROWTH', 'VOLATILITY',
            'LIQUIDITY', 'SENTIMENT', 'TECHNICAL', 'FUNDAMENTAL', 'ALTERNATIVE', 'CUSTOM'
        }

        # 查找所有 FactorCategory.XXX 的使用
        pattern = r'FactorCategory\.(\w+)'
        matches = re.findall(pattern, code)

        for match in matches:
            if match not in valid_categories:
                # 替换为 ALTERNATIVE（最通用的类别）
                code = code.replace(f'FactorCategory.{match}', 'FactorCategory.ALTERNATIVE')
                self.fixes_applied.append(f"修正了无效的 FactorCategory.{match} -> ALTERNATIVE")

        return code

    def _cleanup_whitespace(self, code: str) -> str:
        """清理多余的空行"""
        # 移除连续超过2个空行
        code = re.sub(r'\n{3,}', '\n\n', code)
        # 移除文件开头的空行
        code = code.lstrip('\n')
        # 确保文件结尾有换行
        code = code.rstrip() + '\n'
        return code

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        验证代码是否符合基本要求

        Returns:
            (是否有效, 问题列表)
        """
        issues = []

        # 检查是否有 BaseFactor 导入
        if "from core.base_factor import" not in code:
            issues.append("缺少 core.base_factor 导入")

        # 检查是否有类定义
        if not re.search(r'class\s+\w+\s*\([^)]*BaseFactor[^)]*\)', code):
            issues.append("未找到 BaseFactor 子类定义")

        # 检查是否有 compute 方法
        if "def compute" not in code:
            issues.append("缺少 compute 方法")

        # 检查是否有 validate_inputs 方法
        if "def validate_inputs" not in code:
            issues.append("缺少 validate_inputs 方法")

        # 检查 super().__init__ 是否有参数
        if re.search(r'super\(\)\.__init__\(\s*\)', code):
            issues.append("super().__init__() 缺少 metadata 参数")

        return len(issues) == 0, issues


def fix_generated_code(code: str) -> Tuple[str, List[str]]:
    """
    便捷函数：修正 LLM 生成的代码

    Args:
        code: LLM 生成的原始代码

    Returns:
        (修正后的代码, 修正列表)
    """
    fixer = CodeFixer()
    return fixer.fix(code)
