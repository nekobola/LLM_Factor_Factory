"""
因子注册中心

管理所有已注册因子的元数据、代码和验证结果。
支持 SQLite 持久化存储。
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Type
import json
import hashlib
import pickle

import pandas as pd
from loguru import logger

from .base_factor import BaseFactor, FactorMetadata, FactorResult, ValidationResult, FactorCategory


class FactorRegistry:
    """
    因子注册中心

    功能：
    - 注册/注销因子
    - 查询因子元数据
    - 存储验证结果
    - 因子去重（基于公式哈希）
    - SQLite 持久化

    Example:
        registry = FactorRegistry(db_path="factors.db")

        # 注册因子
        registry.register(factor, validation_result)

        # 查询因子
        factor = registry.get("momentum_20d")

        # 搜索因子
        factors = registry.search(category=FactorCategory.MOMENTUM)
    """

    def __init__(self, db_path: Optional[str] = None, use_sqlite: bool = True):
        """
        初始化注册中心

        Args:
            db_path: 数据库路径，默认为 ~/.factor_factory/factors.db
            use_sqlite: 是否使用 SQLite 持久化
        """
        self.use_sqlite = use_sqlite

        if use_sqlite:
            if db_path is None:
                db_path = str(Path.home() / ".factor_factory" / "factors.db")
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        else:
            self.db_path = None
            self._memory_store: Dict[str, Dict[str, Any]] = {}

        logger.info(f"FactorRegistry initialized, db_path={self.db_path}")

    def _init_db(self):
        """初始化 SQLite 数据库"""
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT,
                    formula TEXT NOT NULL,
                    formula_hash TEXT NOT NULL,
                    lookback_period INTEGER,
                    frequency TEXT,
                    author TEXT,
                    version TEXT,
                    tags TEXT,
                    params TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    validation_passed INTEGER,
                    ic_mean REAL,
                    ic_ir REAL,
                    monotonicity_score REAL,
                    validation_details TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_code (
                    factor_id INTEGER PRIMARY KEY,
                    code BLOB,
                    FOREIGN KEY (factor_id) REFERENCES factors(id)
                )
            """)

            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON factors(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_formula_hash ON factors(formula_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_author ON factors(author)")

            conn.commit()

    def _compute_formula_hash(self, formula: str) -> str:
        """计算公式哈希（用于去重）"""
        normalized = formula.replace(" ", "").replace("\n", "")
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def register(
        self,
        factor: BaseFactor,
        validation_result: Optional[ValidationResult] = None,
        code: Optional[str] = None,
    ) -> int:
        """
        注册因子

        Args:
            factor: 因子实例
            validation_result: 验证结果
            code: 因子源代码（可选）

        Returns:
            因子 ID

        Raises:
            ValueError: 因子已存在或公式重复
        """
        metadata = factor.metadata
        formula_hash = self._compute_formula_hash(metadata.formula)

        # 检查重复
        if self.exists(metadata.name):
            raise ValueError(f"因子 '{metadata.name}' 已存在")

        if self.formula_exists(formula_hash):
            logger.warning(f"公式已存在（hash={formula_hash}），但仍注册为新因子")

        if self.use_sqlite:
            return self._register_sqlite(metadata, validation_result, formula_hash, code)
        else:
            return self._register_memory(metadata, validation_result, formula_hash, code)

    def _register_sqlite(
        self,
        metadata: FactorMetadata,
        validation_result: Optional[ValidationResult],
        formula_hash: str,
        code: Optional[str],
    ) -> int:
        """SQLite 注册"""
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO factors (
                    name, category, description, formula, formula_hash,
                    lookback_period, frequency, author, version, tags, params,
                    created_at, updated_at,
                    validation_passed, ic_mean, ic_ir, monotonicity_score, validation_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.name,
                metadata.category.value,
                metadata.description,
                metadata.formula,
                formula_hash,
                metadata.lookback_period,
                metadata.frequency.value,
                metadata.author,
                metadata.version,
                json.dumps(metadata.tags),
                json.dumps(metadata.params),
                metadata.created_at.isoformat(),
                metadata.updated_at.isoformat(),
                int(validation_result.passed) if validation_result else None,
                validation_result.ic_mean if validation_result else None,
                validation_result.ic_ir if validation_result else None,
                validation_result.monotonicity_score if validation_result else None,
                json.dumps(validation_result.details) if validation_result else None,
            ))

            factor_id = cursor.lastrowid

            # 存储代码
            if code:
                cursor.execute(
                    "INSERT INTO factor_code (factor_id, code) VALUES (?, ?)",
                    (factor_id, pickle.dumps(code))
                )

            conn.commit()

        logger.info(f"Registered factor '{metadata.name}' (id={factor_id})")
        return factor_id

    def _register_memory(
        self,
        metadata: FactorMetadata,
        validation_result: Optional[ValidationResult],
        formula_hash: str,
        code: Optional[str],
    ) -> int:
        """内存注册"""
        factor_id = len(self._memory_store) + 1
        self._memory_store[metadata.name] = {
            "id": factor_id,
            "metadata": metadata,
            "validation": validation_result,
            "formula_hash": formula_hash,
            "code": code,
        }
        return factor_id

    def exists(self, name: str) -> bool:
        """检查因子是否存在"""
        if self.use_sqlite:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM factors WHERE name = ?", (name,)
                )
                return cursor.fetchone() is not None
        else:
            return name in self._memory_store

    def formula_exists(self, formula_hash: str) -> bool:
        """检查公式是否已存在"""
        if self.use_sqlite:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM factors WHERE formula_hash = ?", (formula_hash,)
                )
                return cursor.fetchone() is not None
        else:
            return any(
                v["formula_hash"] == formula_hash
                for v in self._memory_store.values()
            )

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取因子信息

        Args:
            name: 因子名称

        Returns:
            因子信息字典，包含 metadata 和 validation
        """
        if self.use_sqlite:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM factors WHERE name = ?", (name,)
                )
                row = cursor.fetchone()
                if row is None:
                    return None

                return self._row_to_dict(row)
        else:
            if name not in self._memory_store:
                return None
            item = self._memory_store[name]
            return {
                "id": item["id"],
                "metadata": item["metadata"].to_dict(),
                "validation": item["validation"],
            }

    def _row_to_dict(self, row) -> Dict[str, Any]:
        """将数据库行转换为字典"""
        return {
            "id": row["id"],
            "metadata": {
                "name": row["name"],
                "category": row["category"],
                "description": row["description"],
                "formula": row["formula"],
                "lookback_period": row["lookback_period"],
                "frequency": row["frequency"],
                "author": row["author"],
                "version": row["version"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "params": json.loads(row["params"]) if row["params"] else {},
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            },
            "validation": {
                "passed": bool(row["validation_passed"]) if row["validation_passed"] is not None else None,
                "ic_mean": row["ic_mean"],
                "ic_ir": row["ic_ir"],
                "monotonicity_score": row["monotonicity_score"],
                "details": json.loads(row["validation_details"]) if row["validation_details"] else {},
            } if row["validation_passed"] is not None else None,
        }

    def get_code(self, factor_id: int) -> Optional[str]:
        """
        获取因子代码

        Args:
            factor_id: 因子 ID

        Returns:
            因子代码字符串，如果不存在返回 None
        """
        if self.use_sqlite:
            import sqlite3
            import pickle
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT code FROM factor_code WHERE factor_id = ?", (factor_id,)
                )
                row = cursor.fetchone()
                if row:
                    return pickle.loads(row[0])
                return None
        else:
            for item in self._memory_store.values():
                if item.get("id") == factor_id:
                    return item.get("code")
            return None

    def update_code(self, factor_id: int, code: str) -> bool:
        """
        更新因子代码

        Args:
            factor_id: 因子 ID
            code: 新的因子代码

        Returns:
            是否更新成功
        """
        import pickle
        code_bytes = pickle.dumps(code)

        if self.use_sqlite:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE factor_code SET code = ? WHERE factor_id = ?",
                    (code_bytes, factor_id),
                )
                if cursor.rowcount == 0:
                    # 如果没有找到，尝试插入
                    cursor.execute(
                        "INSERT INTO factor_code (factor_id, code) VALUES (?, ?)",
                        (factor_id, code_bytes),
                    )
                conn.commit()
                return True
        else:
            for item in self._memory_store.values():
                if item.get("id") == factor_id:
                    item["code"] = code
                    return True
            return False

    def search(
        self,
        category: Optional[FactorCategory] = None,
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_ic: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        搜索因子

        Args:
            category: 按分类筛选
            author: 按作者筛选
            tags: 按标签筛选
            min_ic: 最小 IC 均值
            limit: 返回数量限制

        Returns:
            因子信息列表
        """
        if self.use_sqlite:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                query = "SELECT * FROM factors WHERE 1=1"
                params = []

                if category:
                    query += " AND category = ?"
                    params.append(category.value)

                if author:
                    query += " AND author = ?"
                    params.append(author)

                if min_ic is not None:
                    query += " AND ic_mean >= ?"
                    params.append(min_ic)

                query += f" ORDER BY created_at DESC LIMIT {limit}"

                cursor = conn.execute(query, params)
                results = [self._row_to_dict(row) for row in cursor.fetchall()]

                # 标签筛选（在 Python 中进行）
                if tags:
                    results = [
                        r for r in results
                        if any(t in r["metadata"]["tags"] for t in tags)
                    ]

                return results
        else:
            results = []
            for item in self._memory_store.values():
                metadata = item["metadata"]
                validation = item["validation"]

                if category and metadata.category != category:
                    continue
                if author and metadata.author != author:
                    continue
                if min_ic and (not validation or validation.ic_mean < min_ic):
                    continue
                if tags and not any(t in metadata.tags for t in tags):
                    continue

                results.append({
                    "id": item["id"],
                    "metadata": metadata.to_dict(),
                    "validation": validation,
                })

            return results[:limit]

    def delete(self, name: str) -> bool:
        """
        删除因子

        Args:
            name: 因子名称

        Returns:
            是否删除成功
        """
        if not self.exists(name):
            return False

        if self.use_sqlite:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM factor_code WHERE factor_id = (SELECT id FROM factors WHERE name = ?)", (name,))
                conn.execute("DELETE FROM factors WHERE name = ?", (name,))
                conn.commit()
        else:
            del self._memory_store[name]

        logger.info(f"Deleted factor '{name}'")
        return True

    def update_validation(
        self,
        name: str,
        validation_result: ValidationResult,
    ) -> bool:
        """
        更新因子验证结果

        Args:
            name: 因子名称
            validation_result: 新的验证结果

        Returns:
            是否更新成功
        """
        if not self.exists(name):
            return False

        if self.use_sqlite:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE factors SET
                        validation_passed = ?,
                        ic_mean = ?,
                        ic_ir = ?,
                        monotonicity_score = ?,
                        validation_details = ?,
                        updated_at = ?
                    WHERE name = ?
                """, (
                    int(validation_result.passed),
                    validation_result.ic_mean,
                    validation_result.ic_ir,
                    validation_result.monotonicity_score,
                    json.dumps(validation_result.details),
                    datetime.now().isoformat(),
                    name,
                ))
                conn.commit()
        else:
            self._memory_store[name]["validation"] = validation_result

        logger.info(f"Updated validation for factor '{name}'")
        return True

    def list_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有因子"""
        return self.search(limit=limit)

    def count(self) -> int:
        """统计因子数量"""
        if self.use_sqlite:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM factors")
                return cursor.fetchone()[0]
        else:
            return len(self._memory_store)

    def export_to_json(self, output_path: str):
        """导出所有因子到 JSON 文件"""
        factors = self.list_all(limit=10000)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(factors, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported {len(factors)} factors to {output_path}")
