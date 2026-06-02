"""
遗传规划演化引擎

使用遗传算法演化生成新因子。
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Tuple
import copy

import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class Gene:
    """
    基因（因子表达式节点）

    Attributes:
        type: 节点类型 (operator, data, constant)
        value: 节点值
        children: 子节点
    """
    type: str  # "operator", "data", "constant"
    value: Any
    children: List["Gene"] = field(default_factory=list)

    def to_string(self) -> str:
        """转换为表达式字符串"""
        if self.type == "data":
            return f"data['{self.value}']"
        elif self.type == "constant":
            return str(self.value)
        elif self.type == "operator":
            if len(self.children) == 1:
                return f"{self.value}({self.children[0].to_string()})"
            elif len(self.children) == 2:
                return f"({self.children[0].to_string()} {self.value} {self.children[1].to_string()})"
        return str(self.value)

    def copy(self) -> "Gene":
        """深拷贝"""
        return Gene(
            type=self.type,
            value=self.value,
            children=[c.copy() for c in self.children],
        )


@dataclass
class Individual:
    """
    个体（一个因子）

    Attributes:
        gene: 基因树
        fitness: 适应度
        metadata: 元数据
    """
    gene: Gene
    fitness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_code(self) -> str:
        """生成 Python 代码"""
        expression = self.gene.to_string()
        return f"""
class EvolvedFactor(BaseFactor):
    def __init__(self):
        metadata = FactorMetadata(
            name="evolved_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            category=FactorCategory.CUSTOM,
            description="Evolved factor",
            formula="{expression}",
            lookback_period=20,
            frequency=DataFrequency.DAILY,
        )
        super().__init__(metadata)

    def compute(self, data):
        return {expression}

    def validate_inputs(self, data):
        return True
"""


@dataclass
class EvolutionConfig:
    """
    演化配置

    Attributes:
        population_size: 种群大小
        generations: 迭代代数
        crossover_rate: 交叉概率
        mutation_rate: 变异概率
        elitism_rate: 精英保留比例
        max_depth: 最大树深度
        tournament_size: 锦标赛大小
    """
    population_size: int = 100
    generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    elitism_rate: float = 0.1
    max_depth: int = 5
    tournament_size: int = 3


class EvolutionEngine:
    """
    遗传规划演化引擎

    通过遗传算法演化生成新因子。

    Example:
        engine = EvolutionEngine()

        # 设置适应度评估函数
        engine.set_fitness_function(lambda factor: evaluate(factor))

        # 运行演化
        best_factors = engine.evolve(n_generations=50)

        # 获取最佳因子
        best = best_factors[0]
        code = best.to_code()
    """

    # 可用数据字段
    DATA_FIELDS = ["open", "high", "low", "close", "volume", "amount", "vwap"]

    # 可用算子
    OPERATORS = {
        # 一元算子
        "unary": [
            ("rank", "rank"),  # 截面排名
            ("zscore", "zscore"),  # 截面标准化
            ("log", "np.log"),
            ("abs", "np.abs"),
            ("sign", "np.sign"),
            ("delay", "lambda x, n: x.shift(n)"),  # 滞后
            ("delta", "lambda x, n: x - x.shift(n)"),  # 差分
            ("pct_change", "lambda x, n: x.pct_change(n)"),  # 收益率
            ("ma", "lambda x, n: x.rolling(n).mean()"),  # 移动平均
            ("std", "lambda x, n: x.rolling(n).std()"),  # 滚动标准差
            ("ts_max", "lambda x, n: x.rolling(n).max()"),  # 滚动最大
            ("ts_min", "lambda x, n: x.rolling(n).min()"),  # 滚动最小
            ("ts_sum", "lambda x, n: x.rolling(n).sum()"),  # 滚动求和
        ],
        # 二元算子
        "binary": [
            ("+", "+"),
            ("-", "-"),
            ("*", "*"),
            ("/", "/"),
            (">", ">"),
            ("<", "<"),
            ("max", "np.maximum"),
            ("min", "np.minimum"),
        ],
    }

    def __init__(self, config: Optional[EvolutionConfig] = None):
        """
        初始化演化引擎

        Args:
            config: 演化配置
        """
        self.config = config or EvolutionConfig()
        self._population: List[Individual] = []
        self._fitness_func: Optional[Callable] = None
        self._generation = 0
        self._history: List[Dict[str, Any]] = []

        logger.info(f"EvolutionEngine initialized, config={self.config}")

    def set_fitness_function(self, func: Callable[[str], float]):
        """
        设置适应度评估函数

        Args:
            func: 接受因子代码，返回适应度值
        """
        self._fitness_func = func

    def random_gene(self, max_depth: int = 3, allow_binary: bool = True) -> Gene:
        """
        随机生成基因树

        Args:
            max_depth: 最大深度
            allow_binary: 是否允许二元算子

        Returns:
            随机生成的基因树
        """
        if max_depth == 0:
            # 叶子节点
            if random.random() < 0.7:
                # 数据字段
                return Gene(
                    type="data",
                    value=random.choice(self.DATA_FIELDS),
                )
            else:
                # 常数
                return Gene(
                    type="constant",
                    value=random.choice([5, 10, 20, 60, 120]),
                )

        # 随机选择节点类型
        if random.random() < 0.3:
            # 数据字段
            return Gene(
                type="data",
                value=random.choice(self.DATA_FIELDS),
            )
        elif random.random() < 0.4:
            # 常数
            return Gene(
                type="constant",
                value=random.choice([5, 10, 20, 60, 120]),
            )
        elif random.random() < 0.5 or not allow_binary:
            # 一元算子
            op_name, op_func = random.choice(self.OPERATORS["unary"])
            return Gene(
                type="operator",
                value=op_func,
                children=[self.random_gene(max_depth - 1, allow_binary)],
            )
        else:
            # 二元算子
            op_name, op_func = random.choice(self.OPERATORS["binary"])
            return Gene(
                type="operator",
                value=op_func,
                children=[
                    self.random_gene(max_depth - 1, allow_binary=False),
                    self.random_gene(max_depth - 1, allow_binary=False),
                ],
            )

    def initialize_population(self):
        """初始化种群"""
        self._population = []

        for _ in range(self.config.population_size):
            gene = self.random_gene(max_depth=self.config.max_depth)
            individual = Individual(gene=gene)
            self._population.append(individual)

        logger.info(f"Initialized population with {len(self._population)} individuals")

    def evaluate_fitness(self):
        """评估种群适应度"""
        if self._fitness_func is None:
            raise ValueError("Fitness function not set")

        for individual in self._population:
            try:
                code = individual.to_code()
                individual.fitness = self._fitness_func(code)
            except Exception as e:
                individual.fitness = 0.0
                logger.warning(f"Fitness evaluation failed: {e}")

    def selection(self) -> List[Individual]:
        """
        选择操作（锦标赛选择）

        Returns:
            被选中的个体列表
        """
        selected = []

        for _ in range(self.config.population_size):
            # 锦标赛选择
            tournament = random.sample(
                self._population,
                min(self.config.tournament_size, len(self._population))
            )
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner.copy())

        return selected

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        交叉操作

        Args:
            parent1: 父代1
            parent2: 父代2

        Returns:
            两个子代
        """
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        # 随机选择交叉点
        gene1 = parent1.gene.copy()
        gene2 = parent2.gene.copy()

        # 简单交换子树
        node1 = self._get_random_node(gene1)
        node2 = self._get_random_node(gene2)

        if node1 and node2:
            node1.type, node2.type = node2.type, node1.type
            node1.value, node2.value = node2.value, node1.value
            node1.children, node2.children = node2.children, node1.children

        return Individual(gene=gene1), Individual(gene=gene2)

    def _get_random_node(self, gene: Gene) -> Optional[Gene]:
        """随机获取树中的一个节点"""
        nodes = []

        def collect(g):
            nodes.append(g)
            for child in g.children:
                collect(child)

        collect(gene)
        return random.choice(nodes) if nodes else None

    def mutate(self, individual: Individual) -> Individual:
        """
        变异操作

        Args:
            individual: 待变异个体

        Returns:
            变异后的个体
        """
        if random.random() > self.config.mutation_rate:
            return individual

        gene = individual.gene.copy()

        # 随机选择变异点
        node = self._get_random_node(gene)
        if node:
            if node.type == "data":
                node.value = random.choice(self.DATA_FIELDS)
            elif node.type == "constant":
                node.value = random.choice([5, 10, 20, 60, 120])
            elif node.type == "operator":
                # 随机替换算子
                if len(node.children) == 1:
                    _, op_func = random.choice(self.OPERATORS["unary"])
                    node.value = op_func

        return Individual(gene=gene)

    def evolve(self, n_generations: Optional[int] = None) -> List[Individual]:
        """
        运行演化

        Args:
            n_generations: 迭代代数

        Returns:
            最优个体列表
        """
        n_generations = n_generations or self.config.generations

        # 初始化
        self.initialize_population()
        self.evaluate_fitness()

        # 记录历史
        best_fitness = max(ind.fitness for ind in self._population)
        self._history.append({
            "generation": 0,
            "best_fitness": best_fitness,
            "avg_fitness": sum(ind.fitness for ind in self._population) / len(self._population),
        })

        logger.info(f"Generation 0: best_fitness={best_fitness:.4f}")

        # 迭代
        for gen in range(1, n_generations + 1):
            self._generation = gen

            # 选择
            selected = self.selection()

            # 交叉
            new_population = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i + 1])
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected[i])

            # 变异
            new_population = [self.mutate(ind) for ind in new_population]

            # 精英保留
            elite_count = int(self.config.population_size * self.config.elitism_rate)
            elite = sorted(self._population, key=lambda x: x.fitness, reverse=True)[:elite_count]
            new_population[:elite_count] = [e.copy() for e in elite]

            self._population = new_population

            # 评估
            self.evaluate_fitness()

            # 记录
            best_fitness = max(ind.fitness for ind in self._population)
            avg_fitness = sum(ind.fitness for ind in self._population) / len(self._population)
            self._history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
            })

            logger.info(f"Generation {gen}: best_fitness={best_fitness:.4f}, avg_fitness={avg_fitness:.4f}")

        # 返回最优个体
        best_individuals = sorted(self._population, key=lambda x: x.fitness, reverse=True)
        return best_individuals[:10]

    def get_history(self) -> List[Dict[str, Any]]:
        """获取演化历史"""
        return self._history
