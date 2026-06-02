"""
FactorFactory AI 主入口

CLI 和程序入口点。
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import date

import yaml
from loguru import logger

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """加载配置文件"""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


async def run_pipeline(
    prompt: str,
    config: dict,
    max_iterations: int = 5,
):
    """运行因子生成 Pipeline"""
    from core.pipeline import FactorPipeline
    from idea_generator.llm_client import LLMClient, LLMConfig, LLMProvider
    from validator import FactorValidator
    from data_engine import AkShareLoader

    # 初始化组件
    llm_config = LLMConfig(
        provider=LLMProvider(config.get("llm", {}).get("default_provider", "anthropic")),
    )
    llm_client = LLMClient(config=llm_config)

    validator = FactorValidator()
    data_loader = AkShareLoader()

    # 创建 Pipeline
    pipeline = FactorPipeline(
        llm_client=llm_client,
        validator=validator,
        data_loader=data_loader,
    )

    # 运行
    logger.info(f"Running pipeline with prompt: {prompt}")
    result = await pipeline.run(prompt, max_iterations=max_iterations)

    if result.success:
        logger.success(f"Factor generated: {result.factor_name}")
        logger.info(f"IC: {result.validation.ic_mean:.4f}, IR: {result.validation.ic_ir:.4f}")
    else:
        logger.error(f"Pipeline failed: {result.error}")

    return result


async def run_evolution(
    n_generations: int,
    config: dict,
):
    """运行遗传规划演化"""
    from idea_generator.evolution import EvolutionEngine, EvolutionConfig

    engine = EvolutionEngine(config=EvolutionConfig(generations=n_generations))

    # 设置适应度函数（需要用户提供）
    def fitness_func(code: str) -> float:
        # TODO: 实现适应度评估
        return 0.0

    engine.set_fitness_function(fitness_func)

    logger.info(f"Running evolution for {n_generations} generations")
    best_factors = engine.evolve()

    return best_factors


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="FactorFactory AI - 全自动金融因子挖掘与验证系统"
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # generate 命令
    gen_parser = subparsers.add_parser("generate", help="生成因子")
    gen_parser.add_argument(
        "prompt",
        type=str,
        help="因子生成提示（自然语言描述）",
    )
    gen_parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="最大迭代次数",
    )

    # evolve 命令
    evol_parser = subparsers.add_parser("evolve", help="遗传规划演化")
    evol_parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="演化代数",
    )

    # validate 命令
    val_parser = subparsers.add_parser("validate", help="验证因子")
    val_parser.add_argument(
        "--factor-file",
        type=str,
        required=True,
        help="因子代码文件路径",
    )

    # web 命令
    web_parser = subparsers.add_parser("web", help="启动 Web 界面")

    # 通用参数
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(verbose=args.verbose)

    # 加载配置
    config = load_config(args.config)

    # 执行命令
    if args.command == "generate":
        result = asyncio.run(run_pipeline(
            prompt=args.prompt,
            config=config,
            max_iterations=args.max_iterations,
        ))
        return 0 if result.success else 1

    elif args.command == "evolve":
        factors = asyncio.run(run_evolution(
            n_generations=args.generations,
            config=config,
        ))
        logger.success(f"Evolution complete, best factors: {len(factors)}")
        return 0

    elif args.command == "validate":
        logger.info(f"Validating factor from: {args.factor_file}")
        # TODO: 实现因子验证
        return 0

    elif args.command == "web":
        import subprocess
        logger.info("Starting Streamlit web app...")
        subprocess.run([
            "streamlit", "run", "web/app.py",
            "--server.port", str(config.get("web", {}).get("port", 8501)),
        ])
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
