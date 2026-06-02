"""
测试 LLM API 连接

验证 OpenAI 兼容接口是否配置正确。
"""

import asyncio
import sys
from pathlib import Path
import traceback

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from idea_generator.llm_client import LLMClient, LLMConfig, LLMProvider


async def test_llm_connection():
    """测试 LLM 连接"""

    print("=" * 60)
    print("Testing LLM API Connection")
    print("=" * 60)

    # 配置 Qwen3 API
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        api_key="sk-2024312350-a20f8d25",
        base_url="http://10.13.66.5:20165/v1",
        temperature=0.7,
        max_tokens=1024,
    )

    print(f"\nConfig:")
    print(f"  Provider: {config.provider.value}")
    print(f"  Model: {config.model}")
    print(f"  Base URL: {config.base_url}")
    print(f"  API Key: {config.api_key[:20]}...")

    # 创建客户端
    client = LLMClient(config=config)

    # 测试 1: 简单对话
    print("\n" + "-" * 60)
    print("Test 1: Simple chat")
    print("-" * 60)

    try:
        response = await client.chat(
            messages=[{"role": "user", "content": "Hello, please say 'OK' to confirm connection."}],
        )
        print(f"Response: {response[:200]}...")
        print("Status: SUCCESS")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        return False

    # 测试 2: 生成因子思路
    print("\n" + "-" * 60)
    print("Test 2: Generate factor idea")
    print("-" * 60)

    try:
        idea = await client.generate_idea("Generate a volume-based momentum factor")
        print(f"Factor idea:\n{idea[:500]}...")
        print("Status: SUCCESS")
    except Exception as e:
        print(f"Error: {e}")
        return False

    # 测试 3: 生成代码
    print("\n" + "-" * 60)
    print("Test 3: Generate code")
    print("-" * 60)

    try:
        code = await client.generate_code(
            idea="Factor name: volume_momentum\n"
                 "Category: momentum\n"
                 "Description: Volume-based momentum factor\n"
                 "Formula: (volume / volume.rolling(20).mean()) * close.pct_change(20)"
        )
        print(f"Generated code:\n{code[:500]}...")
        print("Status: SUCCESS")
    except Exception as e:
        print(f"Error: {e}")
        return False

    # 打印统计
    print("\n" + "=" * 60)
    print("Usage Stats")
    print("=" * 60)
    stats = client.get_usage_stats()
    print(f"  Provider: {stats['provider']}")
    print(f"  Model: {stats['model']}")
    print(f"  Total tokens: {stats['total_tokens']}")

    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_llm_connection())
    sys.exit(0 if success else 1)
