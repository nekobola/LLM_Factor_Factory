"""
测试 Tushare 连接
"""

import sys
from pathlib import Path
import os

# 清除代理设置
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

token = os.getenv("TUSHARE_TOKEN")
print(f"Tushare Token: {token[:10]}...{token[-10:] if token else 'Not set'}")

try:
    import tushare as ts
    ts.set_token(token)
    pro = ts.pro_api()

    # 测试获取日线数据 (daily 接口通常有权限)
    print("Testing Tushare API with daily data...")
    df = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240110')
    print(f"Success! Got {len(df)} rows")
    if len(df) > 0:
        print(df.head())
    else:
        print("No data returned, but API is working.")

except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
