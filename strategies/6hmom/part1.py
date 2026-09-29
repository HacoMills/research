import os as _os
from pathlib import Path as _Path
# OKX 密钥从 quant/data/.env 读取 (OKX_API_KEY / OKX_SECRET / OKX_PASSWORD), 不写在代码里
_env = next((p / "data" / ".env" for p in _Path(__file__).resolve().parents if (p / "data" / ".env").exists()), None)
if _env:
    for _l in _env.read_text(encoding="utf-8").splitlines():
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.split("=", 1)
            _os.environ.setdefault(_k.strip(), _v.strip())
import ccxt
import pandas as pd
import time #处理时间的包

exchange = ccxt.okx({
    'apiKey': _os.environ.get('OKX_API_KEY', ''),
    'secret': _os.environ.get('OKX_SECRET', ''),
    'password': _os.environ.get('OKX_PASSWORD', ''),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap' 
    }  
    'proxies': {
        'http': 'http://127.0.0.1:29290',  
        'https': 'http://127.0.0.1:29290', 
    }
})

exchange_open = ccxt.okx({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'  
    },
    'proxies': {
        'http': 'http://127.0.0.1:29290',  
        'https': 'http://127.0.0.1:29290', 
    }
})

def fetch_full_ohlcv( #定义函数
    exchange,
    symbol: str,
    timeframe: str = "6h",
    since_str: str = "2021-01-01T00:00:00Z", #国际通用时间戳
    until_str: str = "2024-12-31T23:59:59Z",
    limit: int = 1000,
) -> pd.DataFrame  

since_ms = exchange.parse8601(since_str) #处理时间戳
until_ms = exchange.parse8601(until_str)

timeframe_ms = exchange.parse_timeframe(timeframe) * 1000