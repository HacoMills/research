"""
从 OKX 抓取历史 OHLCV 数据。

原 notebook 里交易所实例是带私钥初始化的（用于未来接实盘下单）。
这里只保留“抓公开行情数据”这一条路径，用不需要鉴权的公开 exchange
实例即可，避免任何人 clone 这个仓库后被迫处理你的账户密钥。
如果你确实需要鉴权接口（比如查持仓、下单），用 build_authenticated_exchange()
从环境变量里读取 key，不要在代码里写死。
"""
import time

import ccxt
import pandas as pd
from tqdm import tqdm

from . import config


def build_public_exchange() -> ccxt.okx:
    """构建一个不需要 API Key 的 OKX 公开数据接口。"""
    options = {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    if config.OKX_HTTP_PROXY:
        options["proxies"] = {
            "http": config.OKX_HTTP_PROXY,
            "https": config.OKX_HTTP_PROXY,
        }
    return ccxt.okx(options)


def build_authenticated_exchange() -> ccxt.okx:
    """构建带鉴权的 OKX 接口，key 全部来自环境变量。"""
    if not (config.OKX_API_KEY and config.OKX_API_SECRET and config.OKX_API_PASSWORD):
        raise RuntimeError(
            "缺少 OKX API 凭证：请在 .env 里设置 OKX_API_KEY / OKX_API_SECRET / OKX_API_PASSWORD"
        )
    options = {
        "apiKey": config.OKX_API_KEY,
        "secret": config.OKX_API_SECRET,
        "password": config.OKX_API_PASSWORD,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    }
    if config.OKX_HTTP_PROXY:
        options["proxies"] = {
            "http": config.OKX_HTTP_PROXY,
            "https": config.OKX_HTTP_PROXY,
        }
    return ccxt.okx(options)


def fetch_large_ohlcv_okx(
    exchange: ccxt.okx,
    symbol: str = "BTC/USDT",
    timeframe: str = "15m",
    target_limit: int = 50000,
) -> pd.DataFrame:
    """分页抓取长周期 OHLCV 数据（OKX 单次请求上限 100 条）。"""
    all_ohlcv = []
    limit_per_request = 100
    tf_ms = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}.get(timeframe, 15 * 60 * 1000)

    current_time = exchange.milliseconds()
    since = current_time - (target_limit * tf_ms)

    pbar = tqdm(total=target_limit, desc=f"抓取 {symbol} {timeframe}")
    while len(all_ohlcv) < target_limit:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_request)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            pbar.update(len(ohlcv))
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
            if len(ohlcv) < limit_per_request:
                break
        except Exception as e:  # noqa: BLE001 - 抓取过程中的网络异常，重试即可
            print(f"抓取数据时发生错误: {e}")
            time.sleep(1)
    pbar.close()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    if len(df) > target_limit:
        df = df.tail(target_limit)

    return df
