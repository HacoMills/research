"""
美股面板数据 (用来在本地近似 WorldQuant BRAIN 的 USA / TOP3000)
================================================================
股票池: 标普 1500 (S&P 500 + 400 + 600) 当前成分股, 维基百科列表, 带 GICS 行业 / 子行业.
行情:   Yahoo Finance 日线 (yfinance), 复权 OHLC + 成交量.

和 BRAIN 的差距:
  - BRAIN 的 TOP3000 按流动性动态调整, 包含后来退市的股票; 这里是 "当前" 成分股 (幸存者偏差)
  - 行业分类用的是当前的 GICS, 不是历史时点的分类
"""

import io
import pickle
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

try:
    from .paths import CACHE_DIR
except ImportError:
    from paths import CACHE_DIR

US_CACHE = CACHE_DIR / "us_stock"
_WIKI = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "SP400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "SP600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


def sp1500_universe(refresh: bool = False) -> pd.DataFrame:
    """返回 DataFrame[ticker, index, sector, subindustry], ticker 已转成 Yahoo 写法 (BRK.B → BRK-B)."""
    path = US_CACHE / "sp1500_universe.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    import requests

    rows = []
    for name, url in _WIKI.items():
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
        t = pd.read_html(io.StringIO(html))[0]
        rows.append(pd.DataFrame({"ticker": t["Symbol"].astype(str).str.replace(".", "-", regex=False),
                                  "index": name, "sector": t["GICS Sector"],
                                  "subindustry": t["GICS Sub-Industry"]}))
    uni = pd.concat(rows).drop_duplicates("ticker").reset_index(drop=True)
    US_CACHE.mkdir(parents=True, exist_ok=True)
    uni.to_csv(path, index=False)
    return uni


def load_us_panel(tickers, start: str = "2014-10-01", refresh: bool = False,
                  chunk: int = 200) -> Dict[str, pd.DataFrame]:
    """{open, high, low, close, volume} 宽表 (日期 × 股票), 复权价. 当天已下载过就直接读缓存."""
    path = US_CACHE / f"us_panel_{pd.Timestamp(start):%Y%m%d}.pkl"
    tickers = list(tickers)
    if path.exists() and not refresh:
        P = pickle.loads(path.read_bytes())
        if set(tickers) <= set(P["close"].columns):
            return {k: v[tickers] for k, v in P.items()}

    import yfinance as yf

    parts = {f: [] for f in ("Open", "High", "Low", "Close", "Volume")}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        df = yf.download(batch, start=start, auto_adjust=True, progress=False, threads=True,
                         group_by="column")
        for f in parts:
            parts[f].append(df[f])
        print(f"  已下载 {min(i + chunk, len(tickers))}/{len(tickers)}")
    P = {f.lower(): pd.concat(v, axis=1).sort_index() for f, v in parts.items()}
    P = {k: v.loc[:, ~v.columns.duplicated()].dropna(axis=1, how="all") for k, v in P.items()}
    US_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(P))
    print(f"  缓存: {path} ({P['close'].shape[0]} 天 × {P['close'].shape[1]} 只, {datetime.now():%Y-%m-%d})")
    return P
