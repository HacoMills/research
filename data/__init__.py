# data 模块 — 统一数据抓取
# 从 fetch_data 导出常用接口

from .fetch_data import (
    create_exchange,
    fetch_ohlcv,
    fetch_or_cache,
    fetch_funding_rate,
    fetch_funding_or_cache,
    fetch_open_interest,
    fetch_oi_or_cache,
    fetch_batch,
    fetch_top_symbols,
    short_name,
    DEFAULT_SYMBOLS,
    CACHE_DIR,
)
