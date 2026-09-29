#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  统一数据抓取模块

  整合来源:
    - 15min_garch.ipynb (OHLCV / 资金费率 / 未平仓合约)
    - turtle_screener.py (多品种 OHLCV 批量抓取)

  API Key 通过环境变量 或 .env 文件配置:
    OKX_API_KEY / OKX_SECRET / OKX_PASSWORD / OKX_PROXY

  使用方式:
    # 作为模块导入 (海龟策略 / 其他脚本)
    from data.fetch_data import create_exchange, fetch_or_cache

    # 作为独立脚本运行
    python fetch_data.py ohlcv --symbol BTC/USDT --timeframe 4h
    python fetch_data.py funding --symbol BTC/USDT:USDT
    python fetch_data.py batch --timeframe 4h
    python fetch_data.py --refresh          # 强制刷新全部缓存
═══════════════════════════════════════════════════════════════
"""

import sys, os, time, warnings, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════
#  环境变量 & 默认值
# ════════════════════════════════════════════════════════════════

def _load_env():
    """尝试加载 .env 文件 (同目录或上一级)."""
    for d in [Path(__file__).parent, Path(__file__).parent.parent]:
        env_file = d / '.env'
        if env_file.exists():
            for line in env_file.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())
            break

_load_env()

# 从环境变量读取
OKX_API_KEY  = os.environ.get('OKX_API_KEY', '')
OKX_SECRET   = os.environ.get('OKX_SECRET', '')
OKX_PASSWORD = os.environ.get('OKX_PASSWORD', '')
OKX_PROXY    = os.environ.get('OKX_PROXY', '')      # e.g. http://127.0.0.1:29290

# 路径统一在 data/paths.py 定义; 币圈缓存放在 data/cache/okx/
try:                                   # 作为 data 包导入时
    from .paths import DATA_DIR, OKX_CACHE as CACHE_DIR
except ImportError:                    # 直接 python fetch_data.py 运行时
    from paths import DATA_DIR, OKX_CACHE as CACHE_DIR


# 默认品种列表 (OKX USDT永续)
DEFAULT_SYMBOLS = [
    'BTC/USDT:USDT',   'ETH/USDT:USDT',   'SOL/USDT:USDT',
    'BNB/USDT:USDT',   'XRP/USDT:USDT',   'DOGE/USDT:USDT',
    'ADA/USDT:USDT',   'AVAX/USDT:USDT',  'DOT/USDT:USDT',
    'LINK/USDT:USDT',  'UNI/USDT:USDT',   'ATOM/USDT:USDT',
    'NEAR/USDT:USDT',  'APT/USDT:USDT',   'ARB/USDT:USDT',
    'OP/USDT:USDT',    'SUI/USDT:USDT',   'FIL/USDT:USDT',
    'LTC/USDT:USDT',   'ETC/USDT:USDT',  'ZEC/USDT:USDT',
]


# ════════════════════════════════════════════════════════════════
#  配置文件 & 交易所实例
# ════════════════════════════════════════════════════════════════

_config_cache = None

def _find_config_json():
    """查找 config.json (带缓存): strategies/turtle/ → quant 根目录 → data/."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    import json as _json
    root = Path(__file__).resolve().parent.parent          # quant/
    for d in [root / 'strategies' / 'turtle',               # config.json 实际放在这里
              root,
              Path(__file__).parent]:                        # data/
        p = d / 'config.json'
        if p.exists():
            try:
                with open(p) as f:
                    _config_cache = _json.load(f)
                    return _config_cache
            except Exception:
                pass
    _config_cache = {}
    return _config_cache


def has_auth_config():
    """检查是否有 API 认证配置 (环境变量 或 config.json)."""
    if OKX_API_KEY:
        return True
    return bool(_find_config_json().get('apiKey'))


def create_exchange(exchange_id='okx', need_auth=False):
    """
    创建 ccxt 交易所实例.

    认证来源优先级: 环境变量 > config.json
    代理来源优先级: 环境变量 > config.json

    need_auth=True 时带上 API Key (用于持仓查询等私有接口).
    公开数据 (K线/资金费率) 不需要 Key.
    """
    import ccxt

    cfg_file = _find_config_json()

    config = {
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    }

    # 代理: 环境变量 > config.json
    proxy = OKX_PROXY or cfg_file.get('proxy', '')
    if proxy:
        config['proxies'] = {
            'http': proxy,
            'https': proxy,
        }

    if need_auth:
        # 认证: 环境变量 > config.json
        if OKX_API_KEY:
            config['apiKey'] = OKX_API_KEY
            config['secret'] = OKX_SECRET
            config['password'] = OKX_PASSWORD
        elif cfg_file.get('apiKey'):
            config['apiKey'] = cfg_file['apiKey']
            config['secret'] = cfg_file.get('secret', '')
            config['password'] = cfg_file.get('password', '')

    cls = getattr(ccxt, exchange_id)
    exchange = cls(config)

    # 模拟盘: config.json 中 "demo": true → 切换到沙盒 API 端点
    if cfg_file.get('demo'):
        exchange.set_sandbox_mode(True)

    return exchange


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════

def short_name(symbol):
    """'BTC/USDT:USDT' → 'BTC'"""
    return symbol.split('/')[0]


def safe_filename(symbol, suffix=''):
    """生成安全文件名: BTC/USDT:USDT → BTC_USDT_USDT"""
    name = symbol.replace('/', '_').replace(':', '_')
    if suffix:
        name += f'_{suffix}'
    return name


# ════════════════════════════════════════════════════════════════
#  动态品种获取 (按市值排名, 交叉验证 OKX 可用性)
# ════════════════════════════════════════════════════════════════

# 市值前 30 大加密货币 (排除稳定币), 作为 CoinGecko 不可用时的兜底
_MCAP_FALLBACK = [
    'BTC',  'ETH',  'XRP',  'BNB',  'SOL',  'DOGE', 'ADA',  'TRX',
    'AVAX', 'LINK', 'SUI',  'TON',  'SHIB', 'XLM',  'DOT',  'HBAR',
    'BCH',  'LTC',  'UNI',  'NEAR', 'APT',  'PEPE', 'ICP',  'RENDER',
    'FET',  'ARB',  'OP',   'FIL',  'ATOM', 'ETC',  'ZEC',
]


def _fetch_mcap_ranking(top=50):
    """
    从 CoinGecko 免费 API 获取市值排名.
    返回 ['BTC', 'ETH', ...], 失败返回空列表.
    """
    import urllib.request
    import json as _json
    import ssl

    url = ('https://api.coingecko.com/api/v3/coins/markets'
           '?vs_currency=usd&order=market_cap_desc'
           f'&per_page={top}&page=1&sparkline=false')
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
        })

        # 代理: 环境变量 > config.json
        proxy = OKX_PROXY or _find_config_json().get('proxy', '')
        handlers = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({
                'http': proxy, 'https': proxy,
            }))
        handlers.append(urllib.request.HTTPSHandler(
            context=ssl.create_default_context()))
        opener = urllib.request.build_opener(*handlers)

        with opener.open(req, timeout=15) as resp:
            data = _json.loads(resp.read())
        return [c['symbol'].upper() for c in data if c.get('symbol')]
    except Exception:
        return []


def fetch_top_symbols(exchange=None, exchange_id='okx', top_n=20,
                      min_volume_usdt=0):
    """
    按市值排名获取永续合约 Top N 品种.

    逻辑:
      1. 查询 OKX 所有 USDT 永续合约, 确认哪些品种有合约
      2. 从 CoinGecko 获取全球市值排名 (失败则用内置兜底列表)
      3. 取市值排名靠前 且 OKX 有合约的 Top N

    Parameters
    ----------
    exchange      : ccxt 实例 (不传则自动创建)
    exchange_id   : 交易所 ID
    top_n         : 返回前 N 个品种

    Returns
    -------
    list[str] — 如 ['BTC/USDT:USDT', 'ETH/USDT:USDT', ...]
    """
    STABLECOINS = {
        'USDC', 'USDT', 'DAI', 'BUSD', 'TUSD', 'FDUSD', 'PYUSD',
        'UST', 'GUSD', 'PAX', 'SUSD', 'LUSD', 'FRAX', 'USDD', 'EURT',
    }
    LEVERAGED_SUFFIXES = ('3L', '3S', '2L', '2S', '5L', '5S',
                          'BULL', 'BEAR', 'UP', 'DOWN')

    try:
        if exchange is None:
            exchange = create_exchange(exchange_id)

        # ── 1. 获取 OKX 上可用的 USDT 永续合约 ──
        print(f'  获取 {exchange_id.upper()} 可用合约 ...', end='', flush=True)
        response = exchange.publicGetMarketTickers({'instType': 'SWAP'})
        raw_tickers = response.get('data', [])

        available = set()
        for t in raw_tickers:
            inst_id = t.get('instId', '')
            if inst_id.endswith('-USDT-SWAP'):
                base = inst_id.split('-')[0].upper()
                if base not in STABLECOINS:
                    if not any(base.endswith(s) for s in LEVERAGED_SUFFIXES):
                        available.add(base)

        print(f' {len(available)} 个合约', flush=True)

        # ── 2. 获取市值排名 ──
        print('  获取市值排名 ...', end='', flush=True)
        ranked = _fetch_mcap_ranking(top=50)
        if ranked:
            print(f' CoinGecko Top {len(ranked)} ✓')
        else:
            ranked = _MCAP_FALLBACK
            print(' CoinGecko 不可用, 使用内置排名')

        # ── 3. 交叉: 按市值顺序, 只取 OKX 有合约的 ──
        result = []
        for base in ranked:
            if base in available:
                result.append(f'{base}/USDT:USDT')
                if len(result) >= top_n:
                    break

        if not result:
            print('  无符合条件品种, 使用默认列表')
            return DEFAULT_SYMBOLS[:top_n]

        print(f'  市值 Top {len(result)} (OKX 永续合约):')
        for i, sym in enumerate(result, 1):
            print(f'    {i:>2d}. {short_name(sym)}')

        return result

    except Exception as e:
        print(f'  失败 ({e}), 使用默认列表')
        return DEFAULT_SYMBOLS[:top_n]


# ════════════════════════════════════════════════════════════════
#  OHLCV K线数据
# ════════════════════════════════════════════════════════════════

def fetch_ohlcv(exchange, symbol, timeframe='4h', start_date='2022-01-01',
                limit_per_request=300, since_ms=None):
    """
    分页拉取完整历史 K 线.
    来源: turtle_screener.py + 15min_garch.ipynb (fetch_ultra_long_ohlcv)

    参数:
      since_ms: 可选, 毫秒时间戳. 如果提供, 则从该时间戳开始拉取 (覆盖 start_date).
                用于增量更新: 传入已有数据最后一根K线的时间戳+1.
    """
    if since_ms is not None:
        since = since_ms
    else:
        since = exchange.parse8601(start_date + 'T00:00:00Z')
    all_data = []
    tag = short_name(symbol)
    print(f'  {tag:>6s} 获取中...', end='', flush=True)

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since,
                                           limit=limit_per_request)
        except Exception as e:
            print(f' 错误: {e}')
            return None

        if not candles:
            break

        all_data.extend(candles)
        since = candles[-1][0] + 1

        if len(candles) < limit_per_request:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not all_data:
        print(' 无数据')
        return None

    df = pd.DataFrame(all_data,
                       columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    print(f' {len(df):>6d} 根K线  ({df.index[0].date()} ~ {df.index[-1].date()})')
    return df


def fetch_or_cache(exchange, symbol, timeframe='4h', start_date='2022-01-01',
                   cache_dir=None, force=False):
    """
    有缓存读缓存, 否则拉取并存盘.
    这是外部调用的主入口.
    """
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = safe_filename(symbol, timeframe) + '.csv'
    path = cache_dir / fname

    if path.exists() and not force:
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        tag = short_name(symbol)
        print(f'  {tag:>6s} 缓存加载 {len(df):>6d} 根K线')
        return df

    df = fetch_ohlcv(exchange, symbol, timeframe, start_date)
    if df is not None:
        df.to_csv(path)
    return df


# ════════════════════════════════════════════════════════════════
#  资金费率 (Funding Rate)
# ════════════════════════════════════════════════════════════════

def fetch_funding_rate(exchange, symbol='BTC/USDT:USDT', target_days=400):
    """
    抓取 OKX 资金费率历史.
    来源: 15min_garch.ipynb Cell 28 (fetch_funding_rate_history_v4)
    资金费率每8小时结算一次, ccxt 单次上限100条, 需要分页拉取.
    """
    all_funding = []
    limit_per_request = 100
    current_time = exchange.milliseconds()
    since = current_time - (target_days * 24 * 60 * 60 * 1000)
    tag = short_name(symbol)

    print(f'  {tag:>6s} 资金费率获取中 (目标 {target_days} 天)...', end='', flush=True)

    while True:
        try:
            data = exchange.fetch_funding_rate_history(symbol, since=since,
                                                       limit=limit_per_request)
            if not data:
                break

            all_funding.extend(data)
            since = data[-1]['timestamp'] + 1

            time.sleep(exchange.rateLimit / 1000)

            if len(data) < limit_per_request:
                break
            if since > current_time:
                break

        except Exception as e:
            print(f' 错误: {e}')
            break

    if not all_funding:
        print(' 无数据')
        return pd.DataFrame()

    df = pd.DataFrame(all_funding)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'fundingRate']].dropna()
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
    df.set_index('timestamp', inplace=True)

    print(f' {len(df)} 条记录')
    return df


def fetch_funding_or_cache(exchange, symbol='BTC/USDT:USDT', target_days=400,
                           cache_dir=None, force=False):
    """资金费率的缓存版本."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = safe_filename(symbol, 'funding') + '.csv'
    path = cache_dir / fname

    if path.exists() and not force:
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        tag = short_name(symbol)
        print(f'  {tag:>6s} 资金费率缓存加载 {len(df)} 条')
        return df

    df = fetch_funding_rate(exchange, symbol, target_days)
    if not df.empty:
        df.to_csv(path)
    return df


# ════════════════════════════════════════════════════════════════
#  未平仓合约量 (Open Interest)
# ════════════════════════════════════════════════════════════════

def fetch_open_interest(exchange, symbol='BTC/USDT:USDT', timeframe='1h',
                        target_limit=3000):
    """
    抓取未平仓合约量历史 (可选, 不是所有交易所都支持).
    来源: 15min_garch.ipynb Cell 28 (fetch_open_interest_history_v4)
    """
    tag = short_name(symbol)
    print(f'  {tag:>6s} 未平仓合约量获取中...', end='', flush=True)

    try:
        all_oi = []
        since = exchange.milliseconds() - target_limit * 60 * 60 * 1000

        while len(all_oi) < target_limit:
            oi = exchange.fetch_open_interest_history(symbol, timeframe=timeframe,
                                                      since=since, limit=100)
            if not oi:
                break
            all_oi.extend(oi)
            since = oi[-1]['timestamp'] + 1
            time.sleep(exchange.rateLimit / 1000)
            if len(oi) < 100:
                break

        if not all_oi:
            print(' 无数据')
            return pd.DataFrame()

        df = pd.DataFrame(all_oi)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'openInterestAmount']].dropna()
        df.set_index('timestamp', inplace=True)
        print(f' {len(df)} 条')
        return df

    except Exception as e:
        print(f' 不可用: {e}')
        return pd.DataFrame()


def fetch_oi_or_cache(exchange, symbol='BTC/USDT:USDT', timeframe='1h',
                      target_limit=3000, cache_dir=None, force=False):
    """未平仓合约量的缓存版本."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = safe_filename(symbol, 'oi') + '.csv'
    path = cache_dir / fname

    if path.exists() and not force:
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        tag = short_name(symbol)
        print(f'  {tag:>6s} OI缓存加载 {len(df)} 条')
        return df

    df = fetch_open_interest(exchange, symbol, timeframe, target_limit)
    if not df.empty:
        df.to_csv(path)
    return df


# ════════════════════════════════════════════════════════════════
#  账户持仓 & 余额 (需要 API Key)
# ════════════════════════════════════════════════════════════════

def fetch_account_positions(exchange):
    """
    获取 OKX 真实持仓 (需要认证).
    返回非零持仓列表, 每项包含: symbol, name, side, contracts,
    entry_price, mark_price, unrealized_pnl, leverage 等.
    """
    try:
        positions = exchange.fetch_positions()
        active = []
        for p in positions:
            contracts = float(p.get('contracts', 0) or 0)
            if contracts > 0:
                active.append({
                    'symbol':          p['symbol'],
                    'name':            short_name(p['symbol']),
                    'side':            p.get('side', ''),
                    'contracts':       contracts,
                    'contract_size':   float(p.get('contractSize', 1) or 1),
                    'entry_price':     float(p.get('entryPrice', 0) or 0),
                    'mark_price':      float(p.get('markPrice', 0) or 0),
                    'liq_price':       float(p.get('liquidationPrice', 0) or 0),
                    'unrealized_pnl':  float(p.get('unrealizedPnl', 0) or 0),
                    'leverage':        float(p.get('leverage', 1) or 1),
                    'notional':        float(p.get('notional', 0) or 0),
                    'margin':          float(p.get('initialMargin', 0)
                                             or p.get('margin', 0) or 0),
                    'percentage':      float(p.get('percentage', 0) or 0),
                })
        return active
    except Exception as e:
        print(f'  ✗ 获取持仓失败: {e}')
        return []


def fetch_account_balance(exchange, currency='USDT'):
    """获取账户余额 (需要认证)."""
    try:
        balance = exchange.fetch_balance()
        info = balance.get(currency, {})
        return {
            'total': float(info.get('total', 0) or 0),
            'free':  float(info.get('free', 0) or 0),
            'used':  float(info.get('used', 0) or 0),
        }
    except Exception as e:
        print(f'  ✗ 获取余额失败: {e}')
        return {'total': 0, 'free': 0, 'used': 0}


# ════════════════════════════════════════════════════════════════
#  批量抓取
# ════════════════════════════════════════════════════════════════

def fetch_batch(symbols=None, timeframe='4h', start_date='2022-01-01',
                exchange_id='okx', cache_dir=None, force=False):
    """
    批量抓取多品种 OHLCV 数据, 返回 {symbol: DataFrame} 字典.
    turtle_screener.py 的 main 会调用这个.
    """
    symbols = symbols or DEFAULT_SYMBOLS
    exchange = create_exchange(exchange_id)
    data = {}

    print(f'\n📊 批量获取 {len(symbols)} 个品种 ({timeframe}) ...\n')

    for sym in symbols:
        df = fetch_or_cache(exchange, sym, timeframe, start_date,
                            cache_dir=cache_dir, force=force)
        if df is not None and len(df) > 50:
            data[sym] = df
        elif df is not None:
            print(f'  ⚠️ {short_name(sym)} 数据不足 ({len(df)} 根), 跳过')

    print(f'\n✅ 成功获取 {len(data)}/{len(symbols)} 个品种')
    return data


# ════════════════════════════════════════════════════════════════
#  CLI 入口
# ════════════════════════════════════════════════════════════════

def load_okx(symbol: str, start_date: str = "2022-01-01", base_tf: str = "15m",
             refresh: bool = False):
    """
    OKX 永续合约 K线, 带增量缓存 (原 turtle_timeframe_backtest.fetch_or_load).
    symbol: 'BTC' 或 'BTC/USDT:USDT'. 缓存在 data/cache/okx/{tag}_{周期}_{起始日}.csv,
    与旧脚本的缓存文件名相同, 可以直接复用.
    有缓存时从最后一根 K线本身开始拉 (覆盖上次可能未收盘的那根).
    """
    if "/" not in symbol:
        symbol = f"{symbol.upper()}/USDT:USDT"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = symbol.replace("/", "").replace(":", "_").lower()
    cache_file = CACHE_DIR / f"{tag}_{base_tf}_{start_date}.csv"

    if cache_file.exists() and refresh:
        cache_file.unlink()

    ex = create_exchange()
    if cache_file.exists():
        old = pd.read_csv(cache_file, parse_dates=["timestamp"], index_col="timestamp")
        old = old[~old.index.duplicated(keep="first")]
        since_ms = int(old.index.max().timestamp() * 1000)
        try:
            new = fetch_ohlcv(ex, symbol, timeframe=base_tf, start_date=start_date, since_ms=since_ms)
        except Exception as e:
            print(f"  [!] {symbol} 增量拉取失败, 使用缓存: {type(e).__name__}")
            new = None
        if new is not None and not new.empty:
            df = pd.concat([old, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(cache_file)
        else:
            df = old
        return df

    print(f"  从 OKX 全量拉取 {symbol} {base_tf} (从 {start_date}), 首次可能需要几分钟 ...")
    df = fetch_ohlcv(ex, symbol, timeframe=base_tf, start_date=start_date)
    if df is None or df.empty:
        return None
    df.to_csv(cache_file)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='统一数据抓取工具 (OKX 永续合约)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python fetch_data.py ohlcv --symbol BTC/USDT --timeframe 4h
  python fetch_data.py ohlcv --symbol BTC/USDT --timeframe 15m --start 2022-01-01
  python fetch_data.py funding --symbol BTC/USDT:USDT --days 400
  python fetch_data.py oi --symbol BTC/USDT:USDT
  python fetch_data.py batch --timeframe 4h
  python fetch_data.py batch --timeframe 4h --refresh
        """)

    parser.add_argument('command', nargs='?', default='batch',
                        choices=['ohlcv', 'funding', 'oi', 'batch'],
                        help='抓取类型 (默认: batch)')
    parser.add_argument('--symbol', '-s', default='BTC/USDT:USDT',
                        help='交易对 (默认: BTC/USDT:USDT)')
    parser.add_argument('--timeframe', '-t', default='4h',
                        help='K线周期 (默认: 4h)')
    parser.add_argument('--start', default='2022-01-01',
                        help='起始日期 (默认: 2022-01-01)')
    parser.add_argument('--days', type=int, default=400,
                        help='资金费率抓取天数 (默认: 400)')
    parser.add_argument('--exchange', '-e', default='okx',
                        help='交易所 (默认: okx)')
    parser.add_argument('--refresh', action='store_true',
                        help='强制刷新缓存')
    parser.add_argument('--cache-dir', default=None,
                        help=f'缓存目录 (默认: {CACHE_DIR})')

    args = parser.parse_args()
    cache = args.cache_dir or str(CACHE_DIR)

    exchange = create_exchange(args.exchange)

    if args.command == 'ohlcv':
        df = fetch_or_cache(exchange, args.symbol, args.timeframe,
                            args.start, cache_dir=cache, force=args.refresh)
        if df is not None:
            print(f'\n数据预览:\n{df.head()}\n...\n{df.tail()}')

    elif args.command == 'funding':
        df = fetch_funding_or_cache(exchange, args.symbol, args.days,
                                     cache_dir=cache, force=args.refresh)
        if not df.empty:
            print(f'\n数据预览:\n{df.head()}\n...\n{df.tail()}')

    elif args.command == 'oi':
        df = fetch_oi_or_cache(exchange, args.symbol, cache_dir=cache,
                                force=args.refresh)
        if not df.empty:
            print(f'\n数据预览:\n{df.head()}\n...\n{df.tail()}')

    elif args.command == 'batch':
        fetch_batch(timeframe=args.timeframe, start_date=args.start,
                    exchange_id=args.exchange, cache_dir=cache,
                    force=args.refresh)


if __name__ == '__main__':
    main()