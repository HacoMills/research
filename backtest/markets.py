"""
第 1 层 (数据层) 的回测入口 + 各市场的交易规则
=============================================
数据下载、缓存、复权都在 quant/data/ 里 (market_data.py / fetch_data.py);
这里只做两件事:
  1. load(market, symbol, cfg)   按市场调用对应的数据函数, 返回统一格式的 K线
  2. spec_for(market, symbol, df) 返回该品种的交易规则 (手续费、滑点、乘数、涨跌停...)

品种写法 "市场:代码":
  crypto:BTC       OKX 永续合约 (15 分钟数据合成各周期)
  future:RB        国内期货, 新浪单月合约按持仓量换月拼成复权连续
  stock:600519     A股 (默认 CSMAR 本地文件, 可改 AKShare)
  fx:EURUSD        外汇 (东方财富日线)

周期:
  加密货币          15min / 30min / 1h / 4h / 1d / 1w
  期货 / A股 / 汇率  1d / 1w   (只有日线数据, 日内周期会被跳过)
"""

from dataclasses import replace
from typing import Dict, Optional, Tuple

import pandas as pd

from .engine import MarketSpec

# ──────────────────────────────────────────────────
# 交易规则 (可在配置文件 [costs] 里覆盖)
# ──────────────────────────────────────────────────

SPECS: Dict[str, MarketSpec] = {
    "crypto": MarketSpec(name="crypto", days_per_year=365.25, initial_capital=10_000,
                         open_fee=0.0005, close_fee=0.0005, slippage=0.0003),
    "stock": MarketSpec(name="stock", days_per_year=252, initial_capital=1_000_000,
                        allow_short=False, open_fee=0.00025, close_fee=0.00025, sell_tax=0.0005,
                        lot_size=100, cash_only=True, slippage=0.001),
    # 期货滑点默认每次成交 1 跳 (最小变动价位, 见 FUTURE_TICKS); 表里没有的品种退回 3 个基点
    "future": MarketSpec(name="future", days_per_year=252, initial_capital=10_000_000,
                         open_fee=0.0001, close_fee=0.0001, lot_size=1, slippage=0.0003,
                         slippage_ticks=1),
    "fx": MarketSpec(name="fx", days_per_year=260, initial_capital=1_000_000,
                     open_fee=0.00002, close_fee=0.00002, slippage=0.0001),
}

# 期货合约乘数 (每手多少单位). 交易所可能调整, 使用前请核对.
FUTURE_MULTIPLIERS = {
    # 上期所 / 上期能源
    "CU": 5, "AL": 5, "ZN": 5, "PB": 5, "NI": 1, "SN": 1, "AU": 1000, "AG": 15,
    "RB": 10, "HC": 10, "SS": 5, "BU": 10, "RU": 10, "FU": 10, "SP": 10,
    "SC": 1000, "LU": 10, "NR": 10, "AO": 20, "BR": 5, "EC": 50,
    # 大商所
    "I": 100, "J": 100, "JM": 60, "M": 10, "Y": 10, "P": 10, "A": 10, "B": 10,
    "C": 10, "CS": 10, "L": 5, "V": 5, "PP": 5, "EG": 10, "EB": 5, "PG": 20,
    "JD": 10, "LH": 16,
    # 郑商所
    "TA": 5, "MA": 10, "SR": 10, "CF": 5, "OI": 10, "RM": 10, "FG": 20, "SA": 20,
    "AP": 10, "CJ": 5, "UR": 20, "SF": 5, "SM": 5, "PK": 5, "PF": 5, "PX": 5, "SH": 30,
    # 广期所
    "SI": 5, "LC": 1,
    # 中金所
    "IF": 300, "IH": 300, "IC": 200, "IM": 200, "T": 10000, "TF": 10000,
    "TS": 20000, "TL": 10000,
}

# 最小变动价位 (元). 交易所可能调整, 使用前请核对.
FUTURE_TICKS = {
    # 中金所: 股指 0.2 点; 国债 TS 0.002 元, T/TF 0.005 元, TL 0.01 元
    "IF": 0.2, "IH": 0.2, "IC": 0.2, "IM": 0.2,
    "TS": 0.002, "TF": 0.005, "T": 0.005, "TL": 0.01,
    # 上期所 / 上期能源
    "CU": 10, "AL": 5, "ZN": 5, "PB": 5, "NI": 10, "SN": 10, "AU": 0.02, "AG": 1,
    "RB": 1, "HC": 1, "SS": 5, "BU": 1, "RU": 5, "FU": 1, "SP": 2, "SC": 0.1,
    "LU": 1, "NR": 5, "AO": 1, "BR": 5,
    # 大商所
    "I": 0.5, "J": 0.5, "JM": 0.5, "M": 1, "Y": 2, "P": 2, "A": 1, "B": 1, "C": 1, "CS": 1,
    "L": 1, "V": 1, "PP": 1, "EG": 1, "EB": 1, "PG": 1, "JD": 1, "LH": 5,
    # 郑商所
    "TA": 2, "MA": 1, "SR": 1, "CF": 5, "OI": 1, "RM": 1, "FG": 1, "SA": 1, "AP": 1,
    "UR": 1, "SF": 2, "SM": 2, "PK": 2, "PF": 2, "PX": 2, "SH": 1,
}

# 按手收取的手续费 (元/手). 在表里的品种不再按费率收. 国债期货交易所手续费 3 元/手.
FUTURE_FEE_PER_LOT = {"TS": 3, "TF": 3, "T": 3, "TL": 3}

# 周期: 名称 → (resample 规则, 每天K线数)
CRYPTO_TIMEFRAMES = {"15min": (None, 96), "30min": ("30min", 48), "1h": ("1h", 24),
                     "4h": ("4h", 6), "1d": ("1D", 1), "1w": ("1W", 1 / 7)}
DAILY_TIMEFRAMES = {"1d": (None, 1), "1w": ("W-FRI", 1 / 5)}


def timeframes_for(market: str) -> Dict[str, Tuple[Optional[str], float]]:
    return CRYPTO_TIMEFRAMES if market == "crypto" else DAILY_TIMEFRAMES


def stock_limit_pct(code: str, index: pd.DatetimeIndex) -> pd.Series:
    """
    每天的涨跌停幅度 (%): 主板 10%; 科创板 20%; 创业板 2020-08-24 起 20%; 北交所 30%.
    ST 股 (5%) 无法从代码判断, 未处理.
    """
    s = pd.Series(10.0, index=index)
    if code.startswith(("688", "689")):
        s[:] = 20.0
    elif code.startswith(("300", "301")):
        s[index >= pd.Timestamp("2020-08-24")] = 20.0
    elif code.startswith(("8", "4", "92")):
        s[:] = 30.0
    return s


def base_specs(costs: dict = None) -> Dict[str, MarketSpec]:
    """配置文件 [costs.future] slippage = 0.0005 之类的覆盖."""
    out = dict(SPECS)
    for market, over in (costs or {}).items():
        if market in out and isinstance(over, dict):
            out[market] = replace(out[market], **over)
    return out


def spec_for(market: str, symbol: str, df: pd.DataFrame, specs: Dict[str, MarketSpec],
             bars_per_day: float = 1, multipliers: dict = None) -> MarketSpec:
    base = specs[market]
    if market == "stock":
        # 周线没有逐日涨跌幅, 不考虑涨跌停
        return replace(base, limit_pct=stock_limit_pct(symbol, df.index) if bars_per_day >= 1 else None)
    if market == "future":
        prod = symbol.upper()
        mult = (multipliers or {}).get(prod, FUTURE_MULTIPLIERS.get(prod))
        if mult is None:
            print(f"    [!] 未知品种 {prod} 的合约乘数, 按 1 处理; 可在配置 [multipliers] 里指定")
            mult = 1
        over = {"multiplier": mult}
        tick = FUTURE_TICKS.get(prod)
        if base.slippage_ticks > 0:
            if tick:
                over["tick_size"] = tick
            else:
                print(f"    [!] 未知品种 {prod} 的最小变动价位, 滑点按 {base.slippage*1e4:g} 个基点处理")
                over["slippage_ticks"] = 0
        if prod in FUTURE_FEE_PER_LOT:
            over.update(fee_per_lot=FUTURE_FEE_PER_LOT[prod], open_fee=0.0, close_fee=0.0)
        return replace(base, **over)
    return base


# ──────────────────────────────────────────────────
# 数据
# ──────────────────────────────────────────────────

def parse_instrument(item: str) -> Tuple[str, str]:
    market, _, symbol = item.partition(":")
    market = market.strip().lower()
    if market not in SPECS or not symbol:
        raise ValueError(f"品种写法应为 市场:代码 (市场 = {'/'.join(SPECS)}), 收到 {item!r}")
    return market, symbol.strip().upper()


def load(market: str, symbol: str, data_cfg: dict) -> Optional[pd.DataFrame]:
    """按市场取基础 K线 (加密货币是 15 分钟, 其余是日线). data_cfg 即配置文件的 [data] 段."""
    from data import market_data
    start = str(data_cfg.get("start", "2015-01-01"))
    refresh = bool(data_cfg.get("refresh", False))

    if market == "crypto":
        from data.fetch_data import load_okx
        # 加密货币缓存按起始日命名, 默认 2022-01-01 与旧脚本共用同一份 15 分钟数据
        return load_okx(symbol, start_date=str(data_cfg.get("crypto_start", "2022-01-01")),
                        base_tf="15m", refresh=refresh)
    if market == "future":
        if data_cfg.get("futures_source", "sina") == "main":
            return market_data.load_future_daily(symbol, start_date=start, refresh=refresh)
        return market_data.load_future_continuous(
            symbol, start_date=start, adjust=data_cfg.get("futures_adjust", "diff"),
            refresh=refresh, verbose=False)
    if market == "fx":
        return market_data.load_fx_daily(symbol, start_date=start, refresh=refresh)
    if market == "stock":
        src = data_cfg.get("stock_source", "csmar")
        if src == "csmar":
            return market_data.load_csmar_daily(symbol, start_date=start)
        if src == "csv":
            return market_data.load_csv(market_data.find_csv(data_cfg.get("csv_dir", "."), symbol),
                                        start_date=start)
        return market_data.load_stock_daily(symbol, start_date=start,
                                            adjust=data_cfg.get("stock_adjust", "hfq"), refresh=refresh)
    raise ValueError(market)


def to_timeframe(df: pd.DataFrame, market: str, tf: str) -> Optional[Tuple[pd.DataFrame, float]]:
    """基础 K线 → 指定周期. 该市场不支持这个周期时返回 None."""
    table = timeframes_for(market)
    if tf not in table:
        return None
    rule, bpd = table[tf]
    if rule is None:
        return df, bpd
    from data.market_data import resample_ohlcv
    out = resample_ohlcv(df, rule)
    if bpd < 1 and "pct_chg" in out.columns:
        out = out.drop(columns="pct_chg")
    return out, bpd
