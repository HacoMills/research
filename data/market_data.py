#!/usr/bin/env python3
"""
A股 / 国内期货 数据加载模块
================================
来源:
  - AKShare (免费): A股日线 (东方财富, 默认后复权)
                    期货: 新浪单月合约 → 自拼复权主力连续 (推荐) / 新浪主力连续 (未复权)
  - CSMAR 日个股回报率 (data/raw/securities/TRD_Dalyr*.xlsx): load_csmar_daily
  - 本地 CSV: 你从 Wind / CSMAR 导出的文件

所有加载函数统一返回:
  index   = timestamp (DatetimeIndex, 升序)
  columns = open, high, low, close, volume  (+ 可选 pct_chg: 当日涨跌幅, 单位 %)

安装:
  pip install akshare

注意:
  1. 新浪期货主力连续 (load_future_daily) 是"直接拼接"的连续合约, 换月时会有
     价格跳空, 可能误触发唐奇安突破.
  2. 股票默认后复权 (hfq): 已发生的历史价格不会被以后的分红改写.
     同时保存不复权价格 (raw_close), 回测时手数/手续费按真实价格算.
  3. 期货推荐 load_future_continuous: 用单月合约按持仓量换月, 自己拼接复权连续,
     换月决策只用当天收盘已知的持仓量, 没有前视偏差.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:                                   # 作为 data 包导入时
    from .paths import CN_FUTURES_CACHE, CN_STOCK_CACHE, CSMAR_CACHE, FX_CACHE, INDEX_CACHE, raw
except ImportError:                    # 直接 python market_data.py 运行时
    from paths import CN_FUTURES_CACHE, CN_STOCK_CACHE, CSMAR_CACHE, FX_CACHE, INDEX_CACHE, raw

# 常见中英文列名 → 统一列名 (兼容 AKShare / Wind / CSMAR 导出)
_COLUMN_MAP = {
    # 时间
    "timestamp": "timestamp", "date": "timestamp", "datetime": "timestamp",
    "trade_date": "timestamp", "日期": "timestamp", "交易日期": "timestamp",
    "Trddt": "timestamp",                       # CSMAR 日个股回报率文件
    # 开高低收
    "open": "open", "开盘": "open", "开盘价": "open", "Opnprc": "open",
    "high": "high", "最高": "high", "最高价": "high", "Hiprc": "high",
    "low": "low", "最低": "low", "最低价": "low", "Loprc": "low",
    "close": "close", "收盘": "close", "收盘价": "close", "Clsprc": "close",
    "今开": "open", "最新价": "close",           # 东方财富外汇历史行情
    # 成交量
    "volume": "volume", "vol": "volume", "成交量": "volume", "Dnshrtrd": "volume",
    # 涨跌幅 (用于判断涨跌停)
    "pct_chg": "pct_chg", "涨跌幅": "pct_chg", "pctchange": "pct_chg",
}

_REQUIRED = ["open", "high", "low", "close"]


# ──────────────────────────────────────────────────
# 列名统一
# ──────────────────────────────────────────────────

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """把各种来源的列名统一成 open/high/low/close/volume, 时间设为索引."""
    df = df.rename(columns={c: _COLUMN_MAP[c] for c in df.columns if c in _COLUMN_MAP})

    missing = [c for c in ["timestamp"] + _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少列: {missing}; 现有列: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if "volume" not in df.columns:
        df["volume"] = 0.0

    keep = ["open", "high", "low", "close", "volume"]
    if "pct_chg" in df.columns:
        keep.append("pct_chg")
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=_REQUIRED)
    return _clean_bars(df)


def _add_pct_chg(df: pd.DataFrame) -> pd.DataFrame:
    """没有涨跌幅列时, 用收盘价自己算 (单位 %)."""
    if "pct_chg" not in df.columns:
        df["pct_chg"] = df["close"].pct_change() * 100
    return df


# ──────────────────────────────────────────────────
# 缓存 (与原版 fetch_or_load 思路一致: 读缓存 → 拉新数据 → 合并去重)
# ──────────────────────────────────────────────────

def _cached_fetch(cache_name: str, fetch_fn, refresh: bool = False,
                  cache_dir: Path = CN_STOCK_CACHE) -> pd.DataFrame:
    """
    fetch_fn(since: str 'YYYYMMDD' | None) -> 已统一列名的 DataFrame.
    有缓存时从缓存最后一天 (含当天) 开始拉, 新数据覆盖旧数据,
    这样上次保存时还没收盘的那根 K 线也会被更新.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}.csv"

    if cache_file.exists() and not refresh:
        old = pd.read_csv(cache_file, parse_dates=["timestamp"], index_col="timestamp")
        # 日线数据当天已经更新过, 直接用缓存 (避免反复请求被服务器拒绝)
        if datetime.fromtimestamp(cache_file.stat().st_mtime).date() == datetime.now().date():
            return old
        since = old.index.max().strftime("%Y%m%d")
        try:
            new = fetch_fn(since)
        except Exception as e:
            print(f"  [!] 网络请求失败, 改用本地缓存 (截至 {old.index.max().date()}): {type(e).__name__}")
            new = None
        if new is not None and not new.empty:
            df = pd.concat([old, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = old
    else:
        df = fetch_fn(None)
        if df is None or df.empty:
            return pd.DataFrame()

    df.to_csv(cache_file, index_label="timestamp")
    return df


def _import_akshare():
    try:
        import akshare as ak
        return ak
    except ImportError:
        raise ImportError("需要安装 AKShare: pip install akshare")


# ──────────────────────────────────────────────────
# A股
# ──────────────────────────────────────────────────

def load_stock_daily(code: str, start_date: str = "2015-01-01",
                     adjust: str = "hfq", refresh: bool = False) -> pd.DataFrame:
    """
    A股日线 (东方财富).
    code   : 6 位代码, 如 '600519' / '000001' / '300750'
    adjust : 'hfq' 后复权 (默认) / 'qfq' 前复权 / '' 不复权

    复权价用于产生信号; 同时拉一份不复权价格放在 raw_close 列,
    回测时手数 (100 股整数倍)、手续费、印花税、资金约束都按真实价格计算.

    为什么默认后复权: 前复权以"最新价"为基准, 以后每次分红送转都会改写全部历史价格,
    今天和下个月拉到的历史数据不一样 (隐含未来信息, 缓存也会与新数据对不上);
    后复权以上市首日为基准, 已发生的历史价格不再变化.
    """
    df = _load_stock_one(code, start_date, adjust, refresh)
    if df.empty or adjust == "":
        if not df.empty:
            df["raw_close"], df["scale"] = df["close"], 1.0
        return df
    raw = _load_stock_one(code, start_date, "", refresh)
    df["raw_close"] = raw["close"].reindex(df.index)
    df = df.dropna(subset=["raw_close"])
    if (df["close"] <= 0).any():
        print(f"  [!] {code} 复权价出现 ≤0 (常见于前复权的老股票), 建议改用 --adjust hfq")
        df = df[df["close"] > 0]
    df["scale"] = df["raw_close"] / df["close"]
    return df


def _load_stock_one(code: str, start_date: str, adjust: str, refresh: bool) -> pd.DataFrame:
    ak = _import_akshare()
    start = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = datetime.now().strftime("%Y%m%d")

    def fetch(since):
        raw = ak.stock_zh_a_hist(symbol=code, period="daily",
                                 start_date=since or start, end_date=end,
                                 adjust=adjust)
        if raw is None or raw.empty:
            return pd.DataFrame()
        return normalize_ohlcv(raw)

    # 前复权的历史价格会随新的分红整体改写, 不能增量拼接, 每次都全量重拉
    df = _cached_fetch(f"stock_{code}_daily_{adjust or 'none'}_{start}", fetch,
                       refresh or adjust == "qfq", cache_dir=CN_STOCK_CACHE)
    if df.empty:
        return df
    df = df[df.index >= pd.Timestamp(start_date)]
    return _add_pct_chg(df)


# ──────────────────────────────────────────────────
# 国内期货
# ──────────────────────────────────────────────────

def load_future_daily(product: str, start_date: str = "2015-01-01",
                      refresh: bool = False) -> pd.DataFrame:
    """
    期货主力连续日线 (新浪, 未复权, 直接拼接; 推荐改用 load_future_continuous).
    product : 品种代码, 如 'RB' (螺纹钢) / 'CU' (铜) / 'IF' (沪深300股指)
              函数内部会转成新浪格式 'RB0'.
    """
    ak = _import_akshare()
    sina_symbol = product.upper() if product.endswith("0") else f"{product.upper()}0"
    start = pd.Timestamp(start_date).strftime("%Y%m%d")

    def fetch(since):
        # 新浪接口本身返回全部历史, since 只用来截取
        raw = ak.futures_main_sina(symbol=sina_symbol, start_date=since or start,
                                   end_date=datetime.now().strftime("%Y%m%d"))
        if raw is None or raw.empty:
            return pd.DataFrame()
        return normalize_ohlcv(raw)

    df = _cached_fetch(f"future_{sina_symbol}_daily_{start}", fetch, refresh,
                       cache_dir=CN_FUTURES_CACHE)
    if df.empty:
        return df
    df = df[df.index >= pd.Timestamp(start_date)]
    return _add_pct_chg(df)


# ──────────────────────────────────────────────────
# 期货: 用单个合约自己拼接复权主力连续 (无前视偏差)
# ──────────────────────────────────────────────────
#
# 步骤:
#   1. 从新浪逐个拉取单月合约日线 (含持仓量 hold), 如 RB2401, RB2405 ...
#   2. 换月规则 (只用当时已知信息):
#        第 t 天收盘后, 如果某个更远月合约的持仓量 > 当前主力合约,
#        就从第 t+1 天起切换到该合约. 只向远月切换, 不会切回近月.
#        当前主力没数据了 (到期) 则强制切到持仓最大的远月合约.
#   3. 复权: 在换月日, 用前一天两个合约的收盘价差 (或比值) 调整之前全部历史:
#        diff  差值复权: 之前价格 + 价差   → 保持价格差不变, 适合海龟 (ATR/止损按点数)
#        ratio 比例复权: 之前价格 × 比值   → 保持涨跌幅不变, 不会出现负价格
#        none  不复权:   直接拼接, 仅用于对比
#
# 为什么复权不会引入前视偏差:
#   后面的换月只会让"之前所有价格"整体加上同一个常数 (或乘同一个比例).
#   唐奇安通道比较的是高低关系, 整体平移/缩放不改变突破信号;
#   差值复权下 ATR 与盈亏点数不变; 比例复权下 ATR 与价格同比例缩放,
#   仓位 = 1%资金/ATR 反向缩放, 盈亏金额也不变.
#   所以任何一天的交易决策, 与之后是否还有换月无关.

CONTRACT_DIR = CN_FUTURES_CACHE / "contracts"


def _clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """去掉坏数据: 价格 ≤ 0 (新浪偶尔会有收盘价为 0 的记录)、最高价低于最低价."""
    if df.empty:
        return df
    px = df[["open", "high", "low", "close"]]
    ok = (px > 0).all(axis=1) & (df["high"] >= df["low"])
    return df[ok]


def _fetch_contract(ak, code: str, refresh: bool = False) -> pd.DataFrame:
    """拉取单个合约日线并缓存. 已到期合约数据不再变化, 缓存后永久复用."""
    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    f = CONTRACT_DIR / f"{code}.csv"
    empty_marker = CONTRACT_DIR / f"{code}.empty"

    yymm = int(code[-4:])
    now_yymm = int(datetime.now().strftime("%y%m"))
    expired = yymm < now_yymm

    if expired and not refresh:
        if f.exists():
            return _clean_bars(pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp"))
        if empty_marker.exists():
            return pd.DataFrame()
    # 未到期合约: 今天已经拉过就直接用缓存, 不重复请求 (还没上市、今天查过为空的也一样)
    today = datetime.now().date()
    if not expired and not refresh:
        if f.exists() and datetime.fromtimestamp(f.stat().st_mtime).date() == today:
            return _clean_bars(pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp"))
        if empty_marker.exists() and datetime.fromtimestamp(empty_marker.stat().st_mtime).date() == today:
            return pd.DataFrame()

    try:
        raw = ak.futures_zh_daily_sina(symbol=code)
    except Exception:
        raw = None
    time.sleep(0.3)   # 控制请求频率

    if raw is None or raw.empty:
        empty_marker.touch()        # 已到期: 永久跳过; 未到期: 当天跳过
        return pd.DataFrame()
    if empty_marker.exists():
        empty_marker.unlink()       # 之前为空、现在有数据了 (新上市)

    df = raw.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume", "hold"]].apply(
        pd.to_numeric, errors="coerce").dropna(subset=["close"])
    df.to_csv(f, index_label="timestamp")
    return _clean_bars(df)


def build_continuous(contracts: dict, adjust: str = "diff") -> pd.DataFrame:
    """
    contracts: {合约代码: DataFrame(open, high, low, close, volume, hold)}
    返回复权主力连续, 额外列:
      contract  当天使用的合约
      raw_close 该合约的原始收盘价
    """
    if adjust not in ("diff", "ratio", "none"):
        raise ValueError("adjust 只能是 diff / ratio / none")

    codes = sorted(contracts, key=lambda c: int(c[-4:]))     # 按到期先后排序
    rank = {c: i for i, c in enumerate(codes)}
    fields = ["open", "high", "low", "close", "volume", "hold"]
    panel = {k: pd.DataFrame({c: contracts[c][k] for c in codes}) for k in fields}
    dates = panel["close"].index
    close, hold = panel["close"], panel["hold"].fillna(0)

    # 用 numpy 数组逐日判断 (列按到期先后排列, 列号越大越远月)
    cl, ho = close.to_numpy(), hold.to_numpy()
    cur, pend = -1, -1
    chosen_col = np.full(len(dates), -1)
    rolls = []            # (换月日在 dates 中的位置, 旧合约, 新合约)

    for i in range(len(dates)):
        av = np.flatnonzero(~np.isnan(cl[i]))          # 当天有收盘价的合约
        if av.size == 0:
            continue
        new = -1
        if cur < 0:
            cur = av[np.argmax(ho[i, av])]
        elif pend >= 0 and not np.isnan(cl[i, pend]):
            new = pend                                  # 昨天收盘决定的换月
        elif np.isnan(cl[i, cur]):                      # 当前主力没数据了 (到期)
            later = av[av > cur]
            new = later[np.argmax(ho[i, later])] if later.size else av[np.argmax(ho[i, av])]
        if new >= 0 and new != cur:
            rolls.append((i, codes[cur], codes[new]))
            cur = new
        pend = -1
        chosen_col[i] = cur

        # 用今天收盘的持仓量决定明天是否换月
        later = av[av > cur]
        if later.size:
            cand = later[np.argmax(ho[i, later])]
            if ho[i, cand] > ho[i, cur]:
                pend = cand

    chosen = [codes[c] if c >= 0 else None for c in chosen_col]

    out = pd.DataFrame(index=dates)
    rows = np.arange(len(dates))
    ok = chosen_col >= 0
    for k in fields:
        vals = panel[k].to_numpy()
        col = np.full(len(dates), np.nan)
        col[ok] = vals[rows[ok], chosen_col[ok]]
        out[k] = col
    out["contract"] = chosen
    out["raw_close"] = out["close"]

    # 复权: 从最后一次换月往前, 逐次调整之前的历史
    price_cols = ["open", "high", "low", "close"]
    for pos, old, new in reversed(rolls):
        if adjust == "none":
            break
        # 找换月前最后一个两个合约都有收盘价的交易日
        both = close[[old, new]].iloc[:pos].dropna()
        if both.empty:
            print(f"    [!] {old}→{new} 没有共同交易日, 此次换月不复权")
            continue
        c_old, c_new = both.iloc[-1][old], both.iloc[-1][new]
        if adjust == "diff":
            out.iloc[:pos, out.columns.get_indexer(price_cols)] += c_new - c_old
        elif c_old > 0:
            out.iloc[:pos, out.columns.get_indexer(price_cols)] *= c_new / c_old

    # scale: 复权价 1 个点 = 真实价格多少点 (回测引擎用它把手数/手续费/盈亏换算回真实价格)
    out["scale"] = (out["raw_close"] / out["close"]) if adjust == "ratio" else 1.0

    out = out.dropna(subset=["close"])
    out.attrs["rolls"] = [(dates[p], o, n) for p, o, n in rolls]
    return out


def load_future_continuous(product: str, start_date: str = "2015-01-01",
                           adjust: str = "diff", refresh: bool = False,
                           verbose: bool = True) -> pd.DataFrame:
    """
    用新浪单月合约拼接复权主力连续.
    product : 品种代码, 如 'RB' / 'CU' / 'IF'
    adjust  : 'diff' 差值复权 (默认, 适合海龟) / 'ratio' 比例复权 / 'none' 不复权
    首次运行需逐个拉取合约 (每个品种约 100~150 次请求), 之后走缓存.
    """
    ak = _import_akshare()
    product = product.upper().rstrip("0")
    start = pd.Timestamp(start_date)
    now = datetime.now()

    # 覆盖区间: 起始日之后到期 ~ 未来 12 个月; 不存在的月份会被自动跳过
    codes = []
    for y in range(start.year, now.year + 2):
        for m in range(1, 13):
            if (y, m) < (start.year, start.month):
                continue
            if (y - now.year) * 12 + (m - now.month) > 12:
                continue
            codes.append(f"{product}{y % 100:02d}{m:02d}")

    if verbose:
        print(f"  拉取 {product} 单月合约 ({len(codes)} 个候选, 已到期的走缓存) ...")
    contracts = {}
    for c in codes:
        d = _fetch_contract(ak, c, refresh=refresh)
        if not d.empty:
            contracts[c] = d
    if not contracts:
        return pd.DataFrame()

    df = build_continuous(contracts, adjust=adjust)
    all_rolls = df.attrs.get("rolls", [])
    df = df[df.index >= start]
    if verbose:
        rolls = [r for r in all_rolls if r[0] >= start]
        print(f"  {len(contracts)} 个合约, 换月 {len(rolls)} 次, 复权方式: {adjust}")
    return _add_pct_chg(df)


# ──────────────────────────────────────────────────
# 汇率 (东方财富, AKShare)
# ──────────────────────────────────────────────────
#
# 可用: USDCNH, EURUSD, USDJPY, AUDUSD, GBPUSD, USDCHF, USDCAD, NZDUSD 等
# 注意: 只有汇率价格, 不含两种货币的利差 (carry); 人民币汇率有管制, 波动较小.

FX_PAIRS = ["USDCNH", "EURUSD", "USDJPY", "AUDUSD", "GBPUSD", "USDCHF", "USDCAD", "NZDUSD"]


def load_fx_daily(pair: str, start_date: str = "2015-01-01",
                  refresh: bool = False) -> pd.DataFrame:
    """外汇日线. pair 如 'EURUSD' / 'USDCNH'. 接口返回全部历史, 按起始日截取."""
    ak = _import_akshare()
    pair = pair.upper()

    def fetch(since):
        raw_df = ak.forex_hist_em(symbol=pair)
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()
        return normalize_ohlcv(raw_df)

    df = _cached_fetch(f"fx_{pair}_daily", fetch, refresh, cache_dir=FX_CACHE)
    if df.empty:
        return df
    df = df[df.index >= pd.Timestamp(start_date)]
    return _add_pct_chg(df)


# ──────────────────────────────────────────────────
# CSMAR 日个股回报率文件 (data/raw/securities/TRD_Dalyr*.xlsx)
# ──────────────────────────────────────────────────
#
# 你现有的导出字段: Stkcd, Trddt, Clsprc, Dnshrtrd, Dsmvosd, Dsmvtll, Dretwd
#   - 只有收盘价, 没有开高低: 回测时 open=high=low=close, 通道和 ATR 都按收盘价算
#     (重新导出时勾选 Opnprc / Hiprc / Loprc, 会自动使用真实开高低)
#   - Clsprc 是不复权价格; 用 Dretwd (含红利再投资) 累乘出复权价格用于产生信号,
#     以第一个交易日为基准 (相当于后复权, 以后的分红不会改写过去的价格)
#
# 首次调用会把全部 TRD_Dalyr*.xlsx 合并成一个 pickle 缓存 (十几个大 Excel, 需要几分钟),
# 之后秒读. Excel 有更新 (修改时间更晚) 时自动重建.
# Excel 放在 data/raw/securities/ (还没挪的话, 旧位置 data/securities/ 也能读);
# 合并后的缓存在 data/cache/csmar/.

SECURITIES_DIR = raw("securities")
_csmar_panel = None     # 进程内缓存, 多只股票回测时只读一次


def _read_dalyr_xlsx(path: Path) -> pd.DataFrame:
    """读一个 CSMAR 导出的 Excel. 第 2、3 行是中文字段名和单位, 跳过."""
    kw = dict(skiprows=[1, 2], dtype={"Stkcd": str})
    try:
        return pd.read_excel(path, engine="calamine", **kw)  # 快很多: pip install python-calamine
    except Exception:
        return pd.read_excel(path, **kw)


def build_csmar_daily(securities_dir: Path = SECURITIES_DIR, force: bool = False) -> pd.DataFrame:
    """合并所有 TRD_Dalyr*.xlsx 为一张长表并缓存到 data/cache/csmar/."""
    securities_dir = Path(securities_dir)
    files = sorted(securities_dir.glob("TRD_Dalyr*.xlsx"))
    if not files:
        raise FileNotFoundError(f"{securities_dir} 下没有 TRD_Dalyr*.xlsx")

    cache = CSMAR_CACHE / "TRD_Dalyr_all.pkl"
    newest = max(f.stat().st_mtime for f in files)
    if cache.exists() and not force and cache.stat().st_mtime >= newest:
        return pd.read_pickle(cache)

    print(f"  首次合并 CSMAR 日度数据: {len(files)} 个 Excel (只需一次, 请稍等) ...")
    parts = []
    for k, f in enumerate(files, 1):
        t0 = time.time()
        d = _read_dalyr_xlsx(f)
        keep = [c for c in ["Stkcd", "Trddt", "Opnprc", "Hiprc", "Loprc", "Clsprc",
                            "Dnshrtrd", "Dretwd", "ChangeRatio"] if c in d.columns]
        parts.append(d[keep])
        print(f"    [{k}/{len(files)}] {f.name}: {len(d):,} 行 ({time.time()-t0:.0f}s)")

    df = pd.concat(parts, ignore_index=True)
    df["Stkcd"] = df["Stkcd"].astype(str).str.strip().str.zfill(6)
    df["Trddt"] = pd.to_datetime(df["Trddt"], errors="coerce")
    for c in df.columns.difference(["Stkcd", "Trddt"]):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.dropna(subset=["Trddt", "Clsprc"])
            .drop_duplicates(["Stkcd", "Trddt"], keep="last")
            .sort_values(["Stkcd", "Trddt"])
            .reset_index(drop=True))
    df["Stkcd"] = df["Stkcd"].astype("category")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    print(f"  已缓存: {cache} ({len(df):,} 行, {df['Stkcd'].nunique()} 只股票)")
    return df


def load_csmar_daily(code: str, start_date: str = None,
                     securities_dir: Path = SECURITIES_DIR) -> pd.DataFrame:
    """
    从 CSMAR 日度数据取一只股票, 返回回测所需格式:
      open/high/low/close : 复权价 (用于信号; 没有开高低时都等于复权收盘价)
      raw_close           : 不复权收盘价 Clsprc (用于手数/费用/资金约束)
      scale               : raw_close / close
      pct_chg             : 当日涨跌幅 % (用于判断涨跌停)
    """
    global _csmar_panel
    if _csmar_panel is None:
        _csmar_panel = build_csmar_daily(securities_dir)
    code = str(code).zfill(6)
    d = _csmar_panel[_csmar_panel["Stkcd"] == code]
    if d.empty:
        return pd.DataFrame()

    d = d.set_index("Trddt").sort_index()
    d = d[d["Dnshrtrd"].fillna(0) > 0]                 # 去掉无成交的日子 (停牌)
    if d.empty:
        return pd.DataFrame()
    ret = d["Dretwd"].fillna(0).copy()
    ret.iloc[0] = 0.0                                   # 第一天作为基准

    # 复权收盘价: 以第一天真实收盘价为起点, 按含红利日收益率累乘
    adj_close = d["Clsprc"].iloc[0] * (1 + ret).cumprod()
    factor = adj_close / d["Clsprc"]                   # 复权价 / 真实价

    out = pd.DataFrame(index=d.index)
    has_ohl = all(c in d.columns for c in ["Opnprc", "Hiprc", "Loprc"])
    out["close"] = adj_close
    if has_ohl:
        out["open"] = d["Opnprc"] * factor
        out["high"] = d["Hiprc"] * factor
        out["low"] = d["Loprc"] * factor
    else:
        out["open"] = out["high"] = out["low"] = out["close"]
    out["volume"] = d["Dnshrtrd"]
    out["raw_close"] = d["Clsprc"]
    out["scale"] = out["raw_close"] / out["close"]
    # 涨跌幅: 优先用 CSMAR 的涨跌幅字段; 否则用不复权收盘价计算 (除权日会略有偏差)
    out["pct_chg"] = (d["ChangeRatio"] * 100 if "ChangeRatio" in d.columns
                      else d["Clsprc"].pct_change() * 100)
    out = out[["open", "high", "low", "close", "volume", "raw_close", "scale", "pct_chg"]]
    out.index.name = "timestamp"
    out.attrs["close_only"] = not has_ohl

    if start_date:
        out = out[out.index >= pd.Timestamp(start_date)]
        out.attrs["close_only"] = not has_ohl
    return out


# ──────────────────────────────────────────────────
# K线周期转换
# ──────────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """把 K线合成更大的周期, 如 '4h' / '1D'. 其他列 (raw_close 等) 取最后一个值."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for c in df.columns:
        agg.setdefault(c, "last")
    return df.resample(rule).agg(agg).dropna(subset=["close"])


# ──────────────────────────────────────────────────
# 波动率指数: 美国 VIX / 中国 QVIX
# ──────────────────────────────────────────────────
#
# 美国 VIX: 优先 FRED (VIXCLS, 1990 年起), 其次 CBOE 官网 CSV, 都失败时读本地
#           data/raw/vix/VIX_History.csv (可从 CBOE 官网手动下载).
#           ⚠️ 美股收盘在北京时间凌晨, 中国市场收盘时只能看到前一天的 VIX, 过滤层会自动滞后一天.
# 中国 QVIX: 期权论坛编制的 ETF 期权波动率指数 (AKShare), 50ETF 版 2015 年起, 300ETF 版 2019 年底起.

def _load_index_cached(name: str, fetch_fn, refresh: bool = False) -> pd.Series:
    INDEX_CACHE.mkdir(parents=True, exist_ok=True)
    f = INDEX_CACHE / f"{name}.csv"
    if f.exists() and not refresh and \
            datetime.fromtimestamp(f.stat().st_mtime).date() == datetime.now().date():
        return pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")["close"]
    try:
        s = fetch_fn()
        if s is not None and len(s):
            s.rename("close").to_frame().to_csv(f, index_label="timestamp")
            return s
    except Exception as e:
        print(f"  [!] {name} 下载失败 ({type(e).__name__}), 尝试使用缓存")
    if f.exists():
        return pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")["close"]
    return pd.Series(dtype=float)


def load_us_vix(refresh: bool = False) -> pd.Series:
    """美国 VIX 日收盘."""
    import io
    import requests

    def fetch():
        errors = []
        try:                                             # 1) FRED
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS", timeout=20)
            d = pd.read_csv(io.StringIO(r.text))
            d.columns = ["date", "close"]
            d["close"] = pd.to_numeric(d["close"], errors="coerce")
            return d.dropna().set_index(pd.to_datetime(d.dropna()["date"]))["close"]
        except Exception as e:
            errors.append(f"FRED: {type(e).__name__}")
        try:                                             # 2) CBOE
            r = requests.get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv", timeout=20)
            d = pd.read_csv(io.StringIO(r.text))
            return pd.Series(d["CLOSE"].values, index=pd.to_datetime(d["DATE"]))
        except Exception as e:
            errors.append(f"CBOE: {type(e).__name__}")
        local = raw("vix", "VIX_History.csv")          # 3) 本地文件
        if local.exists():
            d = pd.read_csv(local)
            return pd.Series(d["CLOSE"].values, index=pd.to_datetime(d["DATE"]))
        raise RuntimeError("; ".join(errors) + f"; 本地也没有 {local}")

    return _load_index_cached("us_vix", fetch, refresh)


def load_qvix(kind: str = "50etf", refresh: bool = False) -> pd.Series:
    """中国期权波动率指数 QVIX. kind: '50etf' (2015 年起) / '300etf' (2019 年底起)."""
    ak = _import_akshare()
    fn = {"50etf": ak.index_option_50etf_qvix, "300etf": ak.index_option_300etf_qvix}[kind]

    def fetch():
        d = fn()
        s = pd.Series(pd.to_numeric(d["close"], errors="coerce").values, index=pd.to_datetime(d["date"]))
        return s.dropna()

    return _load_index_cached(f"qvix_{kind}", fetch, refresh)


def load_csi300(refresh: bool = False) -> pd.Series:
    """
    沪深300 日收盘 (用于和策略做组合比较).
    优先 510300 ETF 后复权价 (含分红、扣管理费, 是实际能买到的产品, 2012 年起);
    拿不到时退回沪深300 价格指数 (不含分红, 每年少算约 2% 股息).
    返回的 Series.attrs["source"] 记录用的是哪一个.
    """
    ak = _import_akshare()

    def fetch_etf():
        d = ak.fund_etf_hist_em(symbol="510300", period="daily", start_date="20100101",
                                end_date="20500101", adjust="hfq")
        return pd.Series(pd.to_numeric(d["收盘"], errors="coerce").values,
                         index=pd.to_datetime(d["日期"])).dropna()

    def fetch_index():
        d = ak.stock_zh_index_daily(symbol="sh000300")
        return pd.Series(pd.to_numeric(d["close"], errors="coerce").values,
                         index=pd.to_datetime(d["date"])).dropna()

    s = _load_index_cached("csi300_etf510300_hfq", fetch_etf, refresh)
    if len(s):
        s.attrs["source"] = "510300 ETF 后复权 (含分红)"
        return s
    s = _load_index_cached("csi300_index", fetch_index, refresh)
    s.attrs["source"] = "沪深300 价格指数 (不含分红)"
    return s


# ──────────────────────────────────────────────────
# 本地 CSV (Wind / CSMAR 导出)
# ──────────────────────────────────────────────────

def load_csv(path: str, start_date: str = None) -> pd.DataFrame:
    """
    读本地 CSV / Excel. 列名支持中英文 (见 _COLUMN_MAP), 至少要有
    日期 + 开高低收 四列. 如有 '涨跌幅' 列会用于判断涨跌停.
    """
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        try:
            raw = pd.read_csv(path)
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="gbk")   # Wind 导出常见编码
    df = normalize_ohlcv(raw)
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    return _add_pct_chg(df)


def find_csv(csv_dir: str, symbol: str) -> str:
    """在目录里找 {symbol}.csv / .xlsx (不区分大小写)."""
    for f in os.listdir(csv_dir):
        stem, ext = os.path.splitext(f)
        if stem.lower() == symbol.lower() and ext.lower() in (".csv", ".xlsx", ".xls"):
            return os.path.join(csv_dir, f)
    raise FileNotFoundError(f"{csv_dir} 里找不到 {symbol}.csv / .xlsx")
