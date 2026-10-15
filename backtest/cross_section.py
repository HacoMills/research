"""
截面回测: 日期 × 股票 的 alpha 宽表 → 组合
==========================================
与 engine.py (单品种时序: 开/平仓布尔信号, ATR 定仓) 不同, 这里处理截面因子:
每天在所有股票之间比较 alpha, 按 alpha 分配权重, 每天调仓.

  面板     load_panel()        AKShare 后复权日线 → {open, high, low, close, volume, pct_chg} 宽表
  算子     rank / ts_decay_linear / ts_std ...   WorldQuant 101 风格, 输入输出都是宽表
  可交易   tradable_mask()     停牌、一字板、收盘封涨跌停的股票当天不能调仓 (沿用昨天的权重)
  组合     long_short_weights() 去均值后按 |alpha| 归一 (多空中性, 总杠杆 1)
           quantile_weights()  按 alpha 分 N 组, 组内等权
  收益     portfolio_returns() 两种成交假设:
             close  当天收盘算 alpha、当天收盘成交 (乐观, 与 engine.py 一致)
             open   当天收盘算 alpha、第二天开盘成交 (现实)
  评估     rank_ic()、换手率、成本

无前视: alpha_t 只用 t 日收盘及以前的数据; 收益从 t 日收盘 (或 t+1 日开盘) 之后才开始计.
"""

import time
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 252
LIMIT_TOL = 0.2          # 涨跌幅距离涨跌停不足 0.2 个百分点, 视为封板 (与 engine.py 一致)


# ──────────────────────────────────────────────────
# 面板数据
# ──────────────────────────────────────────────────

def _load_stock_sina(code: str, start_date: str, refresh: bool) -> pd.DataFrame:
    """新浪后复权日线. 东方财富接口连续请求容易被断开, 截面回测要拉几百只, 默认用新浪."""
    from data.market_data import _add_pct_chg, _cached_fetch, _import_akshare, normalize_ohlcv

    ak = _import_akshare()
    prefix = "sh" if code[0] in "69" else "bj" if code[0] in "48" else "sz"
    start = pd.Timestamp(start_date).strftime("%Y%m%d")

    def fetch(since):
        time.sleep(0.2)                                         # 限速
        raw = ak.stock_zh_a_daily(symbol=prefix + code, start_date=since or start,
                                  end_date=pd.Timestamp.now().strftime("%Y%m%d"), adjust="hfq")
        if raw is None or raw.empty:
            return pd.DataFrame()
        return normalize_ohlcv(raw)

    df = _cached_fetch(f"stock_{code}_daily_hfq_sina_{start}", fetch, refresh)
    if df.empty:
        return df
    return _add_pct_chg(df[df.index >= pd.Timestamp(start_date)])  # 涨跌幅用后复权收盘价算


def load_panel(codes: Iterable[str], start_date: str = "2015-01-01",
               refresh: bool = False, verbose: bool = True, source: str = "sina") -> Dict[str, pd.DataFrame]:
    """逐只拉取后复权日线 (有缓存), 拼成宽表. source: 'sina' / 'em' (东方财富)."""
    from data.market_data import _load_stock_one

    frames, failed = {}, []
    codes = list(codes)
    for k, code in enumerate(codes, 1):
        try:
            if source == "sina":
                df = _load_stock_sina(code, start_date, refresh)
            else:
                df = _load_stock_one(code, start_date, "hfq", refresh)
        except Exception as e:                      # 网络错误: 跳过这只, 最后汇总
            failed.append(f"{code}({type(e).__name__})")
            time.sleep(1.0)
            continue
        if not df.empty:
            frames[code] = df
        if verbose and k % 50 == 0:
            print(f"  已加载 {k}/{len(codes)}")
    if failed and verbose:
        print(f"  [!] {len(failed)} 只拉取失败: {', '.join(failed[:10])}")

    if not frames:
        raise RuntimeError("一只股票都没拉到, 检查网络")
    fields = ["open", "high", "low", "close", "volume", "pct_chg"]
    return {f: pd.DataFrame({c: d[f] for c, d in frames.items()}).sort_index() for f in fields}


def limit_pct(codes: Iterable[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """各股票每天的涨跌停幅度 (%): 主板 10, 科创板 20 (2019-07-22 起), 创业板 20 (2020-08-24 起)."""
    out = pd.DataFrame(10.0, index=index, columns=list(codes))
    for c in out.columns:
        if c.startswith("688"):
            out[c] = 20.0
        elif c.startswith("30"):
            out.loc[out.index >= "2020-08-24", c] = 20.0
    return out


def tradable_mask(P: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """True = 当天收盘可以调仓. 停牌 / 一字板 / 收盘封涨停或跌停 → False (简化: 买卖都不做)."""
    close, high, low, vol, pct = P["close"], P["high"], P["low"], P["volume"], P["pct_chg"]
    lim = limit_pct(close.columns, close.index)
    sealed = pct.abs() >= lim - LIMIT_TOL
    one_price = high <= low
    suspended = close.isna() | (vol.fillna(0) <= 0)
    return ~(sealed | one_price | suspended)


# ──────────────────────────────────────────────────
# 算子 (WorldQuant 101 风格)
# ──────────────────────────────────────────────────

def rank(x: pd.DataFrame) -> pd.DataFrame:
    """截面百分位排名: 每天在所有股票之间排, 0~1."""
    return x.rank(axis=1, pct=True)


def ts_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n).mean()


def ts_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n).std()


def ts_decay_linear(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """过去 n 天线性加权平均, 最近一天权重为 n, 最早一天为 1."""
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    out = sum(x.shift(n - 1 - i) * w[i] for i in range(n))
    return out


# ──────────────────────────────────────────────────
# 组合
# ──────────────────────────────────────────────────

def _hold_when_blocked(target: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
    """不能交易的股票沿用昨天的权重; 能交易但没有 alpha 的权重为 0."""
    w = target.fillna(0.0).where(tradable)
    return w.ffill().fillna(0.0)


def long_short_weights(alpha: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
    """去截面均值 → 按 |alpha| 之和归一: 多头 0.5 + 空头 0.5, 净敞口 0.
    先在所有有 alpha 的股票里算目标权重, 再让不能交易的股票沿用昨天的权重.
    (如果先剔除不能交易的再归一, 大面积停牌/跌停的日子权重会集中到剩下的几只股票上.)"""
    a = alpha.sub(alpha.mean(axis=1), axis=0)
    target = a.div(a.abs().sum(axis=1), axis=0)
    return _hold_when_blocked(target, tradable)


def quantile_weights(alpha: pd.DataFrame, tradable: pd.DataFrame, q: int, n_groups: int = 5) -> pd.DataFrame:
    """第 q 组 (1 = alpha 最低, n_groups = 最高) 等权做多. 分组和归一同样在全部有 alpha 的股票里做."""
    r = alpha.rank(axis=1, pct=True)
    member = (r > (q - 1) / n_groups) & (r <= q / n_groups)
    target = member.astype(float).div(member.sum(axis=1).replace(0, np.nan), axis=0)
    target = target.where(alpha.notna())
    return _hold_when_blocked(target, tradable)


def rebalance_every(w: pd.DataFrame, n: int) -> pd.DataFrame:
    """每 n 天调一次仓, 中间沿用上次的权重."""
    if n <= 1:
        return w
    out = w.copy()
    out.iloc[np.arange(len(w)) % n != 0] = np.nan
    return out.ffill().fillna(0.0)


def partial_adjust(w: pd.DataFrame, tradable: pd.DataFrame, k: float) -> pd.DataFrame:
    """部分调仓: 每天只向目标权重走 k (0~1). 不能交易的股票保持昨天的权重. k=1 等于完全调仓."""
    if k >= 1:
        return w
    target, ok = w.values, tradable.reindex_like(w).fillna(False).values
    out = np.zeros_like(target)
    prev = np.zeros(target.shape[1])
    for i in range(len(target)):
        step = prev + k * (target[i] - prev)
        prev = np.where(ok[i], step, prev)
        out[i] = prev
    return pd.DataFrame(out, index=w.index, columns=w.columns)


def portfolio_returns(w: pd.DataFrame, P: Dict[str, pd.DataFrame], cost: float = 0.0,
                      execution: str = "close") -> pd.DataFrame:
    """
    w: t 日收盘算出的目标权重.  cost: 单边成本 (按 |Δw| 计).
    execution='close': t 日收盘成交, 赚 t→t+1 收盘的收益.
    execution='open' : t+1 日开盘成交. t+1 日的收益 = 旧权重 × 隔夜 + 新权重 × 日内.
    返回 DataFrame[gross, net, turnover], 索引为收益发生的日期.
    """
    close, opn = P["close"].ffill(), P["open"]          # ffill: 停牌复牌当天的涨跌计入持仓
    ret = close.pct_change(fill_method=None).where(P["close"].notna())
    w_prev = w.shift(1).fillna(0.0)                      # t-1 收盘决定、t 日持有的权重
    if execution == "close":
        gross = (w_prev * ret.fillna(0)).sum(axis=1)
        turnover = (w - w_prev).abs().sum(axis=1)        # t 日收盘调仓, 成本记在 t 日
    elif execution == "open":
        overnight = (opn / close.shift(1) - 1).fillna(0)
        intraday = (P["close"] / opn - 1).fillna(0)
        w_prev2 = w.shift(2).fillna(0.0)
        gross = (w_prev2 * overnight).sum(axis=1) + (w_prev * intraday).sum(axis=1)
        turnover = (w_prev - w_prev2).abs().sum(axis=1)
    else:
        raise ValueError("execution 只能是 close / open")
    turnover = turnover.fillna(0.0)
    out = pd.DataFrame({"gross": gross, "turnover": turnover})
    out["net"] = out["gross"] - out["turnover"] * cost
    return out


# ──────────────────────────────────────────────────
# WorldQuant BRAIN 风格的模拟设置
# ──────────────────────────────────────────────────

def neutralize(alpha: pd.DataFrame, groups: Optional[pd.Series] = None) -> pd.DataFrame:
    """groups=None: 市场中性 (减截面均值); 否则在每个组 (行业 / 子行业) 内减均值."""
    if groups is None:
        return alpha.sub(alpha.mean(axis=1), axis=0)
    g = groups.reindex(alpha.columns)
    out = alpha.copy()
    for _, cols in g.groupby(g).groups.items():
        sub = alpha[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out


def truncate(w: pd.DataFrame, cap: float, iters: int = 5) -> pd.DataFrame:
    """单只股票的 |权重| 不超过总账面的 cap (BRAIN 的 Truncation), 截断后重新归一."""
    for _ in range(iters):
        w = w.clip(-cap, cap)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        if (w.abs() <= cap + 1e-9).all().all():
            break
    return w


def brain_simulate(alpha: pd.DataFrame, P: Dict[str, pd.DataFrame], delay: int = 1, decay: int = 0,
                   groups: Optional[pd.Series] = None, truncation: float = 0.08) -> pd.DataFrame:
    """
    近似 BRAIN 的模拟流程: delay → decay → 中性化 → 按 |alpha| 归一 → 截断, 收盘调仓, 不计成本.
    delay=1: t 日的仓位只用 t-1 日及以前的数据算 (alpha 整体滞后一天).
    """
    a = alpha.shift(delay) if delay else alpha
    if decay and decay > 1:
        a = ts_decay_linear(a, decay)
    a = neutralize(a, groups)
    w = a.div(a.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if truncation:
        w = truncate(w, truncation)
    return portfolio_returns(w, P, 0.0, "close")


def brain_metrics(r: pd.DataFrame) -> Dict:
    """
    BRAIN 口径: 收益相对一半账面 (多头或空头的规模) 年化; 换手 = 日均交易额 / 账面;
    Fitness = Sharpe × sqrt(|Returns| / max(Turnover, 0.125)). 都不扣交易成本.
    """
    g, to = r["gross"], r["turnover"]
    sharpe = g.mean() / g.std() * np.sqrt(DAYS_PER_YEAR) if g.std() > 0 else 0.0
    ret = g.mean() * DAYS_PER_YEAR * 2
    turnover = to.mean()
    eq = (1 + 2 * g).cumprod()
    return {"Sharpe": sharpe, "Returns%": ret * 100, "Turnover%": turnover * 100,
            "Fitness": sharpe * np.sqrt(abs(ret) / max(turnover, 0.125)),
            "Drawdown%": (eq / eq.cummax() - 1).min() * 100,
            "Margin(bp)": g.mean() / turnover * 1e4 if turnover > 0 else np.nan}


# ──────────────────────────────────────────────────
# 评估
# ──────────────────────────────────────────────────

def rank_ic(alpha: pd.DataFrame, P: Dict[str, pd.DataFrame], horizon: int = 1) -> pd.Series:
    """t 日 alpha 与 t→t+horizon 收盘收益的截面 Spearman 相关."""
    close = P["close"]
    fwd = close.shift(-horizon) / close - 1
    a, f = alpha.rank(axis=1), fwd.where(alpha.notna()).rank(axis=1)
    ic = a.corrwith(f, axis=1)
    n = (alpha.notna() & fwd.notna()).sum(axis=1)
    return ic[n >= 30].dropna()


def ic_summary(ic: pd.Series) -> Dict:
    if len(ic) < 2:
        return {}
    return {"IC均值": ic.mean(), "IC标准差": ic.std(), "ICIR(年化)": ic.mean() / ic.std() * np.sqrt(DAYS_PER_YEAR),
            "t值": ic.mean() / ic.std() * np.sqrt(len(ic)), "IC>0占比%": (ic > 0).mean() * 100, "天数": len(ic)}
