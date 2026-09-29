"""
执行层: 单品种回测引擎
=====================
信号层告诉引擎"该不该进出场", 过滤层告诉引擎"现在允不允许开新仓", 引擎负责其余一切:

  仓位   每次开仓承担 1% 资金的风险: 手数 = 资金 × 1% / ATR (真实价格)
  止损   信号自带 stop_mult (几倍 ATR), None = 不止损; 开盘价已越过止损位时按开盘价成交 (跳空)
  成本   手续费、印花税、滑点 (按不利方向偏移)
  规则   做空限制、最小交易单位、现金账户上限、A股涨跌停
  盯市   每根K线收盘: 权益 = 已实现资金 + 持仓浮盈

复权数据 (有 raw_close / scale 列): 信号和止损用复权价, 手数、费用、盈亏用当时的真实价格,
以后的换月、分红不会改变过去任何一笔交易.

与旧版 strategies/turtle/turtle_engine.py 的关系: 逻辑完全相同, 只是把"唐奇安通道"换成可替换的信号.
signal=Donchian(60,20) 且不过滤时, 结果与旧版逐位一致 (见 tests).
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class MarketSpec:
    name: str                        # crypto / stock / future / fx
    days_per_year: float             # 年化用: 加密 365.25, A股/期货 252, 汇率 260
    initial_capital: float
    allow_short: bool = True
    open_fee: float = 0.0            # 开仓手续费率
    close_fee: float = 0.0           # 平仓手续费率
    sell_tax: float = 0.0            # 卖出印花税 (仅 A股)
    lot_size: float = 0.0            # 最小交易单位 (0 = 可以是小数)
    multiplier: float = 1.0          # 合约乘数
    cash_only: bool = False          # True: 持仓市值不能超过权益 (A股现金账户)
    limit_pct: Optional[pd.Series] = None   # 每根K线的涨跌停幅度 (%), None = 不考虑
    slippage: float = 0.0            # 每次成交的滑点 (比例, 0.0003 = 3 个基点)
    gap_fill: bool = True            # 止损跳空时按开盘价成交
    tick_size: float = 0.0           # 最小变动价位 (期货)
    slippage_ticks: float = 0.0      # >0 时滑点按 "几跳" 算, 代替 slippage 比例
    fee_per_lot: float = 0.0         # 按手收的手续费 (元/手, 如国债期货), 与费率叠加


RISK_PCT = 0.01          # 每次开仓承担 1% 资金的风险
ATR_PERIOD_DAYS = 20     # ATR 周期 (天)
LIMIT_TOL = 0.2          # 涨跌幅距离涨跌停不足 0.2 个百分点, 视为封板


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def atr_bars(bars_per_day: float) -> int:
    return max(int(round(ATR_PERIOD_DAYS * bars_per_day)), 14)


def daily_returns(equity: pd.Series) -> pd.Series:
    """权益曲线 → 日收益 (日内K线取每天最后一个值; 周线则是每周一个值)."""
    eq = equity.dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    return eq.resample("1D").last().dropna().pct_change().dropna()


def performance(equity: pd.Series, periods_per_year: float) -> Dict:
    eq = equity.dropna()
    out = {"sharpe": 0.0, "ann_vol_pct": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0}
    if len(eq) < 2:
        return out
    out["max_drawdown_pct"] = round(float((eq / eq.cummax() - 1).min() * 100), 2)
    r = daily_returns(eq)
    if len(r) > 1 and r.std() > 0:
        ann = math.sqrt(periods_per_year)
        out["sharpe"] = round(float(r.mean() / r.std() * ann), 3)
        out["ann_vol_pct"] = round(float(r.std() * ann * 100), 2)
        downside = r[r < 0].std()
        if downside and downside > 0:
            out["sortino"] = round(float(r.mean() / downside * ann), 3)
    return out


def run_backtest(df: pd.DataFrame, bars_per_day: float, spec: MarketSpec, signal,
                 allow: Optional[pd.Series] = None) -> Dict:
    """
    df: open/high/low/close, 可选 raw_close + scale (复权), pct_chg (A股涨跌停).
    signal: backtest.signals 里的信号对象.  allow: 过滤层给出的布尔序列 (True = 允许开仓).
    """
    n = len(df)
    idx = df.index
    ab = atr_bars(bars_per_day)
    atr_v = calc_atr(df, ab).values

    sig = signal.compute(df, bars_per_day)
    long_entry, short_entry = sig.long_entry, sig.short_entry
    long_exit, short_exit = sig.long_exit, sig.short_exit
    stop_mult = signal.stop_mult

    opn, high, low, close = (df[c].values for c in ("open", "high", "low", "close"))
    vf = allow.reindex(idx).fillna(False).values if allow is not None else None
    mult = spec.multiplier
    raw_close = df["raw_close"].values if "raw_close" in df.columns else close
    scale = df["scale"].values if "scale" in df.columns else np.ones(n)

    limit_up = np.zeros(n, dtype=bool)
    limit_down = np.zeros(n, dtype=bool)
    if spec.limit_pct is not None and "pct_chg" in df.columns:
        pct = df["pct_chg"].values
        lim = spec.limit_pct.reindex(idx).values
        with np.errstate(invalid="ignore"):
            limit_up = pct >= lim - LIMIT_TOL
            limit_down = pct <= -(lim - LIMIT_TOL)

    def to_raw(px_adj, i):
        return raw_close[i] + (px_adj - close[i]) * scale[i]

    def slip(px_adj, i, side):
        if spec.slippage_ticks > 0 and spec.tick_size > 0:          # 按最小变动价位: n 跳 (真实价格) → 复权价
            return px_adj + side * spec.slippage_ticks * spec.tick_size / scale[i]
        if spec.slippage <= 0:
            return px_adj
        return px_adj + side * to_raw(px_adj, i) * spec.slippage / scale[i]

    capital = spec.initial_capital
    cash = capital
    position = 0
    entry_adj = entry_raw = stop = qty = entry_fee = 0.0
    entry_i = 0
    trades = []
    equity = np.full(n, np.nan)
    in_pos = np.zeros(n, dtype=bool)
    n_skip_size = n_skip_limit = n_gap = n_blocked = 0
    warmup = max(sig.warmup, ab) + 2

    def close_position(i, fill_adj, reason):
        nonlocal cash, position
        fill_raw = to_raw(fill_adj, i)
        gross = position * (fill_adj - entry_adj) * scale[i] * qty * mult
        exit_fee = fill_raw * qty * mult * spec.close_fee + spec.fee_per_lot * qty
        tax = (fill_raw if position == 1 else entry_raw) * qty * mult * spec.sell_tax
        cash += gross - exit_fee - tax
        trades.append({"dir": position, "entry": entry_raw, "exit": fill_raw, "qty": qty,
                       "pnl": gross - entry_fee - exit_fee - tax, "reason": reason,
                       "entry_time": idx[entry_i], "exit_time": idx[i], "bars": i - entry_i})
        position = 0

    for i in range(warmup, n):
        if np.isnan(atr_v[i]) or atr_v[i] <= 0:
            equity[i] = cash
            continue
        price = close[i]

        # 出场
        if position != 0:
            fill, reason = None, ""
            if position == 1:
                if stop_mult and spec.gap_fill and opn[i] <= stop:
                    fill, reason = opn[i], "止损(跳空)"
                elif stop_mult and low[i] <= stop:
                    fill, reason = stop, "止损"
                elif long_exit[i]:
                    fill, reason = price, "信号"
                if fill is not None and limit_down[i]:          # 收盘跌停, 卖不出去
                    fill = None
                    n_skip_limit += 1
            else:
                if stop_mult and spec.gap_fill and opn[i] >= stop:
                    fill, reason = opn[i], "止损(跳空)"
                elif stop_mult and high[i] >= stop:
                    fill, reason = stop, "止损"
                elif short_exit[i]:
                    fill, reason = price, "信号"
            if fill is not None:
                n_gap += reason == "止损(跳空)"
                close_position(i, slip(fill, i, -position), reason)

        # 入场
        if position == 0:
            long_sig = bool(long_entry[i])
            short_sig = spec.allow_short and bool(short_entry[i])
            if (long_sig or short_sig) and vf is not None and not vf[i]:
                n_blocked += 1
            elif long_sig or short_sig:
                unit_size = (cash * RISK_PCT) / (atr_v[i] * scale[i])
                if unit_size > 0 and cash > 0:
                    size = unit_size / mult
                    if spec.cash_only:
                        size = min(size, cash / (raw_close[i] * mult))
                    if spec.lot_size > 0:
                        size = math.floor(size / spec.lot_size) * spec.lot_size
                    if long_sig and limit_up[i]:
                        n_skip_limit += 1
                    elif size <= 0:
                        n_skip_size += 1
                    else:
                        position = 1 if long_sig else -1
                        entry_adj = slip(price, i, position)
                        entry_raw = to_raw(entry_adj, i)
                        qty, entry_i = size, i
                        entry_fee = entry_raw * qty * mult * spec.open_fee + spec.fee_per_lot * qty
                        cash -= entry_fee
                        stop = entry_adj - position * (stop_mult or 0) * atr_v[i]

        # 盯市
        if position != 0:
            in_pos[i] = True
            equity[i] = cash + position * (price - entry_adj) * scale[i] * qty * mult
        else:
            equity[i] = cash

    if position != 0:
        close_position(n - 1, slip(close[-1], n - 1, -position), "结束")
        equity[-1] = cash

    return _summarize(equity, in_pos, trades, warmup, n, bars_per_day, spec, capital,
                      n_skip_size, n_skip_limit, n_gap, n_blocked, idx)


def _summarize(equity, in_pos, trades, warmup, n, bars_per_day, spec, capital,
               n_skip_size, n_skip_limit, n_gap, n_blocked, idx) -> Dict:
    eq_series = pd.Series(equity, index=idx)
    base = {"equity": eq_series, "daily_returns": daily_returns(eq_series), "trades": trades,
            "skip_size": n_skip_size, "skip_limit": n_skip_limit, "gap_stops": n_gap,
            "blocked": n_blocked}
    pnl = [t["pnl"] for t in trades]
    if not pnl:
        base.update({"total_return_pct": 0, "annual_return_pct": 0, "final_equity": capital,
                     "n_trades": 0, "win_rate": 0, "profit_factor": 0, "avg_win": 0,
                     "avg_loss": 0, "max_drawdown_pct": 0, "calmar": 0, "years": 0,
                     "sharpe": 0, "ann_vol_pct": 0, "sortino": 0, "exposure_pct": 0})
        return base

    final = float(eq_series.dropna().iloc[-1])
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    gross_loss = abs(sum(losses))
    years = (n - warmup) / (bars_per_day * spec.days_per_year)
    ann_ret = ((final / capital) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else 0
    # 周线时每年只有约 52 个收益观测, 年化因子要跟着变
    perf = performance(eq_series, spec.days_per_year * min(bars_per_day, 1))
    dd = perf["max_drawdown_pct"]
    base.update({
        "total_return_pct": round((final - capital) / capital * 100, 2),
        "annual_return_pct": round(ann_ret, 2),
        "final_equity": round(final, 2),
        "n_trades": len(pnl),
        "win_rate": round(len(wins) / len(pnl) * 100, 1),
        "profit_factor": round(sum(wins) / gross_loss, 3) if gross_loss > 0 else float("inf"),
        "avg_win": round(float(np.mean(wins)), 2) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0,
        "max_drawdown_pct": dd,
        "calmar": round(ann_ret / abs(dd), 3) if dd != 0 else 0,
        "years": round(years, 2),
        "sharpe": perf["sharpe"], "ann_vol_pct": perf["ann_vol_pct"], "sortino": perf["sortino"],
        "exposure_pct": round(float(in_pos[warmup:].mean() * 100), 1),
    })
    return base
