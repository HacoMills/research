#!/usr/bin/env python3
"""
海龟回测引擎 (币圈 / A股 / 期货共用)
====================================
turtle_timeframe_backtest.py 和 turtle_multi_asset.py 都调用这里的 run_backtest.

相比旧版引擎的改进:
  1. 权益曲线逐根盯市: 每根K线收盘时 = 已实现资金 + 持仓浮动盈亏.
     最大回撤、Calmar、夏普都基于这条曲线 (旧版只在平仓时更新, 回撤被低估)
  2. 新增指标: 夏普、年化波动率、Sortino、持仓时间占比
  3. 止损跳空: 开盘价已经越过止损位时, 按开盘价成交 (旧版按止损价, 偏乐观)
  4. 滑点: 每次成交都按不利方向偏移一个比例 (spec.slippage)

策略规则不变: 收盘价突破唐奇安通道入场, 2N 止损, 反向通道出场, 1% 风险/ATR 定仓位.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────
# 市场规则
# ──────────────────────────────────────────────────

@dataclass
class MarketSpec:
    name: str
    timeframes: list                 # [(名称, resample 规则, 每天K线数)]
    days_per_year: float             # 年化用: 加密 365.25, A股/期货 252
    initial_capital: float
    allow_short: bool = True
    open_fee: float = 0.0            # 开仓手续费率
    close_fee: float = 0.0           # 平仓手续费率
    sell_tax: float = 0.0            # 卖出印花税 (仅 A股)
    lot_size: float = 0.0            # 最小交易单位 (0 = 可以是小数)
    multiplier: float = 1.0          # 合约乘数
    cash_only: bool = False          # True: 持仓市值不能超过权益 (A股现金账户)
    limit_pct: Optional[pd.Series] = None   # 每根K线的涨跌停幅度 (%), None = 不考虑
    slippage: float = 0.0            # 每次成交的滑点 (比例, 如 0.0003 = 3 个基点)
    gap_fill: bool = True            # 止损跳空时按开盘价成交


# 策略参数
RISK_PCT = 0.01          # 每个单位承担 1% 资金的风险
STOP_MULT = 2.0          # 2N 止损
ATR_PERIOD_DAYS = 20     # ATR 周期 (天)
LIMIT_TOL = 0.2          # 涨跌幅距离涨跌停不足 0.2 个百分点, 视为封板


# ──────────────────────────────────────────────────
# 指标
# ──────────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calc_donchian(high: pd.Series, low: pd.Series, period: int):
    """过去 period 根的最高/最低价, shift(1) 保证不含当根 (无前视)."""
    return high.rolling(period).max().shift(1), low.rolling(period).min().shift(1)


# ──────────────────────────────────────────────────
# 绩效指标 (基于逐根盯市的权益曲线)
# ──────────────────────────────────────────────────

def daily_returns(equity: pd.Series) -> pd.Series:
    """把权益曲线转成日收益. 日内K线取每天最后一个值; 日线直接用."""
    eq = equity.dropna()
    if eq.empty:
        return pd.Series(dtype=float)
    daily = eq.resample("1D").last().dropna()
    return daily.pct_change().dropna()


def performance(equity: pd.Series, days_per_year: float) -> Dict:
    """夏普、年化波动率、Sortino、最大回撤 (都基于盯市权益)."""
    eq = equity.dropna()
    out = {"sharpe": 0.0, "ann_vol_pct": 0.0, "sortino": 0.0, "max_drawdown_pct": 0.0}
    if len(eq) < 2:
        return out
    dd = (eq / eq.cummax() - 1).min() * 100
    r = daily_returns(eq)
    out["max_drawdown_pct"] = round(float(dd), 2)
    if len(r) > 1 and r.std() > 0:
        ann = math.sqrt(days_per_year)
        out["sharpe"] = round(float(r.mean() / r.std() * ann), 3)
        out["ann_vol_pct"] = round(float(r.std() * ann * 100), 2)
        downside = r[r < 0].std()
        if downside and downside > 0:
            out["sortino"] = round(float(r.mean() / downside * ann), 3)
    return out


# ──────────────────────────────────────────────────
# 回测
# ──────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, entry_days: int, exit_days: int,
                 bars_per_day: int, spec: MarketSpec,
                 vol_filter: pd.Series = None) -> Dict:
    """
    单品种海龟回测.
    df 需要 open/high/low/close; 可选 raw_close + scale (复权数据的真实价格换算),
    pct_chg (A股涨跌停).
    """
    entry_bars = entry_days * bars_per_day
    exit_bars = exit_days * bars_per_day
    atr_bars = ATR_PERIOD_DAYS * bars_per_day
    if bars_per_day == 1:
        atr_bars = max(atr_bars, 14)

    atr = calc_atr(df, atr_bars)
    entry_upper, entry_lower = calc_donchian(df["high"], df["low"], entry_bars)
    exit_upper, exit_lower = calc_donchian(df["high"], df["low"], exit_bars)

    opn = df["open"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr_v = atr.values
    eu, el = entry_upper.values, entry_lower.values
    xu, xl = exit_upper.values, exit_lower.values
    vf = vol_filter.reindex(df.index).fillna(False).values if vol_filter is not None else None
    n = len(df)
    mult = spec.multiplier
    idx = df.index

    # 真实价格换算: 信号用复权价, 手数/费用/盈亏用真实价格 (见 turtle_multi_asset 说明)
    raw_close = df["raw_close"].values if "raw_close" in df.columns else close
    scale = df["scale"].values if "scale" in df.columns else np.ones(n)

    # A股涨跌停
    limit_up = np.zeros(n, dtype=bool)
    limit_down = np.zeros(n, dtype=bool)
    if spec.limit_pct is not None and "pct_chg" in df.columns:
        pct = df["pct_chg"].values
        lim = spec.limit_pct.reindex(idx).values
        with np.errstate(invalid="ignore"):
            limit_up = pct >= lim - LIMIT_TOL
            limit_down = pct <= -(lim - LIMIT_TOL)

    def to_raw(px_adj, i):
        """复权价 → 当根的真实价格."""
        return raw_close[i] + (px_adj - close[i]) * scale[i]

    def slip(px_adj, i, side):
        """按不利方向加滑点. side=+1 买入 (价格变高), -1 卖出 (价格变低)."""
        if spec.slippage <= 0:
            return px_adj
        return px_adj + side * to_raw(px_adj, i) * spec.slippage / scale[i]

    capital = spec.initial_capital
    cash = capital               # 已实现资金 (开仓手续费在开仓时扣除)
    position = 0
    entry_adj = entry_raw = stop = qty = 0.0
    entry_i = 0
    entry_fee = 0.0

    trades = []
    equity = np.full(n, np.nan)  # 逐根盯市权益
    in_pos = np.zeros(n, dtype=bool)
    n_skip_size = n_skip_limit = n_gap = 0

    warmup = max(entry_bars, exit_bars, atr_bars) + 2

    def close_position(i, fill_adj, reason):
        nonlocal cash, position
        fill_raw = to_raw(fill_adj, i)
        gross = position * (fill_adj - entry_adj) * scale[i] * qty * mult
        exit_fee = fill_raw * qty * mult * spec.close_fee
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

        # ── 持仓检查出场 ──
        if position != 0:
            fill, reason = None, ""
            if position == 1:
                if spec.gap_fill and opn[i] <= stop:
                    fill, reason = opn[i], "止损(跳空)"
                elif low[i] <= stop:
                    fill, reason = stop, "止损"
                elif not np.isnan(xl[i]) and price < xl[i]:
                    fill, reason = price, "通道"
                if fill is not None and limit_down[i]:          # A股收盘跌停, 卖不出去
                    fill = None
                    n_skip_limit += 1
            else:
                if spec.gap_fill and opn[i] >= stop:
                    fill, reason = opn[i], "止损(跳空)"
                elif high[i] >= stop:
                    fill, reason = stop, "止损"
                elif not np.isnan(xu[i]) and price > xu[i]:
                    fill, reason = price, "通道"
            if fill is not None:
                if reason == "止损(跳空)":
                    n_gap += 1
                close_position(i, slip(fill, i, -position), reason)

        # ── 空仓检查入场 ──
        if position == 0 and not (vf is not None and not vf[i]):
            unit_size = (cash * RISK_PCT) / (atr_v[i] * scale[i])    # 标的单位数 (真实价格)
            long_sig = not np.isnan(eu[i]) and price > eu[i]
            short_sig = spec.allow_short and not np.isnan(el[i]) and price < el[i]

            if (long_sig or short_sig) and unit_size > 0 and cash > 0:
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
                    entry_fee = entry_raw * qty * mult * spec.open_fee
                    cash -= entry_fee
                    stop = entry_adj - position * STOP_MULT * atr_v[i]

        # ── 逐根盯市 ──
        if position != 0:
            in_pos[i] = True
            equity[i] = cash + position * (price - entry_adj) * scale[i] * qty * mult
        else:
            equity[i] = cash

    # 末尾平仓
    if position != 0:
        close_position(n - 1, slip(close[-1], n - 1, -position), "结束")
        equity[-1] = cash

    return _summarize(equity, in_pos, trades, warmup, n, bars_per_day, spec,
                      capital, n_skip_size, n_skip_limit, n_gap, idx)


def _summarize(equity, in_pos, trades, warmup, n, bars_per_day, spec, capital,
               n_skip_size, n_skip_limit, n_gap, idx) -> Dict:
    eq_series = pd.Series(equity, index=idx)
    base = {"equity_curve": equity, "equity": eq_series,
            "daily_returns": daily_returns(eq_series),
            "skip_size": n_skip_size, "skip_limit": n_skip_limit, "gap_stops": n_gap,
            "trades": trades}

    pnl_list = [t["pnl"] for t in trades]
    n_trades = len(pnl_list)
    if n_trades == 0:
        base.update({"total_return_pct": 0, "annual_return_pct": 0, "final_equity": capital,
                     "n_trades": 0, "win_rate": 0, "profit_factor": 0, "avg_win": 0,
                     "avg_loss": 0, "max_drawdown_pct": 0, "calmar": 0, "years": 0,
                     "sharpe": 0, "ann_vol_pct": 0, "sortino": 0, "exposure_pct": 0})
        return base

    final = float(eq_series.dropna().iloc[-1])
    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p <= 0]
    gross_loss = abs(sum(losses))
    pf = sum(wins) / gross_loss if gross_loss > 0 else float("inf")

    years = (n - warmup) / (bars_per_day * spec.days_per_year)
    ann_ret = ((final / capital) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else 0
    perf = performance(eq_series, spec.days_per_year)
    dd = perf["max_drawdown_pct"]

    base.update({
        "total_return_pct": round((final - capital) / capital * 100, 2),
        "annual_return_pct": round(ann_ret, 2),
        "final_equity": round(final, 2),
        "n_trades": n_trades,
        "win_rate": round(len(wins) / n_trades * 100, 1),
        "profit_factor": round(pf, 3),
        "avg_win": round(float(np.mean(wins)), 2) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0,
        "max_drawdown_pct": dd,
        "calmar": round(ann_ret / abs(dd), 3) if dd != 0 else 0,
        "years": round(years, 2),
        "sharpe": perf["sharpe"],
        "ann_vol_pct": perf["ann_vol_pct"],
        "sortino": perf["sortino"],
        "exposure_pct": round(float(in_pos[warmup:].mean() * 100), 1),
    })
    return base
