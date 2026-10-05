"""
第 4 层补充: 和已有资产组合 (如沪深300)
======================================
回答"这个策略有没有用": 单独持有股票, 和"股票 + 一部分策略"比, 夏普和回撤有没有改善.

两种组合方式:
  配置 (weights)    总资金按比例分: 股票 (1-w) + 策略 w, 每天再平衡到固定比例
  叠加 (overlays)   股票 100% 不变, 再用保证金叠加 k 倍的策略 (期货只占用少量保证金, 实际可行)
策略收益用报告主组合 (有目标波动时是缩放后的).
股票休市、策略有收益的日子 (如只有外汇交易), 股票收益记 0.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _stats(r: pd.Series, dpy: float, stock: pd.Series) -> Dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / dpy
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else 0
    vol = r.std() * math.sqrt(dpy)
    dd = (eq / eq.cummax() - 1).min()
    m = (1 + r).resample("ME").prod() - 1
    sm = (1 + stock.reindex(r.index).fillna(0)).resample("ME").prod() - 1
    down = sm < 0
    return {"年化收益%": cagr * 100, "年化波动%": vol * 100,
            "夏普": r.mean() * dpy / vol if vol > 0 else np.nan,
            "最大回撤%": dd * 100, "Calmar": cagr / abs(dd) if dd < 0 else np.nan,
            "股票下跌月份平均月收益%": m[down].mean() * 100 if down.any() else np.nan,
            "最差月份%": m.min() * 100}


def analyze(strategy: pd.Series, stock_px: pd.Series, dpy: float,
            weights: List[float] = (0, 0.1, 0.2, 0.3, 0.5), overlays: List[float] = (0.5, 1.0)) -> Optional[Dict]:
    """strategy: 策略日收益; stock_px: 股票 (指数/ETF) 日收盘."""
    stock = stock_px.sort_index().pct_change().dropna()
    start = max(strategy.index[0], stock.index[0])
    idx = strategy.index[strategy.index >= start]
    if len(idx) < 250:
        return None
    s = stock.reindex(idx).fillna(0.0)
    t = strategy.reindex(idx).fillna(0.0)

    rows, curves = [], {}
    for w in weights:
        name = "股票 100%" if w == 0 else f"股票 {1-w:.0%} + 策略 {w:.0%}"
        r = (1 - w) * s + w * t
        rows.append({"组合": name, **_stats(r, dpy, s)})
        curves[name] = r
    for k in overlays:
        name = f"股票 100% + 叠加策略 {k:.0%}"
        r = s + k * t
        rows.append({"组合": name, **_stats(r, dpy, s)})
        curves[name] = r
    rows.append({"组合": "策略 100%", **_stats(t, dpy, s)})
    curves["策略 100%"] = t

    # 相关性: 日度 (受收盘时间错位影响) 和月度
    mt = (1 + t).resample("ME").prod() - 1
    ms = (1 + s).resample("ME").prod() - 1
    yr = pd.DataFrame({"沪深300": (1 + s).groupby(s.index.year).prod() - 1,
                       "策略": (1 + t).groupby(t.index.year).prod() - 1})
    for w in weights[1:]:
        yr[f"股票{1-w:.0%}+策略{w:.0%}"] = (1 + curves[f"股票 {1-w:.0%} + 策略 {w:.0%}"]).groupby(t.index.year).prod() - 1
    return {"table": pd.DataFrame(rows), "curves": curves, "start": idx[0], "end": idx[-1],
            "corr_daily": float(s.corr(t)), "corr_monthly": float(ms.corr(mt)),
            "yearly": yr, "source": stock_px.attrs.get("source", "")}
