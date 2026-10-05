"""
第 3 层: 过滤层
===============
过滤器只决定"第 i 根 K线收盘时, 允不允许开新仓". 不影响已有仓位的平仓.
返回与 K线对齐的布尔 Series (True = 允许入场), 只能用第 i 根及以前已知的信息.

内置过滤器:
  none                      不过滤
  garch                     GARCH(1,1) 条件波动率高于过去 20 天均值时才入场 (滚动估计, 无前视)
  garch:20,refit=21         自定义均值窗口 (天) 和重估间隔 (天)
  vix:low,80                VIX 低于过去一年的 80% 分位时才入场 (避开恐慌期)
  vix:high,50               VIX 高于过去一年的中位数时才入场 (只在波动大时做趋势)
  vix:low,80,source=us      指定指数: us (美国 VIX) / qvix50 / qvix300 (中国期权波动率指数)
  vix:low,80,window=504     分位数用过去 504 天 (默认 252)
  vix:low,level=20          用固定点位: VIX 低于 20 才入场 (不用分位数)

VIX 数据源 (source=auto, 默认):
  中国期货、A股 → 中国 QVIX (50ETF, 2015 年起), 与国内收盘同时可得, 当天可用
  汇率、加密   → 美国 VIX, 美股收盘在北京时间凌晨, 只用前一天及以前的值 (滞后一天)
在 VIX 数据开始之前的日子不过滤.

⚠️ 趋势策略在危机时往往赚得最多 (smile), "VIX 高时不入场"可能适得其反, 要用稳定性检验去验证.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class FilterContext:
    market: str             # crypto / future / stock / fx
    symbol: str
    days_per_year: float


class Filter:
    name = "none"

    def allow(self, df: pd.DataFrame, bars_per_day: float, ctx: FilterContext) -> Optional[pd.Series]:
        return None          # None = 不过滤

    def label(self) -> str:
        return "none"


class NoFilter(Filter):
    pass


# ──────────────────────────────────────────────────
# GARCH
# ──────────────────────────────────────────────────

def garch_rolling_filter(df: pd.DataFrame, lookback: int, min_train: int, refit_every: int,
                         verbose: bool = False) -> Optional[pd.Series]:
    """
    滚动估计的 GARCH(1,1) 过滤 (原 turtle_timeframe_backtest.compute_garch_filter_rolling).
    前 min_train 根只训练不过滤; 之后每隔 refit_every 根只用此前的收益重估参数,
    两次重估之间按 σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} 递推. σ_t > 过去 lookback 根均值 → 允许入场.
    """
    try:
        from arch import arch_model
    except ImportError:
        print("  [!] 需要安装 arch 库: pip install arch")
        return None

    r = (np.log(df["close"] / df["close"].shift(1)) * 100).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(r)
    if n < min_train + lookback:
        return None
    rv = r.to_numpy()
    sigma2 = np.full(n, np.nan)
    params, s2_prev = None, None
    for start in range(min_train, n, refit_every):
        try:
            res = arch_model(r.iloc[:start], vol="Garch", p=1, q=1, mean="Zero",
                             rescale=False).fit(disp="off", show_warning=False)
            params = (res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"])
            if s2_prev is None:
                s2_prev = float(res.conditional_volatility.iloc[-1]) ** 2
        except Exception:
            if params is None:
                continue
        omega, alpha, beta = params
        for t in range(start, min(start + refit_every, n)):
            s2_prev = omega + alpha * rv[t - 1] ** 2 + beta * s2_prev
            sigma2[t] = s2_prev

    cond_vol = pd.Series(np.sqrt(sigma2), index=r.index)
    vol_ma = cond_vol.rolling(lookback).mean()
    valid = cond_vol.notna() & vol_ma.notna()
    out = pd.Series(True, index=df.index)
    out.loc[valid[valid].index] = (cond_vol > vol_ma)[valid].values
    return out


class GarchFilter(Filter):
    name = "garch"

    def __init__(self, lookback_days: float = 20, refit: float = 21):
        self.lookback_days, self.refit_days = lookback_days, refit

    def label(self):
        return "garch" if (self.lookback_days, self.refit_days) == (20, 21) else \
            f"garch:{self.lookback_days:g},refit={self.refit_days:g}"

    def allow(self, df, bars_per_day, ctx):
        return garch_rolling_filter(
            df, lookback=max(int(self.lookback_days * bars_per_day), 20),
            min_train=int(ctx.days_per_year * bars_per_day),
            refit_every=max(int(self.refit_days * bars_per_day), 1))


# ──────────────────────────────────────────────────
# VIX
# ──────────────────────────────────────────────────

_VIX_CACHE = {}


def _get_vix(source: str) -> pd.Series:
    if source not in _VIX_CACHE:
        from data import market_data
        s = {"us": lambda: market_data.load_us_vix(),
             "qvix50": lambda: market_data.load_qvix("50etf"),
             "qvix300": lambda: market_data.load_qvix("300etf")}[source]()
        _VIX_CACHE[source] = s.sort_index()
    return _VIX_CACHE[source]


class VixFilter(Filter):
    name = "vix"

    def __init__(self, mode: str = "low", pct: float = 80, source: str = "auto", window: float = 252,
                 level: float = None):
        if mode not in ("low", "high"):
            raise ValueError("vix 过滤的方向只能是 low 或 high")
        self.mode, self.pct, self.source, self.window = mode, float(pct), source, int(window)
        self.level = float(level) if level is not None else None

    def label(self):
        if self.level is not None:
            s = f"vix:{self.mode},level={self.level:g}"
            return s + (f",source={self.source}" if self.source != "auto" else "")
        s = f"vix:{self.mode},{self.pct:g}"
        if self.source != "auto":
            s += f",source={self.source}"
        if self.window != 252:
            s += f",window={self.window}"
        return s

    def resolve_source(self, market: str) -> str:
        if self.source != "auto":
            return self.source
        return "qvix50" if market in ("future", "stock") else "us"

    def allow(self, df, bars_per_day, ctx):
        src = self.resolve_source(ctx.market)
        vix = _get_vix(src)
        if vix.empty:
            print(f"  [!] 拿不到 {src} 数据, {ctx.symbol} 不做 VIX 过滤")
            return None
        # 分位阈值只用过去 window 个交易日 (含当天已知的值)
        if self.level is not None:                        # 固定点位
            thr = pd.Series(self.level, index=vix.index)
        else:
            thr = vix.rolling(self.window, min_periods=self.window // 2).quantile(self.pct / 100)
        ok = (vix < thr) if self.mode == "low" else (vix > thr)
        ok = ok.where(thr.notna())                        # 阈值还算不出来的日子: 不过滤

        # 对齐到 K线: 美国 VIX 只能用"日期早于 K线日期"的值 (滞后一天); QVIX 可用当天
        bar_dates = pd.DatetimeIndex(df.index.normalize())
        lag_days = 1 if src == "us" else 0
        lookup = pd.Series(ok.values, index=ok.index.normalize() + pd.Timedelta(days=lag_days))
        lookup = lookup[~lookup.index.duplicated(keep="last")].sort_index()
        aligned = lookup.reindex(bar_dates, method="ffill")
        aligned.index = df.index
        return aligned.fillna(True).astype(bool)          # VIX 数据开始前不过滤


FILTERS = {"none": NoFilter, "garch": GarchFilter, "vix": VixFilter}


def parse_filter(spec: str) -> Filter:
    """'none' / 'garch' / 'vix:low,80' / 'vix:high,50,source=us' → Filter 实例."""
    name, _, rest = (spec or "none").strip().partition(":")
    name = name.lower()
    if name not in FILTERS:
        raise ValueError(f"未知过滤器 {name!r}, 可选: {', '.join(FILTERS)}")
    args, kwargs = [], {}
    for part in filter(None, (p.strip() for p in rest.split(","))):
        if "=" in part:
            k, v = part.split("=", 1)
            v = v.strip()
            kwargs[k.strip()] = float(v) if v.replace(".", "", 1).isdigit() else v
        else:
            args.append(part if not part.replace(".", "", 1).isdigit() else float(part))
    return FILTERS[name](*args, **kwargs)
