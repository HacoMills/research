"""
第 2 层: 信号层
===============
每个信号只回答一件事: 第 i 根 K线收盘时, 该不该 开多 / 开空 / 平多 / 平空.
只能用第 i 根及以前的数据 (无前视). 仓位、成本、盯市都交给执行层 (engine.py).

内置信号 (参数单位都是"天", 会按 K线周期自动换算成根数):
  donchian:60,20   唐奇安通道 (海龟): 收盘价突破过去 60 天最高/最低价入场, 反向突破 20 天通道出场, 2N 止损
  tsmom:252        时间序列动量 (Moskowitz et al. 2012): 过去 252 天涨则做多, 跌则做空, 不设止损
  ma:50,200        均线交叉: 50 天均线在 200 天均线上方做多, 下方做空, 不设止损
  boll:20,2        布林带突破: 收盘价突破 20 天均线 ± 2 倍标准差入场, 回到均线出场, 2N 止损
  keltner:20,2     肯特纳通道: 突破 20 天 EMA ± 2 倍 ATR 入场, 回到 EMA 出场, 2N 止损
  supertrend:10,3  SuperTrend: 价格在 ATR(10 天) × 3 的跟踪线上方做多、下方做空, 穿线反手
  ma3:10,30,100    三均线: 快 > 中 > 慢 做多, 快 < 中 < 慢 做空; 快线穿回中线出场
  regress:60,2     回归趋势: 过去 60 天对数价格回归斜率的 t 值 > 2 做多, < -2 做空, 回到 0 出场
  riskmom:252,0.5  风险调整动量: 过去 252 天收益 ÷ 同期波动 > 0.5 做多, < -0.5 做空, 回到 0 出场
  knrp:9,2         KNRP (网上流传的组合指标): 随机指标 × RSI ÷ 动量, 其 2 根均线上升做多、下降做空
                   (第二个参数是均线根数, 不是天数; 信号翻转很频繁, 用来检验"网传策略")
  emavwap:200,0.5,60,40,3
                   EMA + VWAP + MFI: 收盘价突破 200 天 EMA ± 0.5% 缓冲, 且高于/低于 20 天滚动 VWAP,
                   MFI(14 天) > 60 / < 40, 大周期 EMA 同向, 最近 3 根连续上涨/下跌时入场; 跌回/涨回 EMA 出场, 2N 止损.
                   大周期: 日内K线用日线 EMA(200 天), 日线用周线 EMA(50 周), 只用已收完的大周期K线.
                   需要成交量和最高/最低价: 外汇 (无成交量) 和只有收盘价的 CSMAR 数据不产生信号.

多周期混合: 用 + 连接多个信号, 每个品种上各自独立跑, 日收益取平均 (相当于把资金平分给几个子策略)
  donchian:20,10 + donchian:60,20 + donchian:120,40 + tsmom:63 + tsmom:126 + tsmom:252

写法: "名称:参数1,参数2", 可附加 stop=N 改止损倍数 (stop=0 表示不止损), 例如
  donchian:20,10          系统1
  donchian:60,20,stop=3   放宽止损到 3N
  tsmom:126,stop=2        半年动量加 2N 止损

新增信号 (不用改框架): 在自己的策略文件夹里写 signals.py, 继承 Signal, 设置 name, 实现 compute();
配置文件里写 signal_module = "strategies/我的策略/signals.py", 框架会自动登记里面的信号.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class SignalArrays:
    long_entry: np.ndarray
    short_entry: np.ndarray
    long_exit: np.ndarray
    short_exit: np.ndarray
    warmup: int                 # 前多少根 K线数据不够, 不产生信号


class Signal:
    name = "base"
    default_stop: Optional[float] = None      # 默认止损倍数 (ATR 的倍数), None = 不止损

    def __init__(self, stop: Optional[float] = None):
        self.stop_mult = self.default_stop if stop is None else (stop or None)

    def compute(self, df: pd.DataFrame, bars_per_day: float) -> SignalArrays:
        raise NotImplementedError

    def label(self) -> str:
        raise NotImplementedError

    def with_params(self, *params):
        """同一种信号换一组参数 (稳定性检验里扫参数用)."""
        return type(self)(*params, stop=self.stop_mult if self.stop_mult else 0)


def _bars(days: float, bars_per_day: float) -> int:
    return max(int(round(days * bars_per_day)), 1)


class Donchian(Signal):
    """唐奇安通道突破 (海龟交易法)."""
    name = "donchian"
    default_stop = 2.0

    def __init__(self, entry_days: float = 60, exit_days: float = 20, stop: Optional[float] = None):
        super().__init__(stop)
        self.entry_days, self.exit_days = entry_days, exit_days

    def params(self):
        return (self.entry_days, self.exit_days)

    def label(self):
        s = f"donchian:{self.entry_days:g},{self.exit_days:g}"
        return s + (f",stop={self.stop_mult:g}" if self.stop_mult != self.default_stop else "")

    def compute(self, df, bars_per_day):
        eb, xb = _bars(self.entry_days, bars_per_day), _bars(self.exit_days, bars_per_day)
        high, low, close = df["high"], df["low"], df["close"]
        # shift(1): 通道只用过去 N 根, 不含当根 (无前视)
        eu, el = high.rolling(eb).max().shift(1).values, low.rolling(eb).min().shift(1).values
        xu, xl = high.rolling(xb).max().shift(1).values, low.rolling(xb).min().shift(1).values
        c = close.values
        with np.errstate(invalid="ignore"):
            return SignalArrays(c > eu, c < el, c < xl, c > xu, max(eb, xb))


class TSMOM(Signal):
    """时间序列动量: 看自己过去 N 天的涨跌."""
    name = "tsmom"
    default_stop = None

    def __init__(self, lookback_days: float = 252, stop: Optional[float] = None):
        super().__init__(stop)
        self.lookback_days = lookback_days

    def params(self):
        return (self.lookback_days,)

    def label(self):
        return f"tsmom:{self.lookback_days:g}" + (f",stop={self.stop_mult:g}" if self.stop_mult else "")

    def compute(self, df, bars_per_day):
        lb = _bars(self.lookback_days, bars_per_day)
        c = df["close"]
        # 复权价的变动点数 × scale / 真实价格 ≈ 真实涨跌幅; 没有复权列时就是普通涨跌幅
        if "scale" in df.columns and "raw_close" in df.columns:
            ret = ((c - c.shift(lb)) * df["scale"] / df["raw_close"].shift(lb)).values
        else:
            ret = (c / c.shift(lb) - 1).values
        with np.errstate(invalid="ignore"):
            return SignalArrays(ret > 0, ret < 0, ret <= 0, ret >= 0, lb)


class MACross(Signal):
    """均线交叉."""
    name = "ma"
    default_stop = None

    def __init__(self, fast_days: float = 50, slow_days: float = 200, stop: Optional[float] = None):
        super().__init__(stop)
        self.fast_days, self.slow_days = fast_days, slow_days

    def params(self):
        return (self.fast_days, self.slow_days)

    def label(self):
        return f"ma:{self.fast_days:g},{self.slow_days:g}" + (f",stop={self.stop_mult:g}" if self.stop_mult else "")

    def compute(self, df, bars_per_day):
        fb, sb = _bars(self.fast_days, bars_per_day), _bars(self.slow_days, bars_per_day)
        c = df["close"]
        f, s = c.rolling(fb).mean().values, c.rolling(sb).mean().values
        with np.errstate(invalid="ignore"):
            return SignalArrays(f > s, f < s, f <= s, f >= s, max(fb, sb))


# ──────────────────────────────────────────────────
# 更多信号 (开关式, 都只用当根及以前的数据)
# ──────────────────────────────────────────────────

def _atr(df, n):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def _bar_returns(df):
    """每根K线的真实收益率 (复权数据用 scale/raw_close 换算, 差值复权的期货也适用)."""
    c = df["close"]
    if "scale" in df.columns and "raw_close" in df.columns:
        return c.diff() * df["scale"] / df["raw_close"].shift(1)
    return c.pct_change()


def _period_return(df, n):
    c = df["close"]
    if "scale" in df.columns and "raw_close" in df.columns:
        return (c - c.shift(n)) * df["scale"] / df["raw_close"].shift(n)
    return c / c.shift(n) - 1


def _fmt(*xs):
    return ",".join(f"{x:g}" for x in xs)


class Bollinger(Signal):
    """布林带突破."""
    name = "boll"
    default_stop = 2.0

    def __init__(self, window_days: float = 20, width: float = 2.0, stop: Optional[float] = None):
        super().__init__(stop)
        self.window_days, self.width = window_days, width

    def label(self):
        return f"boll:{_fmt(self.window_days, self.width)}" + (f",stop={self.stop_mult:g}" if self.stop_mult != self.default_stop else "")

    def compute(self, df, bars_per_day):
        n = _bars(self.window_days, bars_per_day)
        c = df["close"]
        mid, sd = c.rolling(n).mean(), c.rolling(n).std()
        up, lo = (mid + self.width * sd).values, (mid - self.width * sd).values
        c, mid = c.values, mid.values
        with np.errstate(invalid="ignore"):
            return SignalArrays(c > up, c < lo, c < mid, c > mid, n)


class Keltner(Signal):
    """肯特纳通道: EMA ± k × ATR."""
    name = "keltner"
    default_stop = 2.0

    def __init__(self, window_days: float = 20, width: float = 2.0, stop: Optional[float] = None):
        super().__init__(stop)
        self.window_days, self.width = window_days, width

    def label(self):
        return f"keltner:{_fmt(self.window_days, self.width)}" + (f",stop={self.stop_mult:g}" if self.stop_mult != self.default_stop else "")

    def compute(self, df, bars_per_day):
        n = _bars(self.window_days, bars_per_day)
        ema = df["close"].ewm(span=n, adjust=False, min_periods=n).mean()
        atr = _atr(df, n)
        up, lo = (ema + self.width * atr).values, (ema - self.width * atr).values
        c, ema = df["close"].values, ema.values
        with np.errstate(invalid="ignore"):
            return SignalArrays(c > up, c < lo, c < ema, c > ema, n)


class SuperTrend(Signal):
    """SuperTrend: ATR 跟踪线, 价格穿线即反手, 一直在场."""
    name = "supertrend"
    default_stop = None

    def __init__(self, atr_days: float = 10, mult: float = 3.0, stop: Optional[float] = None):
        super().__init__(stop)
        self.atr_days, self.mult = atr_days, mult

    def label(self):
        return f"supertrend:{_fmt(self.atr_days, self.mult)}" + (f",stop={self.stop_mult:g}" if self.stop_mult else "")

    def compute(self, df, bars_per_day):
        n = _bars(self.atr_days, bars_per_day)
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        atr = _atr(df, n).values
        hl2 = (h + l) / 2
        ub, lb = hl2 + self.mult * atr, hl2 - self.mult * atr
        T = len(c)
        fub, flb = np.full(T, np.nan), np.full(T, np.nan)
        direction = np.zeros(T)
        for i in range(T):
            if np.isnan(atr[i]):
                continue
            if i == 0 or np.isnan(fub[i - 1]):
                fub[i], flb[i], direction[i] = ub[i], lb[i], 1
                continue
            fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
            flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
            if direction[i - 1] == 1:
                direction[i] = -1 if c[i] < flb[i] else 1
            else:
                direction[i] = 1 if c[i] > fub[i] else -1
        up, dn = direction == 1, direction == -1
        return SignalArrays(up, dn, dn, up, n + 1)


class MA3(Signal):
    """三均线排列."""
    name = "ma3"
    default_stop = None

    def __init__(self, fast_days: float = 10, mid_days: float = 30, slow_days: float = 100,
                 stop: Optional[float] = None):
        super().__init__(stop)
        self.fast_days, self.mid_days, self.slow_days = fast_days, mid_days, slow_days

    def label(self):
        return f"ma3:{_fmt(self.fast_days, self.mid_days, self.slow_days)}" + (f",stop={self.stop_mult:g}" if self.stop_mult else "")

    def compute(self, df, bars_per_day):
        fb, mb, sb = (_bars(d, bars_per_day) for d in (self.fast_days, self.mid_days, self.slow_days))
        c = df["close"]
        f, m, s = (c.rolling(b).mean().values for b in (fb, mb, sb))
        with np.errstate(invalid="ignore"):
            return SignalArrays((f > m) & (m > s), (f < m) & (m < s), f < m, f > m, max(fb, mb, sb))


class Regress(Signal):
    """回归趋势: 过去 N 天对数价格对时间回归, 用斜率的 t 值判断趋势是否显著."""
    name = "regress"
    default_stop = None

    def __init__(self, window_days: float = 60, t_entry: float = 2.0, stop: Optional[float] = None):
        super().__init__(stop)
        self.window_days, self.t_entry = window_days, t_entry

    def label(self):
        return f"regress:{_fmt(self.window_days, self.t_entry)}" + (f",stop={self.stop_mult:g}" if self.stop_mult else "")

    def compute(self, df, bars_per_day):
        n = max(_bars(self.window_days, bars_per_day), 5)
        r = _bar_returns(df).fillna(0)
        y = np.log1p(r).cumsum()                         # 用真实收益累积的对数价格 (复权方式不影响)
        x = pd.Series(np.arange(len(y), dtype=float), index=y.index)
        vx, vy = x.rolling(n).var(), y.rolling(n).var()
        slope = y.rolling(n).cov(x) / vx
        r2 = (slope ** 2 * vx / vy).clip(upper=1 - 1e-12)
        resid_var = vy * (1 - r2) * (n - 1) / (n - 2)
        t = (slope / np.sqrt(resid_var / ((n - 1) * vx))).values
        k = self.t_entry
        with np.errstate(invalid="ignore"):
            return SignalArrays(t > k, t < -k, t < 0, t > 0, n + 1)


class RiskMom(Signal):
    """风险调整动量: 过去 N 天收益 ÷ 同期波动 (相当于一个 z 值)."""
    name = "riskmom"
    default_stop = None

    def __init__(self, lookback_days: float = 252, z_entry: float = 0.5, stop: Optional[float] = None):
        super().__init__(stop)
        self.lookback_days, self.z_entry = lookback_days, z_entry

    def label(self):
        return f"riskmom:{_fmt(self.lookback_days, self.z_entry)}" + (f",stop={self.stop_mult:g}" if self.stop_mult else "")

    def compute(self, df, bars_per_day):
        n = _bars(self.lookback_days, bars_per_day)
        ret = _period_return(df, n)
        vol = _bar_returns(df).rolling(n).std() * np.sqrt(n)
        z = (ret / vol).replace([np.inf, -np.inf], np.nan).values
        k = self.z_entry
        with np.errstate(invalid="ignore"):
            return SignalArrays(z > k, z < -k, z <= 0, z >= 0, n + 1)


class KNRP(Signal):
    """网上流传的 KNRP: 随机指标 × RSI ÷ 动量 (比率形式, 约 100), 再取 smooth 根均线, 上升做多下降做空."""
    name = "knrp"
    default_stop = None

    def __init__(self, length_days: float = 9, smooth_bars: float = 2, stop: Optional[float] = None):
        super().__init__(stop)
        self.length_days, self.smooth_bars = length_days, smooth_bars

    def label(self):
        return f"knrp:{_fmt(self.length_days, self.smooth_bars)}" + (f",stop={self.stop_mult:g}" if self.stop_mult else "")

    def compute(self, df, bars_per_day):
        n = max(_bars(self.length_days, bars_per_day), 2)
        m = max(int(self.smooth_bars), 1)
        h, l, c = df["high"], df["low"], df["close"]
        d = c.diff()
        gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        hh, ll = h.rolling(n).max(), l.rolling(n).min()
        stoch = 100 * (c - ll) / (hh - ll)
        mom = 100 * (1 + _period_return(df, n))            # 比率形式的动量, 避免除以 0
        knrp = (stoch * rsi / mom).replace([np.inf, -np.inf], np.nan)
        avg = knrp.rolling(m).mean()
        up, dn = (avg > avg.shift(1)).values, (avg < avg.shift(1)).values
        return SignalArrays(up, dn, dn, up, n + m + 1)


class EmaVwapMfi(Signal):
    """200 天 EMA + 滚动 VWAP + MFI + 大周期 EMA + 连续涨跌. 需要成交量和真实的最高/最低价."""
    name = "emavwap"
    default_stop = 2.0

    def __init__(self, ema_days: float = 200, buffer_pct: float = 0.5, mfi_buy: float = 60,
                 mfi_sell: float = 40, consec: float = 3, vwap_days: float = 20, mfi_days: float = 14,
                 htf_daily_days: float = 200, htf_weeks: float = 50, stop: Optional[float] = None):
        super().__init__(stop)
        self.ema_days, self.buffer_pct, self.mfi_buy, self.mfi_sell = ema_days, buffer_pct, mfi_buy, mfi_sell
        self.consec, self.vwap_days, self.mfi_days = int(consec), vwap_days, mfi_days
        self.htf_daily_days, self.htf_weeks = htf_daily_days, htf_weeks

    def label(self):
        s = f"emavwap:{_fmt(self.ema_days, self.buffer_pct, self.mfi_buy, self.mfi_sell, self.consec)}"
        return s + (f",stop={self.stop_mult:g}" if self.stop_mult != self.default_stop else "")

    def _htf_ema(self, df, bars_per_day):
        """大周期 EMA, 只用已经收完的大周期K线, 对齐到当前周期的每根K线."""
        c = df["close"]
        if bars_per_day > 1:                               # 日内K线 → 日线 EMA
            hc = c.resample("1D").last().dropna()
            ema = hc.ewm(span=int(self.htf_daily_days), adjust=False, min_periods=int(self.htf_daily_days)).mean()
            avail = ema.copy(); avail.index = avail.index + pd.Timedelta(days=1)
        else:                                              # 日线 → 周线 EMA
            hc = c.resample("W-SUN").last().dropna()
            ema = hc.ewm(span=int(self.htf_weeks), adjust=False, min_periods=int(self.htf_weeks)).mean()
            avail = ema.copy(); avail.index = avail.index + pd.Timedelta(days=1)
        return avail.reindex(avail.index.union(df.index)).ffill().reindex(df.index)

    def compute(self, df, bars_per_day):
        n_ema = _bars(self.ema_days, bars_per_day)
        T = len(df)
        no = np.zeros(T, dtype=bool)
        vol = df["volume"] if "volume" in df.columns else None
        if vol is None or vol.fillna(0).sum() <= 0 or (df["high"] == df["low"]).mean() > 0.5:
            return SignalArrays(no, no, no, no, T)          # 没有成交量或只有收盘价: 不产生信号
        h, l, c = df["high"], df["low"], df["close"]
        ema = c.ewm(span=n_ema, adjust=False, min_periods=n_ema).mean()
        up_b, lo_b = ema * (1 + self.buffer_pct / 100), ema * (1 - self.buffer_pct / 100)
        tp = (h + l + c) / 3
        nv = _bars(self.vwap_days, bars_per_day)
        vwap = (tp * vol).rolling(nv).sum() / vol.rolling(nv).sum()
        nm = _bars(self.mfi_days, bars_per_day)
        flow = tp * vol
        pos = flow.where(tp > tp.shift(1), 0.0).rolling(nm).sum()
        neg = flow.where(tp < tp.shift(1), 0.0).rolling(nm).sum()
        mfi = 100 - 100 / (1 + pos / neg)
        htf = self._htf_ema(df, bars_per_day)
        k = self.consec
        ups = (c.diff() > 0).astype(float).rolling(k).sum() == k
        dns = (c.diff() < 0).astype(float).rolling(k).sum() == k
        le = (c > up_b) & (c > vwap) & (mfi > self.mfi_buy) & (c > htf) & ups
        se = (c < lo_b) & (c < vwap) & (mfi < self.mfi_sell) & (c < htf) & dns
        return SignalArrays(le.fillna(False).values, se.fillna(False).values,
                            (c < ema).values, (c > ema).values, max(n_ema, nv, nm) + 1)



class Blend(Signal):
    """
    多周期 / 多信号混合. 不直接产生买卖信号: 组合层 (portfolio.Runner) 在每个品种上把每个子信号
    各跑一遍, 再把日收益等权平均. 从最慢的子信号开始有收益的那天起算, 保证每天都是全部子信号的平均.
    """
    name = "blend"

    def __init__(self, parts):
        super().__init__(None)
        self.parts = list(parts)

    def label(self):
        return " + ".join(p.label() for p in self.parts)

    def compute(self, df, bars_per_day):
        raise TypeError("混合信号由组合层分别运行各个子信号, 不能直接交给引擎")


SIGNALS: Dict[str, type] = {
    "donchian": Donchian, "tsmom": TSMOM, "ma": MACross,
    "boll": Bollinger, "keltner": Keltner, "supertrend": SuperTrend, "ma3": MA3,
    "regress": Regress, "riskmom": RiskMom, "knrp": KNRP, "emavwap": EmaVwapMfi,
}


def load_modules(modules) -> list:
    """
    加载用户自己写的信号 / 过滤器模块, 把里面的 Signal / Filter 子类按 name 登记.
    modules: 一个或多个, 可以是文件路径 "strategies/xxx/signals.py" 或模块名 "strategies.xxx.signals".
    返回新登记的名字.
    """
    import importlib
    import importlib.util
    import inspect
    from pathlib import Path
    from .filters import FILTERS, Filter

    if not modules:
        return []
    if isinstance(modules, str):
        modules = [modules]
    root = Path(__file__).resolve().parent.parent            # quant 根目录
    added = []
    for m in modules:
        if m.endswith(".py"):
            path = Path(m) if Path(m).is_absolute() else root / m
            if not path.exists():
                raise FileNotFoundError(f"找不到信号模块 {path}")
            spec = importlib.util.spec_from_file_location(f"user_signals_{path.parent.name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(m)
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, Signal) and obj is not Signal and obj.name != "base":
                SIGNALS[obj.name] = obj
                added.append(obj.name)
            elif issubclass(obj, Filter) and obj is not Filter and obj.name not in ("none",):
                FILTERS[obj.name] = obj
                added.append(obj.name)
    return sorted(set(added))


def parse_signal(spec: str) -> Signal:
    """'donchian:60,20' / 'tsmom:252' / 'ma:50,200,stop=2' / 'a + b + c' (混合) → Signal 实例."""
    if "+" in spec:
        return Blend(parse_signal(p) for p in spec.split("+") if p.strip())
    name, _, rest = spec.strip().partition(":")
    name = name.lower()
    if name not in SIGNALS:
        raise ValueError(f"未知信号 {name!r}, 可选: {', '.join(SIGNALS)}")
    args, kwargs = [], {}
    for part in filter(None, (p.strip() for p in rest.split(","))):
        if "=" in part:
            k, v = part.split("=", 1)
            kwargs[k.strip()] = float(v)
        else:
            args.append(float(part))
    return SIGNALS[name](*args, **kwargs)
