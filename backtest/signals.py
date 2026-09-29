"""
第 2 层: 信号层
===============
每个信号只回答一件事: 第 i 根 K线收盘时, 该不该 开多 / 开空 / 平多 / 平空.
只能用第 i 根及以前的数据 (无前视). 仓位、成本、盯市都交给执行层 (engine.py).

内置信号 (参数单位都是"天", 会按 K线周期自动换算成根数):
  donchian:60,20   唐奇安通道 (海龟): 收盘价突破过去 60 天最高/最低价入场, 反向突破 20 天通道出场, 2N 止损
  tsmom:252        时间序列动量 (Moskowitz et al. 2012): 过去 252 天涨则做多, 跌则做空, 不设止损
  ma:50,200        均线交叉: 50 天均线在 200 天均线上方做多, 下方做空, 不设止损

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


SIGNALS: Dict[str, type] = {"donchian": Donchian, "tsmom": TSMOM, "ma": MACross}


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
