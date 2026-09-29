"""
新策略模板: 复制整个 _template 文件夹, 改名 (用英文, 如 bollinger), 再改下面的信号.

信号只回答一件事: 第 i 根 K线收盘时, 该不该 开多 / 开空 / 平多 / 平空.
只能用第 i 根及以前的数据 (rolling 之类的计算天然满足; 用到"未来"的 shift(-1) 绝对不行).
仓位、止损执行、手续费、滑点、组合、报告都由框架负责.

配置文件里写:
    [strategy]
    signal_module = "strategies/_template/signals.py"
    signal = "boll:20,2"
"""

import numpy as np

from backtest.signals import Signal, SignalArrays, _bars


class Bollinger(Signal):
    """布林带突破: 收盘价突破上轨做多, 跌破下轨做空, 回到中轨平仓."""
    name = "boll"                 # 配置里用的名字
    default_stop = 2.0            # 默认 2N 止损; None = 不止损

    def __init__(self, window_days: float = 20, width: float = 2.0, stop=None):
        super().__init__(stop)
        self.window_days, self.width = window_days, width

    def label(self):
        return f"boll:{self.window_days:g},{self.width:g}"

    def compute(self, df, bars_per_day):
        n = _bars(self.window_days, bars_per_day)        # 天数 → K线根数
        c = df["close"]
        mid = c.rolling(n).mean()
        sd = c.rolling(n).std()
        upper, lower = (mid + self.width * sd).values, (mid - self.width * sd).values
        c, mid = c.values, mid.values
        with np.errstate(invalid="ignore"):
            return SignalArrays(
                long_entry=c > upper,      # 空仓时: 开多
                short_entry=c < lower,     # 空仓时: 开空
                long_exit=c < mid,         # 持多时: 平仓
                short_exit=c > mid,        # 持空时: 平仓
                warmup=n,                  # 前 n 根数据不够, 不产生信号
            )
