"""
波动率突破策略（Volatility Breakout Strategy）——最终版执行逻辑。

策略思路：不预测方向，只用模型判断"接下来会不会有大波动"（prob_breakout_v5
超过阈值），一旦真实价格先触碰上/下屏障中的任意一个，就跟随那个方向开仓，
之后用固定止盈 + 移动止损（trailing stop）管理仓位。

这版 run_backtest 是原始 notebook 里经过好几轮修正后的最终形态，修正过
两个曾经让回测结果失真的问题：
1. 用 high/low 而不是 close 判断是否真的触碰了屏障/止盈/止损——只用收盘价
   判断会系统性低估盘中触发止盈止损的次数，让回测比真实交易更"顺"。
2. 进出场价格都额外扣一道 touch_slippage（触碰屏障那一刻的滑点），
   比只在下单成交时扣一次滑点更贴近真实的挂单/触发机制。

注：原 notebook 里这版逻辑是在一个叫 VolatilityBreakoutStrategy_Mid_Fixed
的类里、通过继承一个只存在于当时 Jupyter 内核内存、没有被保存进 .ipynb
文件的父类 VolatilityBreakoutStrategy_Mid 实现的——这也是原 notebook
"探索了太多版本、没有清晰保留最终版"这个问题的一个具体例子。这里把它
和最早的干净基类（notebook 里标注为"核心策略"的版本）合并成一个自洽、
可独立运行的类。
"""
import numpy as np
import pandas as pd

from . import config


class VolatilityBreakoutStrategy:
    def __init__(
        self,
        prob_threshold: float = 0.6,
        max_hold: int = 48,
        signal_window: int = 12,
        upper_mult: float = 0.8,
        lower_mult: float = 0.8,
        tp_mult: float = 3.0,
        trail_mult: float = 1.2,
        fee_rate: float = 0.0005,
        slippage_bps: float = 0.0002,
        touch_slippage: float = 0.0005,
        target_risk: float = 0.01,
        max_leverage: float = 1.5,
    ):
        # 策略核心参数
        self.prob_threshold = prob_threshold  # 模型预测"会突破"的概率阈值
        self.max_hold = max_hold              # 最大持仓K线数（时间止损）
        self.signal_window = signal_window    # 等待突破的时间窗口

        # 屏障/止盈止损乘数（都是相对 GARCH 波动率的倍数）
        self.upper_mult = upper_mult
        self.lower_mult = lower_mult
        self.tp_mult = tp_mult
        self.trail_mult = trail_mult

        # 交易摩擦成本
        self.fee_rate = fee_rate              # 单边手续费
        self.slippage = slippage_bps          # 下单成交滑点
        self.touch_slippage = touch_slippage  # 触碰屏障/止盈止损那一刻的额外滑点

        # 风险管理
        self.target_risk = target_risk        # 单笔交易目标风险敞口（占总资金比例）
        self.max_leverage = max_leverage       # 最大允许杠杆

    def calculate_actual_execution_price(self, theoretical_price, direction, is_entry):
        """滑点永远让实际成交价比理论价更差：买入更贵，卖出更便宜。"""
        is_buying = (direction == 1 and is_entry) or (direction == -1 and not is_entry)
        if is_buying:
            return theoretical_price * (1 + self.slippage)
        return theoretical_price * (1 - self.slippage)

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行历史回测，返回逐笔交易明细 DataFrame（entry_time, exit_time,
        direction, exit_reason, hold_bars, net_pnl）。

        Args:
            df: 需包含 close/high/low、garch_var_oos、prob_breakout_v5 几列，
                通常是 features.create_all_features + labeling.apply_breakout_labels
                + model.walk_forward_xgboost 拼接之后的最终数据集。
        """
        if "high" not in df.columns or "low" not in df.columns:
            raise ValueError("数据中缺少 high/low 列，无法用真实盘中触碰判断出入场")

        close_arr = df["close"].values
        high_arr = df["high"].values
        low_arr = df["low"].values
        garch_var_arr = df["garch_var_oos"].abs().values
        prob_arr = df["prob_breakout_v5"].values
        index_arr = df.index
        n = len(df)

        trades = []
        i = 0

        while i < n - self.signal_window - self.max_hold - 1:
            if prob_arr[i] <= self.prob_threshold:
                i += 1
                continue

            p0 = close_arr[i]
            current_vol = garch_var_arr[i]

            upper_barrier = p0 * (1 + current_vol * self.upper_mult)
            lower_barrier = p0 * (1 - current_vol * self.lower_mult)

            path_high = high_arr[i + 1 : i + 1 + self.signal_window]
            path_low = low_arr[i + 1 : i + 1 + self.signal_window]

            hit_upper = np.where(path_high >= upper_barrier)[0]
            hit_lower = np.where(path_low <= lower_barrier)[0]
            idx_upper = hit_upper[0] if len(hit_upper) > 0 else np.inf
            idx_lower = hit_lower[0] if len(hit_lower) > 0 else np.inf

            position_size = min(self.target_risk / current_vol, self.max_leverage)

            if idx_upper < idx_lower:
                direction = 1
                entry_idx = i + 1 + int(idx_upper)
                theoretical_entry = upper_barrier * (1 + self.touch_slippage)
            elif idx_lower < idx_upper:
                direction = -1
                entry_idx = i + 1 + int(idx_lower)
                theoretical_entry = lower_barrier * (1 - self.touch_slippage)
            else:
                i += 1  # 信号窗口内没有真实触碰，挂单失效
                continue

            actual_entry = self.calculate_actual_execution_price(theoretical_entry, direction, is_entry=True)

            if direction == 1:
                tp_price = actual_entry * (1 + current_vol * self.tp_mult)
                current_sl = actual_entry * (1 - current_vol * self.trail_mult)
                highest_seen = actual_entry
            else:
                tp_price = actual_entry * (1 - current_vol * self.tp_mult)
                current_sl = actual_entry * (1 + current_vol * self.trail_mult)
                lowest_seen = actual_entry

            exit_idx = min(entry_idx + self.max_hold, n - 1)
            theoretical_exit = close_arr[exit_idx]
            exit_reason = "Max Hold Time limit"

            for j in range(entry_idx + 1, entry_idx + self.max_hold + 1):
                if j >= n:
                    break
                current_high = high_arr[j]
                current_low = low_arr[j]

                if direction == 1:
                    if current_high > highest_seen:
                        highest_seen = current_high
                        current_sl = max(current_sl, highest_seen * (1 - current_vol * self.trail_mult))
                    if current_high >= tp_price:
                        exit_idx, theoretical_exit, exit_reason = j, tp_price * (1 + self.touch_slippage), "Take Profit"
                        break
                    if current_low <= current_sl:
                        exit_idx, theoretical_exit, exit_reason = j, current_sl * (1 - self.touch_slippage), "Trailing Stop"
                        break
                else:
                    if current_low < lowest_seen:
                        lowest_seen = current_low
                        current_sl = min(current_sl, lowest_seen * (1 + current_vol * self.trail_mult))
                    if current_low <= tp_price:
                        exit_idx, theoretical_exit, exit_reason = j, tp_price * (1 - self.touch_slippage), "Take Profit"
                        break
                    if current_high >= current_sl:
                        exit_idx, theoretical_exit, exit_reason = j, current_sl * (1 + self.touch_slippage), "Trailing Stop"
                        break

            actual_exit = self.calculate_actual_execution_price(theoretical_exit, direction, is_entry=False)

            if direction == 1:
                raw_return = (actual_exit - actual_entry) / actual_entry
            else:
                raw_return = (actual_entry - actual_exit) / actual_entry

            pnl = raw_return * position_size
            fee = position_size * self.fee_rate * 2
            net_pnl = pnl - fee

            trades.append(
                {
                    "entry_time": index_arr[entry_idx],
                    "exit_time": index_arr[exit_idx],
                    "direction": direction,
                    "exit_reason": exit_reason,
                    "hold_bars": exit_idx - entry_idx,
                    "net_pnl": net_pnl,
                }
            )
            i = exit_idx

        return pd.DataFrame(trades)


def build_default_strategy() -> VolatilityBreakoutStrategy:
    """用 config.STRATEGY_DEFAULTS 里记录的、notebook 里最终选定的参数组合构建策略实例。"""
    return VolatilityBreakoutStrategy(**config.STRATEGY_DEFAULTS)
