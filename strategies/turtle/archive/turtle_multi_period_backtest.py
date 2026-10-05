#!/usr/bin/env python3
"""
海龟交易系统 — 多周期回测 & E-ratio 信号质量分析
=================================================
数据: BTC/USDT 15分钟 K线 (2022-01 ~ 2026-07)
测试入场通道周期: 10, 15, 20, 30, 40, 60 天
  （每天 = 96 根 15min K线）

对每个入场周期:
  1. E-ratio 信号质量 (MFE / MAE 各观察窗口)
  2. 完整回测: ATR仓位管理 + 2N止损 + 反向通道出场
  3. 汇总对比表 + 图表
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, sys, os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────

BARS_PER_DAY = 96                 # 15min K线
INITIAL_CAPITAL = 10_000.0        # 初始资金 (USDT)
RISK_PCT = 0.01                   # 单笔风险 1%
STOP_MULT = 2.0                   # 止损 = 2×ATR
ATR_PERIOD_DAYS = 20              # ATR 周期 (天)
COMMISSION_PCT = 0.05 / 100       # OKX taker 手续费

# 入场通道周期 (天数)
ENTRY_PERIODS_DAYS = [10, 15, 20, 30, 40, 60]

# 出场通道 = 入场通道的一半 (海龟规则: 系统1=20/10, 系统2=60/20)
def exit_period_days(entry_days: int) -> int:
    return max(entry_days // 2, 5)

# E-ratio 观察窗口 (根 K线)
E_RATIO_WINDOWS = [1, 4, 8, 16, 32, 64, 96]  # 15min~1天


# ──────────────────────────────────────────────────
# 数据加载 & 指标计算
# ──────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"加载数据: {len(df)} 行, "
          f"{df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
    return df


def calc_atr(df: pd.DataFrame, period_bars: int) -> pd.Series:
    """Wilder 平滑 ATR"""
    h = df["high"]
    l = df["low"]
    c = df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period_bars, min_periods=period_bars, adjust=False).mean()


def calc_donchian(df: pd.DataFrame, period_bars: int):
    """唐奇安通道 (用已完成K线, shift(1))"""
    upper = df["high"].rolling(period_bars).max().shift(1)
    lower = df["low"].rolling(period_bars).min().shift(1)
    return upper, lower


# ──────────────────────────────────────────────────
# E-ratio 计算
# ──────────────────────────────────────────────────

def calc_e_ratio(df: pd.DataFrame, entry_bars: int, atr: pd.Series,
                 windows: List[int]) -> Dict:
    """
    E-ratio = mean(MFE/ATR) / mean(MAE/ATR) 对每个观察窗口
    MFE = Maximum Favorable Excursion (入场后最大浮盈)
    MAE = Maximum Adverse Excursion (入场后最大浮亏)
    """
    entry_upper, entry_lower = calc_donchian(df, entry_bars)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr_vals = atr.values
    n = len(df)

    # 找出所有突破信号
    signals = []  # (idx, direction, atr_at_entry)
    for i in range(entry_bars + 1, n):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue
        if not np.isnan(entry_upper.iloc[i]) and close[i] > entry_upper.iloc[i]:
            signals.append((i, 1, atr_vals[i]))   # 做多信号
        elif not np.isnan(entry_lower.iloc[i]) and close[i] < entry_lower.iloc[i]:
            signals.append((i, -1, atr_vals[i]))   # 做空信号

    # 对每个窗口计算 MFE/MAE
    results = {"windows": windows, "e_ratios": [], "avg_mfe": [], "avg_mae": [],
               "n_signals": len(signals), "n_long": 0, "n_short": 0}

    results["n_long"] = sum(1 for _, d, _ in signals if d == 1)
    results["n_short"] = sum(1 for _, d, _ in signals if d == -1)

    for w in windows:
        mfe_list = []
        mae_list = []
        for idx, direction, atr_e in signals:
            end = min(idx + w, n - 1)
            if end <= idx:
                continue
            if direction == 1:  # 多头
                mfe = (max(high[idx+1:end+1]) - close[idx]) / atr_e if end > idx else 0
                mae = (close[idx] - min(low[idx+1:end+1])) / atr_e if end > idx else 0
            else:  # 空头
                mfe = (close[idx] - min(low[idx+1:end+1])) / atr_e if end > idx else 0
                mae = (max(high[idx+1:end+1]) - close[idx]) / atr_e if end > idx else 0
            mfe_list.append(max(mfe, 0))
            mae_list.append(max(mae, 0))

        avg_mfe = np.mean(mfe_list) if mfe_list else 0
        avg_mae = np.mean(mae_list) if mae_list else 0
        e_ratio = avg_mfe / avg_mae if avg_mae > 0 else float("inf")

        results["e_ratios"].append(round(e_ratio, 4))
        results["avg_mfe"].append(round(avg_mfe, 4))
        results["avg_mae"].append(round(avg_mae, 4))

    return results


# ──────────────────────────────────────────────────
# 完整回测引擎
# ──────────────────────────────────────────────────

@dataclass
class Trade:
    entry_idx: int
    direction: int          # 1=多, -1=空
    entry_price: float
    qty: float
    stop_price: float
    exit_idx: int = -1
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


def run_backtest(df: pd.DataFrame, entry_bars: int, exit_bars: int,
                 atr: pd.Series) -> Dict:
    """
    完整海龟回测 (单Unit, 无金字塔)
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr_vals = atr.values
    n = len(df)

    entry_upper, entry_lower = calc_donchian(df, entry_bars)
    exit_upper, exit_lower = calc_donchian(df, exit_bars)

    equity = INITIAL_CAPITAL
    peak_equity = equity
    position = 0        # 1=多, -1=空, 0=空仓
    entry_price = 0.0
    stop_price = 0.0
    qty = 0.0
    trades: List[Trade] = []
    equity_curve = np.full(n, np.nan)
    current_trade = None

    warmup = max(entry_bars, exit_bars, ATR_PERIOD_DAYS * BARS_PER_DAY) + 1

    for i in range(warmup, n):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            equity_curve[i] = equity
            continue

        price = close[i]

        # ── 持仓中: 检查出场 ──
        if position != 0 and current_trade is not None:
            exit_signal = False
            exit_reason = ""

            if position == 1:
                # 多头: 止损 or 跌破出场通道
                if low[i] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "止损"
                    exit_signal = True
                elif not np.isnan(exit_lower.iloc[i]) and price < exit_lower.iloc[i]:
                    exit_price = price
                    exit_reason = "通道出场"
                    exit_signal = True
            else:
                # 空头: 止损 or 突破出场通道
                if high[i] >= stop_price:
                    exit_price = stop_price
                    exit_reason = "止损"
                    exit_signal = True
                elif not np.isnan(exit_upper.iloc[i]) and price > exit_upper.iloc[i]:
                    exit_price = price
                    exit_reason = "通道出场"
                    exit_signal = True

            if exit_signal:
                pnl = position * (exit_price - entry_price) * qty
                commission = (entry_price + exit_price) * qty * COMMISSION_PCT
                pnl -= commission
                equity += pnl

                current_trade.exit_idx = i
                current_trade.exit_price = exit_price
                current_trade.pnl = pnl
                current_trade.pnl_pct = pnl / (entry_price * qty) if entry_price * qty > 0 else 0
                current_trade.exit_reason = exit_reason
                trades.append(current_trade)

                position = 0
                current_trade = None

        # ── 空仓: 检查入场 ──
        if position == 0:
            unit_size = (equity * RISK_PCT) / atr_vals[i] if atr_vals[i] > 0 else 0
            if unit_size <= 0:
                equity_curve[i] = equity
                continue

            if not np.isnan(entry_upper.iloc[i]) and price > entry_upper.iloc[i]:
                # 做多
                position = 1
                entry_price = price
                qty = unit_size
                stop_price = price - STOP_MULT * atr_vals[i]
                current_trade = Trade(
                    entry_idx=i, direction=1, entry_price=price,
                    qty=qty, stop_price=stop_price
                )
            elif not np.isnan(entry_lower.iloc[i]) and price < entry_lower.iloc[i]:
                # 做空
                position = -1
                entry_price = price
                qty = unit_size
                stop_price = price + STOP_MULT * atr_vals[i]
                current_trade = Trade(
                    entry_idx=i, direction=-1, entry_price=price,
                    qty=qty, stop_price=stop_price
                )

        equity_curve[i] = equity
        peak_equity = max(peak_equity, equity)

    # 如果最后还有持仓，按收盘平掉
    if position != 0 and current_trade is not None:
        exit_price = close[-1]
        pnl = position * (exit_price - entry_price) * qty
        commission = (entry_price + exit_price) * qty * COMMISSION_PCT
        pnl -= commission
        equity += pnl
        current_trade.exit_idx = n - 1
        current_trade.exit_price = exit_price
        current_trade.pnl = pnl
        current_trade.pnl_pct = pnl / (entry_price * qty) if entry_price * qty > 0 else 0
        current_trade.exit_reason = "回测结束"
        trades.append(current_trade)
        equity_curve[-1] = equity

    # ── 统计指标 ──
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "total_return_pct": 0, "n_trades": 0, "win_rate": 0,
            "avg_pnl": 0, "max_drawdown_pct": 0, "profit_factor": 0,
            "avg_win": 0, "avg_loss": 0, "sharpe": 0,
            "trades": trades, "equity_curve": equity_curve,
            "final_equity": equity
        }

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / n_trades * 100

    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

    # 最大回撤
    eq = pd.Series(equity_curve).dropna()
    if len(eq) > 0:
        running_max = eq.cummax()
        drawdown = (eq - running_max) / running_max * 100
        max_dd = drawdown.min()
    else:
        max_dd = 0

    # 年化Sharpe (基于每笔交易收益率)
    pnl_pcts = [t.pnl_pct for t in trades]
    if len(pnl_pcts) > 1:
        avg_bars_per_trade = np.mean([t.exit_idx - t.entry_idx for t in trades])
        trades_per_year = (365.25 * BARS_PER_DAY) / avg_bars_per_trade if avg_bars_per_trade > 0 else 1
        sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(trades_per_year) if np.std(pnl_pcts) > 0 else 0
    else:
        sharpe = 0

    # 平均持仓时间 (天)
    avg_hold_days = np.mean([(t.exit_idx - t.entry_idx) / BARS_PER_DAY for t in trades])

    return {
        "total_return_pct": round(total_return, 2),
        "final_equity": round(equity, 2),
        "n_trades": n_trades,
        "n_long": sum(1 for t in trades if t.direction == 1),
        "n_short": sum(1 for t in trades if t.direction == -1),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_pnl": round(np.mean([t.pnl for t in trades]), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 3),
        "avg_hold_days": round(avg_hold_days, 1),
        "trades": trades,
        "equity_curve": equity_curve,
    }


# ──────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────

def main():
    # 数据路径
    csv_path = os.path.join(os.path.dirname(__file__),
                            "btc_usdt_15m_2022_to_now.csv")
    if not os.path.exists(csv_path):
        # 备选路径
        _root = next(p for p in Path(__file__).resolve().parents if (p / "data" / "paths.py").exists())
        _raw, _old = _root / "data" / "raw" / "btc_usdt_15m_2022_to_now.csv", _root / "data" / "btc_usdt_15m_2022_to_now.csv"
        csv_path = str(_raw if _raw.exists() or not _old.exists() else _old)   # 优先 data/raw/, 兼容旧位置
    df = load_data(csv_path)

    # ATR (天级别, 但用15min数据计算)
    atr_bars = ATR_PERIOD_DAYS * BARS_PER_DAY
    atr = calc_atr(df, atr_bars)

    print(f"\n{'='*70}")
    print(f"海龟交易系统 — 多入场周期回测")
    print(f"数据: BTC/USDT 15min, {len(df)} 根K线")
    print(f"初始资金: ${INITIAL_CAPITAL:,.0f}  |  单笔风险: {RISK_PCT*100}%  |"
          f"  止损: {STOP_MULT}N  |  手续费: {COMMISSION_PCT*100:.3f}%")
    print(f"{'='*70}\n")

    all_results = {}

    for entry_days in ENTRY_PERIODS_DAYS:
        entry_bars = entry_days * BARS_PER_DAY
        exit_days = exit_period_days(entry_days)
        exit_bars = exit_days * BARS_PER_DAY

        print(f"━━━ 入场 {entry_days}天 / 出场 {exit_days}天"
              f"  ({entry_bars}/{exit_bars} 根K线) ━━━")

        # E-ratio
        e_result = calc_e_ratio(df, entry_bars, atr, E_RATIO_WINDOWS)
        print(f"  信号数: {e_result['n_signals']}"
              f"  (多 {e_result['n_long']} / 空 {e_result['n_short']})")
        print(f"  E-ratio (窗口=根K线):")
        for w, er, mfe, mae in zip(e_result["windows"], e_result["e_ratios"],
                                    e_result["avg_mfe"], e_result["avg_mae"]):
            hours = w * 15 / 60
            label = f"{hours:.0f}h" if hours < 24 else f"{hours/24:.1f}d"
            bar = "█" * min(int(er * 10), 40) if er < float("inf") else "████████████████████"
            print(f"    {w:>4} ({label:>5}): {er:>7.3f}  "
                  f"MFE={mfe:.3f}  MAE={mae:.3f}  {bar}")

        # 回测
        bt = run_backtest(df, entry_bars, exit_bars, atr)
        print(f"  ── 回测结果 ──")
        print(f"    总收益:     {bt['total_return_pct']:>+8.2f}%"
              f"  (${bt['final_equity']:>10,.2f})")
        print(f"    交易次数:   {bt['n_trades']}"
              f"  (多 {bt.get('n_long',0)} / 空 {bt.get('n_short',0)})")
        print(f"    胜率:       {bt['win_rate']:>6.1f}%")
        print(f"    盈亏比(PF): {bt['profit_factor']:>6.3f}")
        print(f"    平均盈利:   ${bt['avg_win']:>8.2f}")
        print(f"    平均亏损:   ${bt['avg_loss']:>8.2f}")
        print(f"    最大回撤:   {bt['max_drawdown_pct']:>8.2f}%")
        print(f"    Sharpe:     {bt['sharpe']:>8.3f}")
        print(f"    平均持仓:   {bt.get('avg_hold_days', 0):>6.1f} 天")
        print()

        all_results[entry_days] = {
            "e_ratio": e_result,
            "backtest": bt,
            "exit_days": exit_days,
        }

    # ──────────────────────────────────────────────────
    # 汇总对比表
    # ──────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"{'汇总对比表':^70}")
    print(f"{'='*70}")

    header = (f"{'入场':>4}{'出场':>5}{'信号':>6}{'交易':>5}"
              f"{'收益%':>8}{'胜率%':>7}{'PF':>7}"
              f"{'回撤%':>8}{'Sharpe':>8}{'E1h':>7}{'E1d':>7}")
    print(header)
    print("─" * len(header))

    for ed in ENTRY_PERIODS_DAYS:
        r = all_results[ed]
        bt = r["backtest"]
        er = r["e_ratio"]
        # E-ratio at 4 bars (1h) and 96 bars (1d)
        e1h = er["e_ratios"][E_RATIO_WINDOWS.index(4)] if 4 in E_RATIO_WINDOWS else 0
        e1d = er["e_ratios"][E_RATIO_WINDOWS.index(96)] if 96 in E_RATIO_WINDOWS else 0

        print(f"{ed:>4}d {r['exit_days']:>4}d {er['n_signals']:>5}"
              f" {bt['n_trades']:>5}"
              f" {bt['total_return_pct']:>+7.1f}"
              f" {bt['win_rate']:>6.1f}"
              f" {bt['profit_factor']:>6.2f}"
              f" {bt['max_drawdown_pct']:>7.1f}"
              f" {bt['sharpe']:>7.3f}"
              f" {e1h:>6.3f}"
              f" {e1d:>6.3f}")

    # ──────────────────────────────────────────────────
    # 生成图表
    # ──────────────────────────────────────────────────

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        # 中文字体 (Linux 通用)
        for font in ["WenQuanYi Micro Hei", "Noto Sans CJK SC",
                      "SimHei", "DejaVu Sans"]:
            try:
                rcParams["font.sans-serif"] = [font]
                rcParams["axes.unicode_minus"] = False
                # 快速测试字体
                fig_test, ax_test = plt.subplots()
                ax_test.set_title("测试")
                plt.close(fig_test)
                break
            except Exception:
                continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Turtle System — Multi-Period Backtest (BTC 15min)",
                     fontsize=14, fontweight="bold")

        periods = ENTRY_PERIODS_DAYS
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(periods)))

        # ── 1. 权益曲线 ──
        ax = axes[0, 0]
        for ed, c in zip(periods, colors):
            eq = pd.Series(all_results[ed]["backtest"]["equity_curve"]).dropna()
            # 降采样到日级别
            eq_daily = eq.iloc[::BARS_PER_DAY]
            ax.plot(range(len(eq_daily)), eq_daily.values,
                    label=f"{ed}d", color=c, linewidth=1.2)
        ax.set_title("Equity Curve")
        ax.set_ylabel("Equity ($)")
        ax.set_xlabel("Trading Days")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)

        # ── 2. 收益 & 回撤 柱状图 ──
        ax = axes[0, 1]
        returns = [all_results[ed]["backtest"]["total_return_pct"] for ed in periods]
        drawdowns = [abs(all_results[ed]["backtest"]["max_drawdown_pct"]) for ed in periods]
        x = np.arange(len(periods))
        w = 0.35
        bars1 = ax.bar(x - w/2, returns, w, label="Return %",
                       color=[c if r >= 0 else "red" for r, c in zip(returns, colors)])
        bars2 = ax.bar(x + w/2, drawdowns, w, label="Max DD %",
                       color="salmon", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{ed}d" for ed in periods])
        ax.set_title("Return vs Max Drawdown")
        ax.set_ylabel("%")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linewidth=0.5)

        # ── 3. E-ratio 热力图 ──
        ax = axes[1, 0]
        e_matrix = []
        for ed in periods:
            e_matrix.append(all_results[ed]["e_ratio"]["e_ratios"])
        e_matrix = np.array(e_matrix)
        # 截断极值方便显示
        e_display = np.clip(e_matrix, 0, 3)
        im = ax.imshow(e_display, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=2.0)
        ax.set_yticks(range(len(periods)))
        ax.set_yticklabels([f"{ed}d" for ed in periods])
        window_labels = []
        for w in E_RATIO_WINDOWS:
            hours = w * 15 / 60
            window_labels.append(f"{hours:.0f}h" if hours < 24 else f"{hours/24:.0f}d")
        ax.set_xticks(range(len(E_RATIO_WINDOWS)))
        ax.set_xticklabels(window_labels)
        ax.set_title("E-ratio Heatmap (green=good)")
        ax.set_xlabel("Observation Window")
        ax.set_ylabel("Entry Period")
        fig.colorbar(im, ax=ax, shrink=0.8)
        # 在格子里标数字
        for i in range(len(periods)):
            for j in range(len(E_RATIO_WINDOWS)):
                val = e_matrix[i, j]
                txt = f"{val:.2f}" if val < 10 else f"{val:.1f}"
                text_color = "white" if e_display[i, j] < 0.8 or e_display[i, j] > 1.7 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=text_color)

        # ── 4. 关键指标雷达对比 ──
        ax = axes[1, 1]
        metrics = ["Win Rate", "PF", "Sharpe", "Trades", "E1d"]
        metric_data = {}
        for ed in periods:
            bt = all_results[ed]["backtest"]
            er = all_results[ed]["e_ratio"]
            e1d = er["e_ratios"][E_RATIO_WINDOWS.index(96)] if 96 in E_RATIO_WINDOWS else 0
            metric_data[ed] = [
                bt["win_rate"],
                bt["profit_factor"],
                bt["sharpe"],
                bt["n_trades"],
                e1d
            ]

        # 标准化到 [0, 1] 显示
        metric_arr = np.array([metric_data[ed] for ed in periods])
        if metric_arr.shape[0] > 0:
            mins = metric_arr.min(axis=0)
            maxs = metric_arr.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1
            normalized = (metric_arr - mins) / ranges

            x_pos = np.arange(len(metrics))
            for i, (ed, c) in enumerate(zip(periods, colors)):
                ax.plot(x_pos, normalized[i], "o-", color=c,
                        label=f"{ed}d", linewidth=1.5, markersize=5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics)
            ax.set_ylabel("Normalized Score")
            ax.set_title("Key Metrics Comparison")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)

        plt.tight_layout()
        fig_dir = next(p for p in Path(__file__).resolve().parents if (p / "data" / "paths.py").exists()) / "figures" / "turtle"   # quant/figures/turtle/
        fig_dir.mkdir(parents=True, exist_ok=True)
        chart_path = str(fig_dir / "turtle_backtest_results.png")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n图表已保存: {chart_path}")

    except ImportError:
        print("\n[!] matplotlib 未安装, 跳过图表生成")
    except Exception as e:
        print(f"\n[!] 图表生成出错: {e}")

    print("\n回测完成。")


if __name__ == "__main__":
    main()
