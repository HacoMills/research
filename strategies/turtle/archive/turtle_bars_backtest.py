#!/usr/bin/env python3
"""
海龟交易系统 — 按K线根数回测（非天数）
========================================
通道周期直接用根数:
  系统A: 20根入场 / 10根出场
  系统B: 60根入场 / 20根出场
在不同时间框架 (15min/30min/1h/4h/1d) 上对比

仓位管理: 1% 资金 / ATR  |  止损: 2N
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings, os

warnings.filterwarnings("ignore")

INITIAL_CAPITAL = 10_000.0
RISK_PCT = 0.01
STOP_MULT = 2.0
ATR_BARS = 20              # ATR 也用 20 根K线
COMMISSION_PCT = 0.05 / 100

TIMEFRAMES = [
    ("15min", None,   96),
    ("30min", "30min", 48),
    ("1h",   "1h",   24),
    ("4h",   "4h",    6),
    ("1d",   "1D",    1),
]

# 通道周期 = 根K线（不再乘以 bars_per_day）
SYSTEMS = [
    ("20/10根", 20, 10),
    ("60/20根", 60, 20),
]


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    print(f"原始数据: {len(df)} 行, {df.index[0]} ~ {df.index[-1]}")
    return df


def resample_ohlcv(df, rule):
    return df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()


def calc_atr(df, period):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def run_backtest(df, entry_bars, exit_bars, atr_bars, bars_per_day):
    atr = calc_atr(df, atr_bars)

    entry_upper = df["high"].rolling(entry_bars).max().shift(1)
    entry_lower = df["low"].rolling(entry_bars).min().shift(1)
    exit_upper  = df["high"].rolling(exit_bars).max().shift(1)
    exit_lower  = df["low"].rolling(exit_bars).min().shift(1)

    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    atr_v = atr.values
    n = len(df)

    equity = INITIAL_CAPITAL
    position = 0
    entry_price = stop_price = qty = 0.0
    trades_pnl = []
    equity_curve = np.full(n, np.nan)

    warmup = max(entry_bars, exit_bars, atr_bars) + 2

    for i in range(warmup, n):
        if np.isnan(atr_v[i]) or atr_v[i] <= 0:
            equity_curve[i] = equity
            continue

        price = close[i]

        # 出场
        if position != 0:
            hit = False
            ep = price
            if position == 1:
                if low[i] <= stop_price:
                    ep, hit = stop_price, True
                elif not np.isnan(exit_lower.iloc[i]) and price < exit_lower.iloc[i]:
                    ep, hit = price, True
            else:
                if high[i] >= stop_price:
                    ep, hit = stop_price, True
                elif not np.isnan(exit_upper.iloc[i]) and price > exit_upper.iloc[i]:
                    ep, hit = price, True
            if hit:
                pnl = position * (ep - entry_price) * qty
                pnl -= (entry_price + ep) * qty * COMMISSION_PCT
                equity += pnl
                trades_pnl.append(pnl)
                position = 0

        # 入场
        if position == 0 and equity > 0:
            unit = (equity * RISK_PCT) / atr_v[i]
            if unit > 0:
                if not np.isnan(entry_upper.iloc[i]) and price > entry_upper.iloc[i]:
                    position, entry_price, qty = 1, price, unit
                    stop_price = price - STOP_MULT * atr_v[i]
                elif not np.isnan(entry_lower.iloc[i]) and price < entry_lower.iloc[i]:
                    position, entry_price, qty = -1, price, unit
                    stop_price = price + STOP_MULT * atr_v[i]

        equity_curve[i] = equity

    # 末尾平仓
    if position != 0:
        ep = close[-1]
        pnl = position * (ep - entry_price) * qty
        pnl -= (entry_price + ep) * qty * COMMISSION_PCT
        equity += pnl
        trades_pnl.append(pnl)
        equity_curve[-1] = equity

    # 统计
    nt = len(trades_pnl)
    if nt == 0:
        return {"total_ret": 0, "ann_ret": 0, "n": 0, "wr": 0,
                "pf": 0, "avg_w": 0, "avg_l": 0, "dd": 0,
                "calmar": 0, "eq": equity_curve}

    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p <= 0]
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0

    total_ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    total_bars = n - warmup
    years = total_bars / (bars_per_day * 365.25) if bars_per_day > 0 else 1
    ann_ret = ((equity / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 and equity > 0 else 0

    eq_s = pd.Series(equity_curve).dropna()
    dd = ((eq_s - eq_s.cummax()) / eq_s.cummax() * 100).min() if len(eq_s) > 1 else 0

    return {
        "total_ret": round(total_ret, 1),
        "ann_ret": round(ann_ret, 1),
        "n": nt,
        "wr": round(len(wins)/nt*100, 1),
        "pf": round(gp/gl, 2) if gl > 0 else 999,
        "avg_w": round(np.mean(wins), 1) if wins else 0,
        "avg_l": round(np.mean(losses), 1) if losses else 0,
        "dd": round(dd, 1),
        "calmar": round(ann_ret / abs(dd), 3) if dd != 0 else 0,
        "eq": equity_curve,
    }


def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "btc_usdt_15m_2022_to_now.csv")
    if not os.path.exists(csv_path):
        _root = next(p for p in Path(__file__).resolve().parents if (p / "data" / "paths.py").exists())
        _raw, _old = _root / "data" / "raw" / "btc_usdt_15m_2022_to_now.csv", _root / "data" / "btc_usdt_15m_2022_to_now.csv"
        csv_path = str(_raw if _raw.exists() or not _old.exists() else _old)   # 优先 data/raw/, 兼容旧位置

    df_raw = load_data(csv_path)

    print(f"\n{'='*85}")
    print(f"海龟系统 — 按K线根数回测 (非天数)")
    print(f"仓位: 1%资金/ATR  |  止损: 2N  |  ATR周期: {ATR_BARS}根  |  手续费: 0.05%")
    print(f"{'='*85}\n")

    all_results = {}

    for tf_name, tf_rule, bpd in TIMEFRAMES:
        df_tf = df_raw.copy() if tf_rule is None else resample_ohlcv(df_raw, tf_rule)
        print(f"▸ {tf_name}  ({len(df_tf)} 根K线)")

        for sys_name, eb, xb in SYSTEMS:
            if len(df_tf) < max(eb, xb, ATR_BARS) + 10:
                print(f"  {sys_name}: 数据不足, 跳过")
                continue

            # 换算成实际时间，方便理解
            bar_min = {"15min": 15, "30min": 30, "1h": 60, "4h": 240, "1d": 1440}
            entry_hours = eb * bar_min[tf_name] / 60
            exit_hours = xb * bar_min[tf_name] / 60
            entry_label = f"{entry_hours:.0f}h" if entry_hours < 24 else f"{entry_hours/24:.1f}d"
            exit_label = f"{exit_hours:.0f}h" if exit_hours < 24 else f"{exit_hours/24:.1f}d"

            r = run_backtest(df_tf, eb, xb, ATR_BARS, bpd)
            all_results[(tf_name, sys_name)] = r

            print(f"  {sys_name} (={entry_label}/{exit_label})"
                  f"  收益:{r['total_ret']:>+7.1f}%  年化:{r['ann_ret']:>+5.1f}%"
                  f"  交易:{r['n']:>4}  胜率:{r['wr']:>5.1f}%"
                  f"  PF:{r['pf']:>5.2f}  回撤:{r['dd']:>6.1f}%"
                  f"  Calmar:{r['calmar']:>6.3f}")
        print()

    # 汇总表
    bar_min = {"15min": 15, "30min": 30, "1h": 60, "4h": 240, "1d": 1440}

    print(f"\n{'='*100}")
    print(f"{'汇总对比表':^100}")
    print(f"{'='*100}")
    print(f"{'K线':>5} {'系统':>8} {'实际窗口':>12} {'交易':>5}"
          f" {'总收益%':>8} {'年化%':>6} {'胜率%':>6} {'PF':>6}"
          f" {'平均赢':>9} {'平均亏':>9} {'回撤%':>7} {'Calmar':>7}")
    print("─" * 100)

    for tf_name, _, _ in TIMEFRAMES:
        for sys_name, eb, xb in SYSTEMS:
            key = (tf_name, sys_name)
            if key not in all_results:
                continue
            r = all_results[key]
            eh = eb * bar_min[tf_name] / 60
            xh = xb * bar_min[tf_name] / 60
            el = f"{eh:.0f}h" if eh < 24 else f"{eh/24:.1f}d"
            xl = f"{xh:.0f}h" if xh < 24 else f"{xh/24:.1f}d"

            print(f"{tf_name:>5} {sys_name:>8} {el:>5}/{xl:<5}"
                  f" {r['n']:>5}"
                  f" {r['total_ret']:>+7.1f}"
                  f" {r['ann_ret']:>+5.1f}"
                  f" {r['wr']:>5.1f}"
                  f" {r['pf']:>5.2f}"
                  f" {r['avg_w']:>8.1f}"
                  f" {r['avg_l']:>8.1f}"
                  f" {r['dd']:>6.1f}"
                  f" {r['calmar']:>6.3f}")
        print("─" * 100)

    # 图表
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        tf_names = [t[0] for t in TIMEFRAMES]
        sys_names = [s[0] for s in SYSTEMS]
        colors = ["#2196F3", "#FF5722"]

        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        fig.suptitle("Turtle System — Fixed Bar Lookback (60/20 & 20/10 bars)\n"
                     "BTC/USDT, 1% Risk/ATR, 2N Stop",
                     fontsize=13, fontweight="bold")

        x = np.arange(len(tf_names))
        w = 0.35

        # 实际窗口标注
        def time_label(tf, bars):
            h = bars * bar_min[tf] / 60
            return f"{h:.0f}h" if h < 24 else f"{h/24:.0f}d"

        # 1. 总收益
        ax = axes[0, 0]
        for j, (sn, sc) in enumerate(zip(sys_names, colors)):
            eb = SYSTEMS[j][1]
            vals = [all_results.get((tn, sn), {"total_ret": 0})["total_ret"] for tn in tf_names]
            bars = ax.bar(x + (j-0.5)*w, vals, w, label=sn, color=sc, alpha=0.8)
            for i2, (bar, v, tn) in enumerate(zip(bars, vals, tf_names)):
                tl = time_label(tn, eb)
                ax.text(bar.get_x()+bar.get_width()/2, max(v, 0)+8,
                        f"{v:+.0f}%\n({tl})", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels(tf_names)
        ax.set_title("Total Return"); ax.set_ylabel("Return %")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linewidth=0.5)

        # 2. 回撤
        ax = axes[0, 1]
        for j, (sn, sc) in enumerate(zip(sys_names, colors)):
            vals = [abs(all_results.get((tn, sn), {"dd": 0})["dd"]) for tn in tf_names]
            ax.bar(x + (j-0.5)*w, vals, w, label=sn, color=sc, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(tf_names)
        ax.set_title("Max Drawdown"); ax.set_ylabel("Drawdown %")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

        # 3. PF + Calmar
        ax = axes[1, 0]
        ax2 = ax.twinx()
        bw = 0.18
        for j, (sn, sc) in enumerate(zip(sys_names, colors)):
            pf_v = [min(all_results.get((tn, sn), {"pf": 0})["pf"], 5) for tn in tf_names]
            cal_v = [all_results.get((tn, sn), {"calmar": 0})["calmar"] for tn in tf_names]
            off = (j-0.5)*bw*2
            ax.bar(x+off-bw/2, pf_v, bw, label=f"PF {sn}", color=sc, alpha=0.6)
            ax2.plot(x+off, cal_v, "s--", color=sc, markersize=6, label=f"Calmar {sn}")
        ax.set_xticks(x); ax.set_xticklabels(tf_names)
        ax.set_title("Profit Factor (bars) & Calmar Ratio (squares)")
        ax.set_ylabel("PF"); ax2.set_ylabel("Calmar")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        # 4. 权益曲线 60/20根
        ax = axes[1, 1]
        cm = plt.cm.tab10
        for idx, (tf_name, _, bpd) in enumerate(TIMEFRAMES):
            key = (tf_name, "60/20根")
            if key not in all_results:
                continue
            eq = pd.Series(all_results[key]["eq"]).dropna()
            if len(eq) < 2:
                continue
            step = max(bpd, 1)
            eq_d = eq.iloc[::step]
            ax.plot(range(len(eq_d)), eq_d.values, label=f"{tf_name} ({time_label(tf_name, 60)})",
                    color=cm(idx), linewidth=1.2)
        ax.set_title("Equity Curves — 60/20 bars")
        ax.set_ylabel("Equity ($)"); ax.set_xlabel("Trading Days")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.axhline(INITIAL_CAPITAL, color="gray", ls="--", alpha=0.5)

        plt.tight_layout()
        fig_dir = next(p for p in Path(__file__).resolve().parents if (p / "data" / "paths.py").exists()) / "figures" / "turtle"   # quant/figures/turtle/
        fig_dir.mkdir(parents=True, exist_ok=True)
        out = str(fig_dir / "turtle_bars_results.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n图表已保存: {out}")

    except Exception as e:
        print(f"\n[!] 图表出错: {e}")
        import traceback; traceback.print_exc()

    print("\n回测完成。")


if __name__ == "__main__":
    main()
