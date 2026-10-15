"""
降换手实验: 同一个 alpha, 比较三种降换手方法
=============================================
1. 衰减窗口  ts_decay_linear 的 n: 5 (原版) / 10 / 20
2. 调仓频率  每 1 / 5 / 10 天调一次
3. 部分调仓  每天只向目标权重走 k = 1 / 0.3 / 0.1

全部用多空组合、次日开盘成交. 运行 (在 quant 根目录):
    python strategies/xs_alpha/turnover.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run                                                        # noqa: E402
from backtest import cross_section as xs                          # noqa: E402
from backtest.analytics import stats_from_returns                 # noqa: E402
from backtest.report import md_table                              # noqa: E402
from data.paths import REPORTS_DIR                                # noqa: E402

COST = 0.0012
START = "2015-01-01"


def alpha_with_decay(P, n):
    o, h, l, c, v = P["open"], P["high"], P["low"], P["close"], P["volume"]
    ret = c.pct_change(fill_method=None)
    a = -(c - o) / (h - l).where(h > l)
    raw = a * xs.rank(v / xs.ts_mean(v, 20)) * xs.rank(xs.ts_std(ret, 20))
    return xs.ts_decay_linear(raw, n)


def evaluate(label, w, P):
    r = xs.portfolio_returns(w, P, COST, "open")
    r = r.loc[r.index >= START]
    row = {"": label, "日均换手%": r["turnover"].mean() * 100}
    for part, sl in (("全样本", slice(None)), ("样本外", slice(run.IS_END, None))):
        g = stats_from_returns(r["gross"].loc[sl], xs.DAYS_PER_YEAR)
        n = stats_from_returns(r["net"].loc[sl], xs.DAYS_PER_YEAR)
        row[f"{part} 扣费前夏普"] = g["sharpe"]
        row[f"{part} 扣费后夏普"] = n["sharpe"]
    gross_ann = r["gross"].mean() * xs.DAYS_PER_YEAR
    row["盈亏平衡成本bp"] = gross_ann / (r["turnover"].mean() * xs.DAYS_PER_YEAR) * 1e4
    return row


def main():
    P = xs.load_panel(run.csi300_codes(), "2014-10-01", verbose=False)
    tradable = xs.tradable_mask(P)
    rows = []
    for n in (5, 10, 20):
        alpha = alpha_with_decay(P, n)
        ic = xs.rank_ic(alpha, P, 1)
        ic = ic[ic.index >= START]
        w = xs.long_short_weights(alpha, tradable)
        base = f"衰减 {n} 天"
        r0 = evaluate(base, w, P)
        r0["IC均值"] = ic.mean()
        rows.append(r0)
        for every in (5, 10):
            rows.append(evaluate(f"{base} · 每 {every} 天调仓", xs.rebalance_every(w, every), P))
        for k in (0.3, 0.1):
            rows.append(evaluate(f"{base} · 部分调仓 k={k}", xs.partial_adjust(w, tradable, k), P))
        print(f"  衰减 {n} 天 完成")

    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(tab.round(3).to_string(index=False))

    L = ["# 降换手实验", "",
         f"多空组合, 次日开盘成交, 单边成本 {COST * 1e4:.0f} bp; 样本外 = 2022 年起.",
         "盈亏平衡成本 = 扣费前年化收益 / 年化换手, 单边成本低于它才能赚钱.", "",
         *md_table(tab, {"IC均值": "{:.4f}", "盈亏平衡成本bp": "{:.1f}", "日均换手%": "{:.1f}"})]
    out = REPORTS_DIR / run.NAME / "turnover_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"报告: {out}")


if __name__ == "__main__":
    main()
