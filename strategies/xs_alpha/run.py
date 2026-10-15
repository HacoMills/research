"""
截面 alpha 回测: 沪深300 成分股
================================
alpha = ts_decay_linear( a * rank(volume / adv20) * rank(ts_std(returns, 20)), 5 )
    a = -(close - open) / (high - low)      日内反转: 日内涨得越多, 分数越低

运行 (在 quant 根目录):
    python strategies/xs_alpha/run.py
    python strategies/xs_alpha/run.py --start 2018-01-01 --cost 0.0015

注意: 股票池用的是 "当前" 沪深300 成分股, 存在幸存者偏差 (2015 年时很多股票还不在指数里,
那些后来被剔除的股票也不在样本中). 多空组合受影响较小, 纯多头的绝对收益会被高估.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backtest import cross_section as xs                          # noqa: E402
from backtest.analytics import C_BENCH, C_MUTED, C_PORT, _style, stats_from_returns  # noqa: E402
from backtest.report import md_table                              # noqa: E402
from data.paths import FIGURES_DIR, INDEX_CACHE, REPORTS_DIR      # noqa: E402

NAME = "xs_alpha"
IS_END = "2021-12-31"                       # 与趋势项目一致: 2022 年起为样本外


def csi300_codes(refresh: bool = False) -> list:
    path = INDEX_CACHE / "csi300_cons.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype=str)["code"].tolist()
    import akshare as ak
    df = ak.index_stock_cons_csindex(symbol="000300")
    codes = df.iloc[:, 4].astype(str).str.zfill(6).tolist()       # 第 5 列: 成分券代码
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": codes}).to_csv(path, index=False)
    return codes


def compute_alpha(P):
    o, h, l, c, v = P["open"], P["high"], P["low"], P["close"], P["volume"]
    ret = c.pct_change(fill_method=None)
    a = -(c - o) / (h - l).where(h > l)                            # 一字板 high == low → 缺失
    rel_vol = v / xs.ts_mean(v, 20)                                # volume / adv20 (用成交量)
    raw = a * xs.rank(rel_vol) * xs.rank(xs.ts_std(ret, 20))
    return xs.ts_decay_linear(raw, 5)


def stats_row(name, r):
    s = stats_from_returns(r, xs.DAYS_PER_YEAR)
    return {"": name, "年化收益%": s["cagr_pct"], "年化波动%": s["vol_pct"], "夏普": s["sharpe"],
            "最大回撤%": s["max_dd_pct"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--cost", type=float, default=0.0012, help="单边成本, 按 |Δw| 计")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    codes = csi300_codes(args.refresh)
    print(f"股票池: 沪深300 当前成分股 {len(codes)} 只, 加载数据 ...")
    P = xs.load_panel(codes, "2014-10-01", args.refresh)          # 多拉一个季度做预热
    P = {k: v.loc[v.index >= "2014-10-01"] for k, v in P.items()}
    print(f"面板: {P['close'].shape[0]} 天 × {P['close'].shape[1]} 只")

    alpha = compute_alpha(P)
    tradable = xs.tradable_mask(P)
    start = pd.Timestamp(args.start)

    def cut(x):
        return x.loc[x.index >= start]

    # ---- IC ----
    ic = cut(xs.rank_ic(alpha, P, 1))
    ic5 = cut(xs.rank_ic(alpha, P, 5))
    ic_tab = pd.DataFrame([{"": "1 日", **xs.ic_summary(ic)}, {"": "5 日", **xs.ic_summary(ic5)},
                           {"": "1 日 · 样本内", **xs.ic_summary(ic[:IS_END])},
                           {"": "1 日 · 样本外", **xs.ic_summary(ic[IS_END:])}])

    # ---- 多空 ----
    w_ls = xs.long_short_weights(alpha, tradable)
    runs = {}
    for ex in ("close", "open"):
        runs[ex] = cut(xs.portfolio_returns(w_ls, P, args.cost, ex))
    ls_rows = []
    for ex, lab in (("close", "收盘成交"), ("open", "次日开盘成交")):
        r = runs[ex]
        ls_rows.append({**stats_row(f"多空 · {lab} · 扣费前", r["gross"]), "日均换手%": r["turnover"].mean() * 100})
        ls_rows.append({**stats_row(f"多空 · {lab} · 扣费后", r["net"]), "日均换手%": r["turnover"].mean() * 100})
    ls_tab = pd.DataFrame(ls_rows)

    # ---- 样本内外 ----
    split_rows = []
    for ex, lab in (("close", "收盘成交"), ("open", "次日开盘成交")):
        for part, sl in (("样本内", slice(None, IS_END)), ("样本外", slice(IS_END, None))):
            r = runs[ex].loc[sl]
            split_rows.append({**stats_row(f"{lab} · {part} · 扣费前", r["gross"]),
                               "扣费后夏普": stats_from_returns(r["net"], xs.DAYS_PER_YEAR)["sharpe"]})
    split_tab = pd.DataFrame(split_rows)

    # ---- 五分组 ----
    w_ew = xs.quantile_weights(alpha.notna().astype(float).where(alpha.notna()), tradable, 1, 1)  # 全部等权
    ew = cut(xs.portfolio_returns(w_ew, P, 0, "close"))["gross"]
    ew_open = cut(xs.portfolio_returns(w_ew, P, 0, "open"))["gross"]
    q_rows, q_curves = [], {}
    for q in range(1, 6):
        rq = cut(xs.portfolio_returns(xs.quantile_weights(alpha, tradable, q), P, args.cost, "close"))
        q_curves[q] = rq["gross"]
        s = stats_from_returns(rq["gross"], xs.DAYS_PER_YEAR)
        ex_r = rq["gross"] - ew
        q_rows.append({"": f"Q{q}" + (" (alpha 最低)" if q == 1 else " (alpha 最高)" if q == 5 else ""),
                       "年化收益%": s["cagr_pct"], "夏普": s["sharpe"],
                       "相对等权 年化超额%": ex_r.mean() * xs.DAYS_PER_YEAR * 100,
                       "日均换手%": rq["turnover"].mean() * 100})
    q_tab = pd.DataFrame(q_rows)

    # Q5 多头 - 等权 (现实中: 做多前 20% + 股指期货对冲)
    q5 = cut(xs.portfolio_returns(xs.quantile_weights(alpha, tradable, 5), P, args.cost, "open"))
    hedge_rows = [stats_row("Q5 − 等权 · 次日开盘 · 扣费前", q5["gross"] - ew_open),
                  stats_row("Q5 − 等权 · 次日开盘 · 扣费后", q5["net"] - ew_open)]
    hedge_tab = pd.DataFrame(hedge_rows)

    # ---- 成本敏感性 ----
    cost_rows = []
    for c in (0, 0.0005, 0.0012, 0.002):
        row = {"单边成本": f"{c * 1e4:.0f} bp"}
        for ex, lab in (("close", "收盘成交 夏普"), ("open", "次日开盘 夏普")):
            r = runs[ex]
            row[lab] = stats_from_returns(r["gross"] - r["turnover"] * c, xs.DAYS_PER_YEAR)["sharpe"]
        cost_rows.append(row)
    cost_tab = pd.DataFrame(cost_rows)

    # ---- 分年 ----
    yr = pd.DataFrame({
        "收盘成交 扣费前%": runs["close"]["gross"].groupby(lambda d: d.year).sum() * 100,
        "次日开盘 扣费前%": runs["open"]["gross"].groupby(lambda d: d.year).sum() * 100,
        "次日开盘 扣费后%": runs["open"]["net"].groupby(lambda d: d.year).sum() * 100,
        "IC均值": ic.groupby(lambda d: d.year).mean(),
    })
    yr.index.name = "年份"
    yr = yr.reset_index()
    yr["年份"] = yr["年份"].astype(str)

    # ---- 图 ----
    fig_dir = FIGURES_DIR / NAME
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"{NAME}_curves.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for col, ex, color, ls, lab in (("gross", "close", C_PORT, "-", "收盘成交 扣费前"),
                                     ("gross", "open", C_BENCH, "-", "次日开盘 扣费前"),
                                     ("net", "open", C_BENCH, "--", "次日开盘 扣费后")):
        ax1.plot((1 + runs[ex][col]).cumprod(), color=color, ls=ls, lw=1.4, label=lab)
    ax1.axvline(pd.Timestamp(IS_END), color=C_MUTED, lw=1, ls=":")
    ax1.set_title("多空组合净值 (总杠杆 1)")
    ax1.set_yscale("log")
    ax1.legend(frameon=False, fontsize=9)
    _style(ax1)
    ann = [q_curves[q].mean() * xs.DAYS_PER_YEAR * 100 for q in range(1, 6)]
    ax2.bar([f"Q{q}" for q in range(1, 6)], ann, color=[C_MUTED] * 4 + [C_PORT])
    ax2.set_title("五分组 年化收益% (收盘成交, 扣费前)")
    _style(ax2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    # ---- 报告 ----
    rep_dir = REPORTS_DIR / NAME
    rep_dir.mkdir(parents=True, exist_ok=True)
    f2 = {"IC均值": "{:.4f}", "IC标准差": "{:.4f}", "天数": "{:.0f}"}
    L = [f"# 截面 alpha 回测: 沪深300 成分股", "",
         "```", "a     = -(close - open) / (high - low)",
         "alpha = ts_decay_linear( a * rank(volume / adv20) * rank(ts_std(returns, 20)), 5 )", "```", "",
         f"- 区间 {ic.index[0].date()} ~ {ic.index[-1].date()}, 日频, 后复权; "
         f"样本内 ≤ {IS_END}, 样本外 2022 起",
         f"- 股票池: **当前**沪深300 成分股 {P['close'].shape[1]} 只 (有幸存者偏差, 见文末)",
         f"- 单边成本 {args.cost * 1e4:.0f} bp (佣金 + 印花税 + 冲击, 按 |Δw| 计)",
         "- 停牌、一字板、收盘封板的股票当天不调仓, 沿用昨天的权重", "",
         "## 1. Rank IC", "", *md_table(ic_tab, f2), "",
         "## 2. 多空组合 (按 alpha 去均值加权, 多空各 0.5)", "", *md_table(ls_tab), "",
         f"![净值]({Path('../../figures') / NAME / fig_path.name})", "",
         "## 3. 样本内外", "", *md_table(split_tab), "",
         "## 4. 五分组 (收盘成交, 扣费前)", "", *md_table(q_tab), "",
         "## 5. 现实可做的版本: 做多 Q5, 对冲等权组合", "", *md_table(hedge_tab), "",
         "## 6. 成本敏感性 (多空)", "", *md_table(cost_tab), "",
         "## 7. 分年 (多空, 日收益简单加总)", "", *md_table(yr, {"IC均值": "{:.4f}"}), "",
         "## 局限", "",
         "- 股票池是当前成分股, 回测早期的股票池是 '事后挑出来的赢家', 纯多头收益被高估",
         "- adv20 用成交量 (股) 而不是成交额; A 股融券难, 纯多空版本现实中做不了",
         "- 涨跌停按板块近似 (主板 10%, 创业板 2020-08 起 20%, 科创板 20%), 未单独处理 ST (5%)",
         ]
    rep = rep_dir / f"{NAME}_report.md"
    rep.write_text("\n".join(L), encoding="utf-8")
    runs["open"].to_csv(rep_dir / f"{NAME}_ls_open_returns.csv")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    for t in (ic_tab, ls_tab, split_tab, q_tab, hedge_tab, cost_tab, yr):
        print(t.round(4).to_string(index=False)); print()
    print(f"报告: {rep}")


if __name__ == "__main__":
    main()
