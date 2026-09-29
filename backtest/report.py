"""
第 6 层: 报告输出
=================
把组合结果和稳定性检验写成一份 Markdown 报告 + 图片 + 组合日收益 CSV:
  quant/reports/<项目>/<name>_report.md
  quant/reports/<项目>/<name>_returns.csv
  quant/figures/<项目>/<name>_*.png
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import analytics as ta
from .analytics import C_BENCH, C_MUTED, C_PORT, C_SURFACE, C_TEXT, C_TEXT2, _style
from .portfolio import PortfolioResult, within_cross_corr

STRESS_PERIODS = [
    ("2015 股灾", "2015-06-15", "2015-09-15"),
    ("2016 熔断", "2016-01-01", "2016-01-29"),
    ("2018 贸易摩擦", "2018-03-22", "2018-12-31"),
    ("2020 疫情冲击", "2020-01-20", "2020-03-23"),
    ("2022 全球加息", "2022-01-01", "2022-10-31"),
    ("2022 LUNA 崩盘", "2022-05-05", "2022-06-18"),
    ("2022 FTX 破产", "2022-11-06", "2022-11-21"),
    ("2024 年初小盘股下跌", "2024-01-02", "2024-02-05"),
    ("2024-08 日元套息平仓", "2024-07-31", "2024-08-07"),
    ("2025-04 关税冲击", "2025-04-02", "2025-04-09"),
]


def md_table(df: pd.DataFrame, fmt: Dict[str, str] = None, bold_row: int = None) -> List[str]:
    """DataFrame → Markdown 表格. fmt: {列名: '{:.2f}'}."""
    if df is None or df.empty:
        return []
    fmt = fmt or {}
    cols = list(df.columns)
    L = ["| " + " | ".join(str(c) for c in cols) + " |",
         "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in cols) + "|"]
    for i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                s = "—" if np.isnan(v) else fmt.get(c, "{:.2f}").format(v)
            else:
                s = str(v)
            cells.append(f"**{s}**" if bold_row is not None and i == bold_row else s)
        L.append("| " + " | ".join(cells) + " |")
    return L


def _stats_row(name, r, dpy, extra=None):
    s = ta.stats_from_returns(r, dpy)
    if not s:
        return None
    row = {"": name, "起始": str(s["start"].date()), "年化收益%": s["cagr_pct"], "年化波动%": s["vol_pct"],
           "夏普": s["sharpe"], "Sortino": s["sortino"], "最大回撤%": s["max_dd_pct"], "Calmar": s["calmar"]}
    row.update(extra or {})
    return row


FMT = {"年化收益%": "{:+.1f}", "年化波动%": "{:.1f}", "最大回撤%": "{:.1f}", "收益%": "{:+.1f}",
       "年化%": "{:+.1f}", "持仓占比%": "{:.0f}", "交易": "{:.0f}", "交易次数": "{:.0f}",
       "被挡掉的入场": "{:.0f}", "p1": "{:g}", "p2": "{:g}", "p3": "{:g}"}


# ──────────────────────────────────────────────────
# 图
# ──────────────────────────────────────────────────

def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def plot_compare(series: Dict[str, pd.Series], base: str, path: str, title: str) -> Optional[str]:
    plt = _plt()
    if plt is None or not series:
        return None
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=C_SURFACE)
    others = [C_BENCH, "#1f9e89", "#8c6bb1", "#b8860b", "#d6456b"]
    j = 0
    for name, r in series.items():
        eq = (1 + r.dropna()).cumprod()
        if name == base:
            ax.plot(eq, color=C_PORT, linewidth=2.2, label=name, zorder=5)
        else:
            ax.plot(eq, color=others[j % len(others)] if j < len(others) else C_MUTED,
                    linewidth=1.2, label=name, alpha=0.9)
            j += 1
    ax.axhline(1, color=C_TEXT2, linewidth=0.6)
    eqs = [(1 + r.dropna()).cumprod() for r in series.values()]
    lo, hi = min(e.min() for e in eqs), max(e.max() for e in eqs)
    log = lo > 0 and hi / lo > 8
    if log:
        ax.set_yscale("log")
    ax.set_title(title, color=C_TEXT, fontsize=11, loc="left")
    ax.set_ylabel("Growth of 1" + (" (log scale)" if log else ""), color=C_TEXT2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=C_TEXT)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return path


def plot_heatmap(t: pd.DataFrame, col: str, path: str, title: str, xlabel: str, ylabel: str,
                 mark=None) -> Optional[str]:
    plt = _plt()
    if plt is None:
        return None
    piv = t.pivot_table(index="p2", columns="p1", values=col)
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 2.5, 0.6 * piv.shape[0] + 2), facecolor=C_SURFACE)
    v = np.nanmax(np.abs(piv.values)) or 1
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-v, vmax=v, aspect="auto", origin="lower")
    ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{c:g}" for c in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if not np.isnan(val):
                is_mark = mark is not None and (piv.columns[j], piv.index[i]) == tuple(mark)
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > 0.6 * v else C_TEXT,
                        fontweight="bold" if is_mark else "normal")
    ax.set_xlabel(xlabel, color=C_TEXT2, fontsize=9)
    ax.set_ylabel(ylabel, color=C_TEXT2, fontsize=9)
    ax.set_title(title, color=C_TEXT, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return path


def plot_bootstrap(b: Dict, path: str, title: str) -> Optional[str]:
    plt = _plt()
    if plt is None or not b:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=C_SURFACE)
    ax.hist(b["draws"], bins=40, color=C_PORT, alpha=0.75, edgecolor=C_SURFACE)
    for x, ls in ((b["lo"], "--"), (b["hi"], "--"), (b["sharpe"], "-")):
        ax.axvline(x, color=C_BENCH if ls == "-" else C_TEXT2, linestyle=ls, linewidth=1.5)
    ax.axvline(0, color=C_TEXT, linewidth=0.8)
    ax.set_title(title, color=C_TEXT, fontsize=10, loc="left")
    ax.set_xlabel("Annualised Sharpe", color=C_TEXT2, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return path


def _img(p, report_dir):
    return f"![]({os.path.relpath(p, report_dir).replace(os.sep, '/')})\n" if p else ""


# ──────────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────────

def write(cfg: dict, runner, pf: PortfolioResult, bench: pd.Series, report_dir: str, fig_dir: str,
          rob: Dict = None, mix: Dict = None) -> str:
    name = cfg.get("name", "backtest")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    pcfg = cfg.get("portfolio", {})
    dcfg = cfg.get("data", {})
    dpy = pf.dpy
    C, I = pf.by_class, pf.by_inst
    L = [f"# {cfg.get('title', name)}\n",
         f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 配置: `{cfg.get('_path', '')}`  ",
         f"> **信号**: `{pf.signal}` | **过滤**: `{pf.filter}` | 周期: {', '.join(runner.timeframes)} "
         f"| 起始: {dcfg.get('start', '')}  ",
         "> 组合: " + ("各类别内等权, 再类别间等权" if pcfg.get("weighting", "class") == "class" else "所有品种等权")
         + (f" | 单品种先缩放到 {pcfg['instrument_vol']:.0%} 波动" if pcfg.get("instrument_vol") else "")
         + (f" | 组合缩放到 {pcfg['target_vol']:.0%} 目标波动" if pf.scaled is not None else "") + "  ",
         "> 执行: 收盘价成交, 1% 风险/ATR 定仓位, 逐根盯市, 止损跳空按开盘价, 信号用复权价、盈亏用真实价格\n"]
    specs = {i.market: i.spec for i in runner.instruments}
    L.append("| 市场 | 手续费 (单边) | 印花税 | 滑点 | 做空 | 每品种本金 |")
    L.append("|---|---:|---:|---:|---|---:|")
    for m, s in specs.items():
        slip = f"{s.slippage_ticks:g} 跳 (最小变动价位)" if s.slippage_ticks > 0 else f"{s.slippage*1e4:.1f}‱"
        fee = f"{s.open_fee*1e4:.1f}‱" if not s.fee_per_lot else f"{s.fee_per_lot:g} 元/手"
        L.append(f"| {m} | {fee} | {s.sell_tax*1e4:.0f}‱ | {slip} "
                 f"| {'可以' if s.allow_short else '不可以'} | {s.initial_capital:,.0f} |")
    per_lot = sorted({i.symbol for i in runner.instruments if i.spec.fee_per_lot})
    no_tick = sorted({i.symbol for i in runner.instruments
                      if i.market == "future" and i.spec.slippage_ticks > 0 and not i.spec.tick_size})
    if per_lot:
        L.append(f"\n按手收手续费的品种: {', '.join(per_lot)}; 其余期货按费率。")
    if no_tick:
        L.append(f"\n没有最小变动价位数据、滑点按比例计算的品种: {', '.join(no_tick)}。")
    L.append("\n---\n")

    # 一、品种
    L.append("## 一、各品种表现\n")
    rows = []
    for key, r in pf.results.items():
        inst = r["_inst"]
        eq = r["equity"].dropna()
        rows.append({"类别": inst.cls, "品种": inst.symbol, "周期": inst.tf,
                     "起始": str(eq.index[0].date()) if len(eq) else "—", "交易": r["n_trades"],
                     "年化%": r["annual_return_pct"], "年化波动%": r["ann_vol_pct"], "夏普": r["sharpe"],
                     "PF": min(r["profit_factor"], 99), "最大回撤%": r["max_drawdown_pct"],
                     "Calmar": r["calmar"], "持仓占比%": r["exposure_pct"],
                     **({"过滤挡掉": str(r.get("blocked", 0))} if pf.filter != "none" else {})})
    L += md_table(pd.DataFrame(rows), FMT)
    if runner.failed:
        L.append(f"\n取数失败或没有成交, 未参与: {', '.join(runner.failed)}")
    L.append("\n---\n")

    # 二、周期
    if pf.by_tf:
        L.append("## 二、各周期单独组合\n")
        tr = [_stats_row(tf, r, dpy) for tf, r in pf.by_tf.items()]
        tr.append(_stats_row("全部周期合并", pf.returns, dpy))
        L += md_table(pd.DataFrame([x for x in tr if x]), FMT, bold_row=len(tr) - 1)
        corr = pd.DataFrame(pf.by_tf).corr()
        L.append("\n各周期组合日收益相关性:\n")
        L += md_table(corr.reset_index().rename(columns={"index": ""}))
        L.append("\n---\n")

    # 三、类别
    if C.shape[1] > 1:
        L.append("## 三、各资产类别 (类别内等权)\n")
        cr = [_stats_row(c, C[c].dropna(), dpy, {"成员数": sum(1 for k in I.columns if k.startswith(c + ":"))})
              for c in C.columns]
        cr.append(_stats_row("组合 (全样本)", pf.returns, dpy, {"成员数": I.shape[1]}))
        full_start = max(C[c].first_valid_index() for c in C.columns)
        cr.append(_stats_row(f"组合 (全部类别到齐后)", pf.returns.loc[full_start:], dpy, {"成员数": I.shape[1]}))
        L += md_table(pd.DataFrame([x for x in cr if x]), {**FMT, "成员数": "{:.0f}"})
        L.append("\n类别开始时间不同时, 早期的组合只由已有数据的类别构成, \"全部类别到齐后\"才是完整的跨资产组合。\n")
        L.append("类别之间的相关性 (策略日收益):\n")
        L += md_table(C.corr().reset_index().rename(columns={"index": ""}))
        w, c = within_cross_corr(I)
        L.append(f"\n品种两两相关性平均: **类内 {w:.2f}**, **跨类 {c:.2f}** (对照 TSMOM 论文 Table 4)。\n")
        L.append("---\n")

    # 四、目标波动
    if pf.scaled is not None:
        L.append(f"## 四、组合目标波动率 {pcfg['target_vol']:.0%}\n")
        t = pd.DataFrame([x for x in (
            _stats_row("原始", pf.returns, dpy, {"平均杠杆": 1.0, "最高杠杆": 1.0}),
            _stats_row("缩放后", pf.scaled, dpy, {"平均杠杆": pf.leverage.mean(), "最高杠杆": pf.leverage.max()}),
        ) if x])
        L += md_table(t, {**FMT, "平均杠杆": "{:.1f}", "最高杠杆": "{:.1f}"})
        L.append("\n杠杆 = 目标波动 / 过去波动 (EWMA, 质心 60 天, 上限 10 倍), 只用过去数据。"
                 "夏普基本不变, 收益和回撤同比例放大。以下各节使用缩放后的组合。\n")
        L.append("---\n")

    port = pf.main
    # 五、压力时期
    stress = []
    for nm, a, b in STRESS_PERIODS:
        p = port.loc[a:b]
        if len(p) < 3:
            continue
        cum = lambda x: ((1 + x.dropna()).prod() - 1) * 100
        stress.append({"时期": nm, "区间": f"{a} ~ {b}", "策略组合%": cum(p), "买入持有%": cum(bench.loc[a:b]),
                       "当时有数据的类别": ", ".join(c for c in C.columns if C[c].loc[a:b].notna().any())})
    if stress:
        L.append("## 五、压力时期\n")
        L += md_table(pd.DataFrame(stress), {"策略组合%": "{:+.1f}", "买入持有%": "{:+.1f}"})
        L.append("\n日期为大致区间, 用来观察趋势策略在危机中的表现。\n")
        L.append("---\n")

    # 六、组合与风险
    L.append("## 六、组合表现与风险特征\n")
    wc = within_cross_corr(I)
    info = {"returns": port, "by_symbol": (C if C.shape[1] > 1 else I).loc[port.index[0]:],
            "stats": ta.stats_from_returns(port, dpy),
            "avg_corr": wc[1] if C.shape[1] > 1 else wc[0], "n_symbols": I.shape[1],
            "grey_note": f"grey: {C.shape[1]} asset classes" if C.shape[1] > 1 else f"grey: {I.shape[1]} instruments",
            "unit_label": f"{C.shape[1]} 个类别、{I.shape[1]} 个品种×周期",
            "corr_label": "跨类别品种两两相关性平均" if C.shape[1] > 1 else "品种两两相关性平均"}
    L += ta.analysis_section(info, bench, dpy, report_dir, fig_dir, name, f"{pf.signal} | {pf.filter}")

    # 七、稳定性
    if rob:
        L += _robustness_section(rob, pf, name, report_dir, fig_dir)

    # 八、和沪深300 组合
    if mix:
        L += _mix_section(mix, name, report_dir, fig_dir)

    path = os.path.join(report_dir, f"{name}_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    out = pd.DataFrame({"strategy": pf.returns, "benchmark": bench})
    if pf.scaled is not None:
        out["strategy_scaled"] = pf.scaled
    out.to_csv(os.path.join(report_dir, f"{name}_returns.csv"), index_label="date")
    return path


def _mix_section(mix, name, report_dir, fig_dir) -> List[str]:
    L = ["---\n", "## 八、和沪深300 组合: 这个策略有没有用\n",
         f"股票: {mix['source']}; 策略: 报告主组合 (缩放后); 区间 {mix['start'].date()} ~ {mix['end'].date()}。\n",
         "- **配置**: 总资金按比例分给股票和策略, 每天再平衡到固定比例 (简化; 按月再平衡结果相近)",
         "- **叠加**: 股票仓位 100% 不变, 另用保证金叠加策略 (期货只占用少量保证金, 实际可行, 但总风险更高)\n"]
    L += md_table(mix["table"], {"股票下跌月份平均月收益%": "{:+.2f}", "最差月份%": "{:+.1f}", **FMT})
    L.append(f"\n策略与沪深300 的相关性: 日度 **{mix['corr_daily']:.2f}**, 月度 **{mix['corr_monthly']:.2f}**"
             " (日度受两个市场收盘时间错位影响, 月度更可靠)。\n")
    t = mix["table"].set_index("组合")
    base = t.loc["股票 100%"]
    best = t.drop(index=["策略 100%"]).sort_values("夏普", ascending=False).iloc[0]
    L.append(f"- 单独持有沪深300: 夏普 {base['夏普']:.2f}, 最大回撤 {base['最大回撤%']:.1f}%")
    L.append(f"- 夏普最高的组合: **{best.name}**, 夏普 {best['夏普']:.2f}, 最大回撤 {best['最大回撤%']:.1f}%")
    L.append("- \"股票下跌月份平均月收益\"越高, 说明策略在股票亏钱的月份补得越多 (对冲作用)。")
    L.append("- 读法: 加入策略后夏普上升、回撤变小, 说明策略有配置价值, 哪怕它单独的收益不高。\n")
    en = {"股票 100%": "CSI300 100%", "策略 100%": "Strategy 100%"}
    pick = {}
    for k, v in mix["curves"].items():
        if k in en:
            pick[en[k]] = v
        elif k == "股票 80% + 策略 20%" or k == "股票 70% + 策略 30%":
            pick[k.replace("股票", "CSI300").replace("策略", "strategy")] = v
        elif k == "股票 100% + 叠加策略 100%":
            pick["CSI300 100% + strategy overlay 100%"] = v
    p = plot_compare(pick, "CSI300 100%", os.path.join(fig_dir, f"{name}_mix_csi300.png"),
                     "CSI300 alone vs CSI300 + strategy")
    L.append(_img(p, report_dir))
    L.append("逐年收益:\n")
    y = mix["yearly"] * 100
    L += md_table(y.reset_index().rename(columns={"index": "年份"}).assign(年份=lambda d: d["年份"].astype(str)),
                  {c: "{:+.1f}" for c in y.columns})
    L.append("")
    return L


def _robustness_section(rob, pf, name, report_dir, fig_dir) -> List[str]:
    L = ["---\n", "## 七、稳定性检验\n",
         f"样本内 / 样本外分界: **{rob['split'].date()}** (样本内挑参数, 样本外检验)。\n"]
    n = 0

    def h(t):
        nonlocal n
        n += 1
        return f"### 7.{n} {t}\n"

    for kind, title in (("signals", "换信号"), ("filters", "换过滤器")):
        d = rob.get(kind)
        if not d or d["table"].empty:
            continue
        L.append(h(title))
        L += md_table(d["table"], FMT)
        base = pf.signal if kind == "signals" else pf.filter
        p = plot_compare(d["returns"], base, os.path.join(fig_dir, f"{name}_{kind}.png"),
                         f"{'Signals' if kind == 'signals' else 'Filters'}: cumulative return")
        L.append("\n" + _img(p, report_dir))
        if d["corr"] is not None:
            L.append("日收益相关性 (越低, 组合在一起越有分散效果):\n")
            L += md_table(d["corr"].reset_index().rename(columns={"index": ""}))
            L.append("")
        if kind == "filters":
            L.append("过滤器只是挡掉部分入场, 如果\"被挡掉的入场\"很多而夏普没提高, 说明它挡掉的并不是坏交易。\n")

    g = rob.get("grid")
    if g:
        L.append(h("参数网格"))
        t = g["table"]
        cols = [c for c in ("label", "夏普", "样本内夏普", "样本外夏普") if c in t.columns]
        if g["n_params"] == 2:
            base = g["base"]
            mark = (base["p1"], base["p2"]) if base is not None else None
            axes = {"donchian": ("entry days", "exit days"), "ma": ("fast MA days", "slow MA days")}
            xl, yl = axes.get(pf.signal.split(":")[0], ("param 1", "param 2"))
            p = plot_heatmap(t, "夏普", os.path.join(fig_dir, f"{name}_grid.png"),
                             "Sharpe by parameter (full sample, bold = current)", xl, yl, mark)
            L.append(_img(p, report_dir))
        L += md_table(t[cols].sort_values("夏普", ascending=False).reset_index(drop=True))
        b = g["best_is"]
        L.append(f"\n- 夏普为正的参数组合占 **{g['share_positive']:.0%}**; 各参数样本外夏普中位数 **{g['oos_median']:.2f}**")
        L.append(f"- 样本内最好的参数 `{b['label']}`: 样本内 {b['样本内夏普']:.2f} → 样本外 **{b['样本外夏普']:.2f}**")
        if g["base"] is not None:
            L.append(f"- 当前参数 `{g['base']['label']}`: 样本内 {g['base']['样本内夏普']:.2f} → 样本外 {g['base']['样本外夏普']:.2f}")
        L.append("- 读法: 一片颜色相近的高原 = 参数不敏感, 结果可信; 只有一个格子特别亮 = 大概率是过拟合。"
                 "样本内最好的参数到样本外掉得越多, 说明挑参数的收益越不可靠。\n")

    y = rob.get("yearly")
    if y is not None and not y.empty:
        L.append(h("分年度"))
        L += md_table(y.assign(年份=y["年份"].astype(str)), FMT)
        L.append(f"\n赚钱的年份: {int((y['收益%'] > 0).sum())} / {len(y)}。\n")

    b = rob.get("bootstrap")
    if b:
        L.append(h("自助法夏普置信区间"))
        L.append(f"按 {b['block']} 天分块有放回重抽组合日收益 {b['n']} 次 (保留波动聚集和短期自相关):\n")
        L.append(f"- 夏普 **{b['sharpe']:.2f}**, 95% 区间 **[{b['lo']:.2f}, {b['hi']:.2f}]**")
        L.append(f"- 重抽样本中夏普 ≤ 0 的比例: **{b['p_le_0']:.1%}**"
                 + (" (区间不含 0, 收益不太可能纯属运气)" if b["lo"] > 0 else " (区间包含 0, 不能排除运气)"))
        L.append("- 注意: 这只衡量样本内的抽样误差, 不能纠正\"试了很多策略挑最好的\"带来的偏差。\n")
        p = plot_bootstrap(b, os.path.join(fig_dir, f"{name}_bootstrap.png"), "Block-bootstrap Sharpe distribution")
        L.append(_img(p, report_dir))

    c = rob.get("costs")
    if c is not None and not c.empty:
        L.append(h("成本敏感性 (滑点倍数)"))
        L += md_table(c, FMT)
        L.append("\n滑点翻倍后夏普还剩多少, 决定了策略在真实交易中能不能活下来。\n")
    return L
