#!/usr/bin/env python3
"""
组合与风险分析 (参考 Moskowitz, Ooi & Pedersen 2012, Table 3 与 Fig. 4)
======================================================================
1. 组合: 每个品种单独跑海龟 (已按 1%/ATR 做了风险均衡), 再把日收益等权平均,
   相当于 TSMOM 论文里的 diversified 组合. 当天还没开始交易的品种不参与平均.
2. 基准: 同一批品种的等权买入持有 (相当于论文里的 passive long).
3. 风险特征 (月度, Newey-West 标准误):
     CAPM   : 组合 = α + β·基准                → α 是扣除"被动做多"后的超额收益
     Smile  : 组合 = a + b1·基准 + b2·基准²      → b2 > 0 说明大涨大跌时都赚钱
     波动率 : 组合 = a + c·基准波动率 + d·高波动(前20%)哑变量
   注: 高波动阈值用全样本分位数, 这是描述性分析 (论文同样做法), 不能直接当交易规则.
"""

import math
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 图表配色 (经过色盲可辨性校验): 组合=蓝, 基准=橙, 其余为中性灰
C_PORT, C_BENCH = "#2a78d6", "#eb6834"
C_MUTED, C_GRID, C_TEXT, C_TEXT2, C_SURFACE = "#c3c2b7", "#e6e5e0", "#0b0b0b", "#52514e", "#fcfcfb"


# ──────────────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────────────

def _monthly(r: pd.Series) -> pd.Series:
    """日收益 → 月收益 (复利)."""
    try:
        return (1 + r).resample("ME").prod() - 1
    except ValueError:                       # pandas < 2.2
        return (1 + r).resample("M").prod() - 1


def stats_from_returns(r: pd.Series, days_per_year: float) -> Dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / days_per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else 0
    vol = r.std() * math.sqrt(days_per_year)
    dd = (eq / eq.cummax() - 1).min()
    downside = r[r < 0].std() * math.sqrt(days_per_year)
    return {
        "cagr_pct": cagr * 100,
        "vol_pct": vol * 100,
        "sharpe": r.mean() * days_per_year / vol if vol > 0 else 0,
        "sortino": r.mean() * days_per_year / downside if downside > 0 else 0,
        "max_dd_pct": dd * 100,
        "calmar": cagr / abs(dd) if dd < 0 else 0,
        "start": r.index[0], "end": r.index[-1],
    }


def ols_newey_west(y: pd.Series, X: pd.DataFrame, lags: int = 3) -> Dict:
    """带常数项的 OLS, Newey-West (Bartlett 核) 标准误. 不依赖 statsmodels."""
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    yv = d["y"].to_numpy()
    Xv = np.column_stack([np.ones(len(d)), d[X.columns].to_numpy()])
    n, k = Xv.shape
    if n <= k + 2:
        return {}
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    e = yv - Xv @ beta
    Xe = Xv * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    r2 = 1 - (e @ e) / ((yv - yv.mean()) @ (yv - yv.mean())) if yv.std() > 0 else 0
    names = ["const"] + list(X.columns)
    return {"coef": dict(zip(names, beta)), "t": dict(zip(names, beta / np.where(se > 0, se, np.nan))),
            "r2": r2, "n": n}


# ──────────────────────────────────────────────────
# 组合与基准
# ──────────────────────────────────────────────────

def build_portfolio(results: Dict[str, Dict], days_per_year: float) -> Optional[Dict]:
    """results: {品种: run_backtest 结果}. 返回组合日收益、指标、相关性."""
    rets = {s: r["daily_returns"] for s, r in results.items()
            if r and r.get("n_trades", 0) > 0 and len(r.get("daily_returns", [])) > 1}
    if not rets:
        return None
    R = pd.DataFrame(rets).sort_index()
    port = R.mean(axis=1, skipna=True).dropna()       # 当天有数据的品种等权
    corr = R.corr()
    k = len(corr)
    avg_corr = (corr.values.sum() - k) / (k * (k - 1)) if k > 1 else np.nan
    return {"returns": port, "by_symbol": R, "stats": stats_from_returns(port, days_per_year),
            "avg_corr": avg_corr, "n_symbols": k}


def price_returns(px) -> pd.Series:
    """
    买入持有的日收益.
    px 可以是收盘价 Series, 也可以是带 raw_close / scale 列的 DataFrame (复权数据).
    ⚠️ 差值复权的期货价格可能接近 0 甚至为负, 不能直接 pct_change;
       正确做法: 当天价格变动 (复权价的点数 × scale = 真实点数) ÷ 前一天的真实价格.
    """
    if isinstance(px, pd.DataFrame):
        d = px.dropna(subset=["close"]).resample("1D").last().dropna(subset=["close"])
        if "raw_close" in d.columns and "scale" in d.columns:
            r = d["close"].diff() * d["scale"] / d["raw_close"].shift(1)
            return r.replace([np.inf, -np.inf], np.nan).dropna()
        px = d["close"]
    daily = px[px > 0].dropna().resample("1D").last().dropna()
    return daily.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def build_benchmark(prices: Dict, index: pd.DatetimeIndex) -> pd.Series:
    """同一批品种的等权买入持有日收益 (prices: {品种: 行情 DataFrame 或收盘价 Series})."""
    rets = {}
    for s, px in prices.items():
        if px is None or len(px) < 2:
            continue
        rets[s] = price_returns(px)
    if not rets:
        return pd.Series(dtype=float)
    b = pd.DataFrame(rets).mean(axis=1, skipna=True)
    return b.reindex(index).fillna(0.0)


def yearly_returns(r: pd.Series) -> pd.Series:
    return (1 + r).groupby(r.index.year).prod() - 1


# ──────────────────────────────────────────────────
# 风险特征回归
# ──────────────────────────────────────────────────

def risk_regressions(port: pd.Series, bench: pd.Series, days_per_year: float) -> Dict:
    pm, bm = _monthly(port), _monthly(bench)
    # 基准当月的已实现波动率 (年化)
    try:
        rv = bench.resample("ME").std() * math.sqrt(days_per_year)
    except ValueError:
        rv = bench.resample("M").std() * math.sqrt(days_per_year)
    df = pd.concat({"p": pm, "m": bm, "rv": rv}, axis=1).dropna()
    if len(df) < 12:
        return {}
    df["m2"] = df["m"] ** 2
    df["rv_top"] = (df["rv"] >= df["rv"].quantile(0.8)).astype(float)
    return {
        "capm": ols_newey_west(df["p"], df[["m"]]),
        "smile": ols_newey_west(df["p"], df[["m", "m2"]]),
        "vol": ols_newey_west(df["p"], df[["rv", "rv_top"]]),
        "monthly": df,
    }


# ──────────────────────────────────────────────────
# 图表
# ──────────────────────────────────────────────────

def _style(ax):
    ax.set_facecolor(C_SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=8)
    ax.grid(True, color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_portfolio(pf: Dict, bench: pd.Series, path: str, title: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    port = pf["returns"]
    eq = (1 + port).cumprod()
    beq = (1 + bench.reindex(port.index).fillna(0)).cumprod()
    dd = (eq / eq.cummax() - 1) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]}, facecolor=C_SURFACE)
    for s in pf["by_symbol"].columns:                         # 各品种: 浅灰背景线
        r = pf["by_symbol"][s].dropna()
        if len(r) > 1:
            ax1.plot((1 + r).cumprod(), color=C_MUTED, linewidth=0.7, alpha=0.6, zorder=1)
    h_port, = ax1.plot(eq, color=C_PORT, linewidth=2, label="Turtle portfolio", zorder=3)
    h_bench, = ax1.plot(beq, color=C_BENCH, linewidth=2, label="Buy & hold (equal weight)", zorder=2)
    for series, color, name in ((eq, C_PORT, "Turtle"), (beq, C_BENCH, "Buy & hold")):
        ax1.annotate(f"{name} {series.iloc[-1]:.2f}x", (series.index[-1], series.iloc[-1]),
                     xytext=(6, 0), textcoords="offset points", color=C_TEXT, fontsize=8, va="center")
    ax1.axhline(1, color=C_TEXT2, linewidth=0.6)
    lo, hi = min(eq.min(), beq.min()), max(eq.max(), beq.max())
    log = lo > 0 and hi / lo > 8                               # 涨跌幅度很大时用对数坐标
    if log:
        ax1.set_yscale("log")
    ax1.set_ylabel("Growth of 1" + (" (log scale)" if log else ""), color=C_TEXT2, fontsize=9)
    ax1.set_title(title, color=C_TEXT, fontsize=11, loc="left")
    ax1.legend(handles=[h_port, h_bench], frameon=False, fontsize=8, loc="upper left", labelcolor=C_TEXT)
    ax1.text(0.99, 0.02, pf.get("grey_note", f"grey: {pf['n_symbols']} individual instruments"),
             transform=ax1.transAxes, ha="right", fontsize=7, color=C_TEXT2)

    ax2.fill_between(dd.index, dd.values, 0, color=C_PORT, alpha=0.25, linewidth=0)
    ax2.plot(dd, color=C_PORT, linewidth=1)
    ax2.set_ylabel("Drawdown %", color=C_TEXT2, fontsize=9)
    for ax in (ax1, ax2):
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return path


def plot_smile(risk: Dict, path: str, title: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    df = risk["monthly"]
    x, y = df["m"] * 100, df["p"] * 100
    c = risk["smile"]["coef"]
    xs = np.linspace(x.min(), x.max(), 100)
    ys = (c["const"] + c["m"] * xs / 100 + c["m2"] * (xs / 100) ** 2) * 100

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=C_SURFACE)
    ax.axhline(0, color=C_TEXT2, linewidth=0.6)
    ax.axvline(0, color=C_TEXT2, linewidth=0.6)
    ax.scatter(x, y, s=36, color=C_PORT, alpha=0.75, edgecolor=C_SURFACE, linewidth=1.5,
               label="Monthly return", zorder=3)
    ax.plot(xs, ys, color=C_BENCH, linewidth=2, label="Quadratic fit", zorder=4)
    t2 = risk["smile"]["t"]["m2"]
    ax.set_title(f"{title}\nsquared-market coef t = {t2:.2f}", color=C_TEXT, fontsize=10, loc="left")
    ax.set_xlabel("Buy & hold monthly return (%)", color=C_TEXT2, fontsize=9)
    ax.set_ylabel("Turtle portfolio monthly return (%)", color=C_TEXT2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=C_TEXT)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────
# 报告段落
# ──────────────────────────────────────────────────

def _fmt_t(reg, name, pct=False, scale=1.0):
    if not reg or name not in reg["coef"]:
        return "—"
    v = reg["coef"][name] * scale
    s = f"{v*100:.2f}%" if pct else f"{v:.3f}"
    return f"{s} ({reg['t'][name]:.2f})"


def portfolio_summary_table(rows: List[tuple]) -> List[str]:
    """rows: [(名称, portfolio dict)] → 多个组合的对比表."""
    L = ["| 组合 | 品种数 | 年化收益% | 年化波动% | 夏普 | Sortino | 最大回撤% | Calmar | 平均相关性 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, pf in rows:
        if not pf or not pf["stats"]:
            continue
        s = pf["stats"]
        L.append(f"| {name} | {pf['n_symbols']} | {s['cagr_pct']:+.1f} | {s['vol_pct']:.1f} "
                 f"| {s['sharpe']:.2f} | {s['sortino']:.2f} | {s['max_dd_pct']:.1f} "
                 f"| {s['calmar']:.2f} | {pf['avg_corr']:.2f} |")
    return L


def analysis_section(pf: Dict, bench: pd.Series, days_per_year: float, report_dir: str,
                     fig_dir: str, tag: str, title: str) -> List[str]:
    """完整的组合 + 风险分析段落 (含两张图)."""
    L = []
    if not pf or not pf["stats"]:
        return ["组合: 没有可用的品种收益。\n"]
    port = pf["returns"]
    bench = bench.reindex(port.index).fillna(0.0)
    sp, sb = pf["stats"], stats_from_returns(bench, days_per_year)

    L.append(f"#### 组合表现 ({title}; {pf.get('unit_label', str(pf['n_symbols']) + ' 个品种等权')}, "
             f"{sp['start'].date()} ~ {sp['end'].date()})\n")
    L.append("| 指标 | 海龟组合 | 基准: 等权买入持有 |")
    L.append("|---|---:|---:|")
    for key, label, f in [("cagr_pct", "年化收益 %", "{:+.1f}"), ("vol_pct", "年化波动 %", "{:.1f}"),
                          ("sharpe", "夏普", "{:.2f}"), ("sortino", "Sortino", "{:.2f}"),
                          ("max_dd_pct", "最大回撤 %", "{:.1f}"), ("calmar", "Calmar", "{:.2f}")]:
        L.append(f"| {label} | {f.format(sp[key])} | {f.format(sb.get(key, 0))} |")
    L.append(f"\n{pf.get('corr_label', '品种间日收益平均相关性')}: {pf['avg_corr']:.2f}"
             "  (越低, 分散化降波动的效果越好)\n")

    yr = pd.concat({"组合": yearly_returns(port), "基准": yearly_returns(bench)}, axis=1)
    L.append("| 年份 | 海龟组合 | 等权持有 |")
    L.append("|---|---:|---:|")
    for y, row in yr.iterrows():
        L.append(f"| {y} | {row['组合']*100:+.1f}% | {row['基准']*100:+.1f}% |")
    L.append("")

    os.makedirs(fig_dir, exist_ok=True)
    p1 = plot_portfolio(pf, bench, os.path.join(fig_dir, f"turtle_portfolio_{tag}.png"),
                        f"Turtle portfolio vs buy & hold - {tag}")
    if p1:
        L.append(f"![组合净值]({os.path.relpath(p1, report_dir).replace(os.sep, '/')})\n")

    risk = risk_regressions(port, bench, days_per_year)
    L.append(f"#### 风险特征 (月度回归, 括号内为 Newey-West t 值)\n")
    if not risk:
        L.append("样本不足 12 个月, 跳过。\n")
        return L
    capm, smile, vol = risk["capm"], risk["smile"], risk["vol"]
    L.append("| 回归 | α (月) | 基准收益 | 基准收益² | 基准波动率 | 高波动哑变量 | R² |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    L.append(f"| CAPM | {_fmt_t(capm,'const',pct=True)} | {_fmt_t(capm,'m')} | | | | {capm['r2']:.2f} |")
    L.append(f"| Smile | | {_fmt_t(smile,'m')} | {_fmt_t(smile,'m2')} | | | {smile['r2']:.2f} |")
    L.append(f"| 波动率状态 | | | | {_fmt_t(vol,'rv')} | {_fmt_t(vol,'rv_top',pct=True)} | {vol['r2']:.2f} |")
    L.append(f"\n样本 {capm['n']} 个月。读法:")
    a, ta = capm["coef"]["const"], capm["t"]["const"]
    L.append(f"- **CAPM α**: 月均 {a*100:.2f}% (年化约 {a*1200:.1f}%), t = {ta:.2f}。"
             + ("显著, 扣除被动做多后仍有超额收益。" if abs(ta) >= 2 else "不显著, 收益大部分可由被动做多解释。"))
    t2 = smile["t"]["m2"]
    L.append(f"- **Smile**: 基准收益平方项 t = {t2:.2f}。"
             + ("显著为正: 大涨大跌时都赚钱, 类似 TSMOM 的 smile。" if t2 >= 2
                else "不显著: 没有明显的\"危机阿尔法\"。" if t2 > -2 else "显著为负: 极端行情反而亏钱。"))
    tv, td = vol["t"]["rv"], vol["t"]["rv_top"]
    L.append(f"- **波动率状态**: 基准波动率 t = {tv:.2f}, 高波动哑变量 t = {td:.2f}。"
             + ("策略收益与市场波动率状态有关。" if max(abs(tv), abs(td)) >= 2
                else "与市场波动率状态关系不显著 (这也说明按波动率择时的 GARCH 过滤器改善空间有限)。"))
    L.append("- CAPM 的截距是 α (基准是可交易的等权持有组合); Smile 和波动率回归的截距没有 α 含义, 不报告。")
    L.append("- 高波动阈值用全样本 80% 分位数, 是描述性分析, 不能直接当交易规则。\n")

    p2 = plot_smile(risk, os.path.join(fig_dir, f"turtle_smile_{tag}.png"),
                    f"Turtle monthly returns vs buy & hold - {tag}")
    if p2:
        L.append(f"![Smile]({os.path.relpath(p2, report_dir).replace(os.sep, '/')})\n")
    return L
