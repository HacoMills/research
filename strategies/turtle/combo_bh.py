"""
趋势 + 买入持有 组合
====================
趋势策略和同品种买入持有几乎不相关, 两者按滚动风险平价组合:
  1. 每边的权重 ∝ 1 / 过去 250 天的 EWMA 波动 (滞后一天, 无前视)
  2. 组合整体缩放到目标波动, 杠杆设上限
同时比较 "包含汇率" 和 "去掉汇率" (汇率买入持有的方向是任意的, 没有风险溢价).

运行 (在 quant 根目录):
    python strategies/turtle/combo_bh.py
"""

import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # 同目录的 fx_carry

from backtest.__main__ import load_config                         # noqa: E402
from backtest.portfolio import Runner, combine, price_returns, vol_scale  # noqa: E402
from backtest.report import md_table                              # noqa: E402
from backtest.signals import parse_signal                         # noqa: E402
from data.paths import REPORTS_DIR                                # noqa: E402

CONFIG = "strategies/turtle/configs/cross_asset.toml"
TARGET_VOL, COM, MAX_LEV = 0.10, 250, 1.5
START = "2019-03-01"                       # 四类资产都有数据以后, 三种方案用同一区间
PERIODS = [("2019.3–2022.6", "2019-03", "2022-06"), ("2022.7–2026.9", "2022-07", "2026-09"), ("全部", "2019-03", "2026-09")]


def risk_parity(R: pd.DataFrame, com=COM, target=TARGET_VOL, cap=MAX_LEV):
    vol = R.ewm(com=com, min_periods=60).std()
    w = (1 / vol).div((1 / vol).sum(axis=1), axis=0).shift(1)
    c = (w * R).sum(axis=1)
    cv = c.ewm(com=com, min_periods=60).std() * math.sqrt(252)
    lev = (target / cv).clip(upper=cap).shift(1)
    return c * lev, w, lev


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {"夏普": r.mean() / r.std() * math.sqrt(252), "年化%": r.mean() * 252 * 100,
            "波动%": r.std() * math.sqrt(252) * 100, "最大回撤%": (eq / eq.cummax() - 1).min() * 100}


def main():
    cfg = load_config(CONFIG)
    cfg.setdefault("robustness", {})["enabled"] = False
    with redirect_stdout(io.StringIO()):                          # 数据加载日志太长, 不打印
        runner = Runner(cfg).load()
        pf = runner.run(parse_signal(cfg["strategy"]["signal"]), verbose=False)

    by_cls = {}
    for key, r in pf.results.items():
        by_cls.setdefault(r["_inst"].cls, {})[key] = r["daily_returns"]
    bh_cls = {cls: {s: price_returns(df) for s, df in d.items()} for cls, d in runner.prices.items()}

    all_cls = list(by_cls)
    no_fx = [c for c in all_cls if c != "汇率"]
    sleeves = {
        "趋势(含汇率)": vol_scale(combine(by_cls)[0], TARGET_VOL, 252)[0],
        "趋势(去汇率)": vol_scale(combine({c: by_cls[c] for c in no_fx})[0], TARGET_VOL, 252)[0],
        "买入持有(含汇率)": combine(bh_cls)[0],
        "买入持有(去汇率)": combine({c: bh_cls[c] for c in no_fx})[0],
        "外汇利差": load_carry(),
    }
    # A/B/C: 趋势 + 买入持有. B 是没有事后挑选的版本 (买入持有去掉汇率有事前理由: 没有风险溢价);
    # C 连趋势里的汇率也去掉, 是看到结果之后的选择, 只能当参考.
    # D/E: 用外汇利差 (汇率真正的风险溢价, 规则事先固定) 代替 / 补充汇率趋势.
    variants = {
        "A 全部含汇率": ["趋势(含汇率)", "买入持有(含汇率)"],
        "B 趋势含汇率 + 买入持有去汇率": ["趋势(含汇率)", "买入持有(去汇率)"],
        "C 都去掉汇率 (事后挑选)": ["趋势(去汇率)", "买入持有(去汇率)"],
        "D 趋势去汇率 + 买入持有 + 外汇利差": ["趋势(去汇率)", "买入持有(去汇率)", "外汇利差"],
        "E 趋势含汇率 + 买入持有 + 外汇利差": ["趋势(含汇率)", "买入持有(去汇率)", "外汇利差"],
    }
    rows, wrows, combos = [], [], {}
    for label, names in variants.items():
        # 不能 dropna: 只有一边交易的日子 (国内节假日外汇照常交易) 会被整天删掉, 丢掉那天的收益
        R = pd.concat({n: sleeves[n] for n in names}, axis=1)
        R = R.loc[R.index >= START].fillna(0.0)
        combo, w, lev = risk_parity(R)
        combos[label] = combo
        for pname, s, e in PERIODS:
            for n in names:
                x = R[n] if not n.startswith("买入持有") else to_target_vol(R[n])
                rows.append({"方案": label, "区间": pname, "": n, **stats(x.loc[s:e])})
            rows.append({"方案": label, "区间": pname, "": "组合", **stats(combo.loc[s:e])})
        wrows.append({"方案": label, **{f"{n} 资金权重": w[n].loc[START:].mean() for n in names},
                      "杠杆中位数": lev.loc[START:].median()})

    tab, wtab = pd.DataFrame(rows), pd.DataFrame(wrows)
    m = (1 + pd.concat(sleeves, axis=1).loc[START:].fillna(0)).resample("ME").prod() - 1
    corr = m.corr().round(2).reset_index().rename(columns={"index": ""})
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(corr.to_string(index=False)); print()
    print(wtab.round(2).to_string(index=False)); print()
    print(tab[tab[""] == "组合"].round(2).to_string(index=False))

    shown = ["B 趋势含汇率 + 买入持有去汇率", "D 趋势去汇率 + 买入持有 + 外汇利差", "E 趋势含汇率 + 买入持有 + 外汇利差"]
    fig_path = plot_combos({k: combos[k] for k in shown})

    L = ["# 趋势 + 买入持有 (+ 外汇利差) 组合", "",
         f"- 趋势: `{cfg['strategy']['signal']}`, 缩放到 {TARGET_VOL:.0%} 波动; 买入持有: 同样的品种和类别权重; "
         "外汇利差: `fx_carry.py` (规则事先固定)",
         f"- 组合: 滚动风险平价 (EWMA 质心 {COM} 天, 滞后一天), 整体缩放到 {TARGET_VOL:.0%}, 杠杆上限 {MAX_LEV}",
         "- 汇率的买入持有方向是任意的, 没有风险溢价: B 只在买入持有里去掉汇率 (事前理由); C 连趋势里的汇率也去掉 (事后挑选, 仅供参考)",
         f"- 区间从 {START} 开始 (国内期货 2018 年才有数据, 四类资产都到齐以后)", "",
         f"![净值、回撤、滚动夏普](../../figures/turtle/{fig_path.name})", "",
         "## 月度相关性", "", *md_table(corr), "",
         "## 资金权重 (平均) 与杠杆", "", *md_table(wtab), "",
         "## 各方案表现", "", *md_table(tab)]
    out = REPORTS_DIR / "turtle" / "combo_bh_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"\n报告: {out}\n图:   {fig_path}")


def load_carry() -> pd.Series:
    """外汇利差策略的日收益 (已缩放到 10% 波动). 没有就先跑一遍 fx_carry.py."""
    path = REPORTS_DIR / "turtle" / "fx_carry_returns.csv"
    if not path.exists():
        import fx_carry
        fx_carry.main()
    return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]


def to_target_vol(r: pd.Series) -> pd.Series:
    """买入持有本身波动较低, 单独展示时也缩放到目标波动, 方便和趋势比较 (只用过去的波动)."""
    vol = r.ewm(com=COM, min_periods=60).std() * math.sqrt(252)
    return r * (TARGET_VOL / vol).clip(upper=3).shift(1)


def plot_combos(combos: dict) -> Path:
    """三联图: 净值 / 回撤 / 滚动一年夏普. 每条线的标签里带全区间夏普和最大回撤."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from backtest.analytics import C_BENCH, C_GRID, C_PORT, C_SURFACE, C_TEXT, C_TEXT2, _style
    from data.paths import FIGURES_DIR

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = [C_PORT, C_BENCH, "#1baf7a"]          # 分类色前三位 (已通过色盲校验), 顺序固定
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]})
    fig.patch.set_facecolor(C_SURFACE)
    ax_eq, ax_dd, ax_sh = axes
    for (label, r), color in zip(combos.items(), colors):
        r = r.loc[PERIODS[-1][1]:PERIODS[-1][2]].dropna()        # 与表格 "全部" 同一区间
        eq = (1 + r).cumprod()
        dd = (eq / eq.cummax() - 1) * 100
        s = stats(r)
        short = label.split(" ", 1)[0]
        name = f"{label}  夏普 {s['夏普']:.2f} · 最大回撤 {s['最大回撤%']:.1f}%"
        ax_eq.plot(eq, color=color, lw=2, label=name)
        ax_eq.annotate(short, (eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                       va="center", fontsize=9, color=C_TEXT)           # 直接标签, 文字用正文色
        ax_dd.plot(dd, color=color, lw=1.6)
        roll = r.rolling(252).mean() / r.rolling(252).std() * math.sqrt(252)
        ax_sh.plot(roll, color=color, lw=1.6)
    ax_eq.set_ylabel("净值 (起点 = 1)", color=C_TEXT2)
    ax_eq.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C_TEXT)
    ax_eq.set_title(f"组合对比  {PERIODS[-1][1]} ~ {PERIODS[-1][2]} · 滚动风险平价 · 目标波动 {TARGET_VOL:.0%}",
                    loc="left", fontsize=11, color=C_TEXT)
    ax_dd.set_ylabel("回撤 %", color=C_TEXT2)
    ax_sh.axhline(0, color=C_GRID, lw=1)
    ax_sh.set_ylabel("滚动一年夏普", color=C_TEXT2)
    for ax in axes:
        _style(ax)
        ax.grid(axis="y", color=C_GRID, lw=0.6)
    fig.tight_layout()
    out = FIGURES_DIR / "turtle"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "combo_bh.png"
    fig.savefig(path, dpi=140, facecolor=C_SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
