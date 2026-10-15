"""
在本地近似 WorldQuant BRAIN 的设置, 跑同一个 alpha
===================================================
Region USA, 股票池用标普 1500 近似 TOP3000, Truncation 0.08, 不计成本 (和 BRAIN 一样).
对比 Delay / Decay / 中性化方式, 指标按 BRAIN 口径 (Sharpe / Returns / Turnover / Fitness / Margin).

运行 (在 quant 根目录):
    python strategies/xs_alpha/us_brain.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backtest import cross_section as xs                          # noqa: E402
from backtest.report import md_table                              # noqa: E402
from data.paths import REPORTS_DIR                                # noqa: E402
from data.us_stock import load_us_panel, sp1500_universe          # noqa: E402

TRUNCATION = 0.08


def clean_breaks(P, max_jump=2.0):
    """
    价格 ≤ 0 记为缺失; 单日涨跌超过 max_jump (200%) 视为数据断点 (破产重组、反向拆股没复权),
    把断点之前的价格作废, 当作新上市. 例: CHRD 2020-11-20 重组前后的股票被 Yahoo 拼在一起, 单日 +25733%.
    """
    P = {k: v.where(v > 0) if k != "volume" else v for k, v in P.items()}
    ret = P["close"].pct_change(fill_method=None)
    jumps = ret.abs() > max_jump
    for t in jumps.columns[jumps.any()]:
        last = jumps.index[jumps[t]][-1]
        for k in P:
            P[k].loc[P[k].index < last, t] = float("nan")
        print(f"  [!] {t} {last.date()} 单日 {ret.loc[last, t]:+.0%}, 作废之前的价格")
    return P


def compute_alpha(P, adv="dollar"):
    """adv='dollar': BRAIN 口径, adv20 = 20 日平均成交额; adv='shares': 20 日平均成交量."""
    o, h, l, c, v = P["open"], P["high"], P["low"], P["close"], P["volume"]
    ret = c.pct_change(fill_method=None)
    a = -(c - o) / (h - l).where(h > l)
    adv20 = xs.ts_mean(c * v, 20) if adv == "dollar" else xs.ts_mean(v, 20)
    raw = a * xs.rank(v / adv20) * xs.rank(xs.ts_std(ret, 20))
    return xs.ts_decay_linear(raw, 5)


def main():
    uni = sp1500_universe()
    print(f"股票池: 标普 1500 当前成分股 {len(uni)} 只, 下载 / 读取行情 ...")
    P = load_us_panel(uni["ticker"].tolist(), "2014-10-01")
    P = clean_breaks(P)
    print(f"面板: {P['close'].shape[0]} 天 × {P['close'].shape[1]} 只")
    meta = uni.set_index("ticker")
    groups = {"market": None, "sector": meta["sector"], "subindustry": meta["subindustry"]}

    alphas = {"adv20=成交额 (BRAIN)": compute_alpha(P, "dollar"), "adv20=成交量": compute_alpha(P, "shares")}
    periods = {"2015 至今": "2015-01-01", "近 5 年": str(P["close"].index[-1] - pd.DateOffset(years=5))[:10]}

    rows = []
    for aname, alpha in alphas.items():
        for delay in (1, 0):
            for decay in (0, 4):
                for gname, g in groups.items():
                    if aname != "adv20=成交额 (BRAIN)" and (delay, gname) != (1, "subindustry"):
                        continue                                         # 对照组只跑默认设置
                    r = xs.brain_simulate(alpha, P, delay, decay, g, TRUNCATION)
                    for pname, start in periods.items():
                        m = xs.brain_metrics(r.loc[start:])
                        rows.append({"alpha": aname, "Delay": delay, "Decay": decay, "中性化": gname,
                                     "区间": pname, **m})
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(tab.round(2).to_string(index=False))

    out = REPORTS_DIR / "xs_alpha"
    out.mkdir(parents=True, exist_ok=True)
    L = ["# 本地近似 BRAIN: USA, 标普 1500, Truncation 0.08", "",
         "```", "a     = -(close - open) / (high - low)",
         "alpha = ts_decay_linear( a * rank(volume / adv20) * rank(ts_std(returns, 20)), 5 )", "```", "",
         "- 流程: delay → decay → 中性化 → 按 |alpha| 归一 → 单股不超过 8% → 收盘调仓; **不计交易成本** (与 BRAIN 一致)",
         "- Returns 相对一半账面; Fitness = Sharpe × sqrt(|Returns| / max(Turnover, 12.5%)); "
         "Margin = 每交易 1 美元赚多少 (bp)",
         "- 与 BRAIN 的差距: 当前成分股 (幸存者偏差)、当前 GICS 分类、Yahoo 复权数据", "",
         *md_table(tab, {"Delay": "{:.0f}", "Decay": "{:.0f}"})]
    path = out / "us_brain_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n报告: {path}")


if __name__ == "__main__":
    main()
