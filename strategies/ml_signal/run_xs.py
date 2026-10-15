"""
机器学习截面 alpha: 沪深300 成分股
===================================
15 个量价特征 (截面排名) → 模型 → 预测下一期收益的截面排名 → 多空组合.
和手写的 alpha (strategies/xs_alpha) 放在同一个样本外区间比较.

运行 (在 quant 根目录):
    python strategies/ml_signal/run_xs.py                      # 全部模型, 预测 1 天
    python strategies/ml_signal/run_xs.py --models ridge,xgb --horizon 5
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies" / "xs_alpha"))

import run as hand                                                # noqa: E402  手写 alpha
from backtest import cross_section as xs                          # noqa: E402
from backtest import ml                                           # noqa: E402
from backtest.analytics import stats_from_returns                 # noqa: E402
from backtest.report import md_table                              # noqa: E402
from data.paths import REPORTS_DIR                                # noqa: E402

NAME = "ml_signal"


def evaluate(label, alpha, P, tradable, start, horizon, cost):
    ic = xs.rank_ic(alpha, P, horizon)
    ic = ic[ic.index >= start]
    w = xs.long_short_weights(alpha, tradable)
    w = xs.rebalance_every(w, horizon)                 # 预测 h 天就每 h 天调一次仓
    r = xs.portfolio_returns(w, P, cost, "open")
    r = r.loc[r.index >= start]
    row = {"": label, "IC均值": ic.mean(), "ICIR(年化)": ic.mean() / ic.std() * (252 / horizon) ** 0.5,
           "日均换手%": r["turnover"].mean() * 100}
    for part, sl in (("全部", slice(None)), ("2022起", slice("2022-01-01", None))):
        row[f"{part} 扣费前夏普"] = stats_from_returns(r["gross"].loc[sl], xs.DAYS_PER_YEAR)["sharpe"]
        row[f"{part} 扣费后夏普"] = stats_from_returns(r["net"].loc[sl], xs.DAYS_PER_YEAR)["sharpe"]
    row["盈亏平衡成本bp"] = r["gross"].mean() / max(r["turnover"].mean(), 1e-12) * 1e4
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ridge,xgb,lgbm,mlp")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--cost", type=float, default=0.0012)
    args = ap.parse_args()

    P = xs.load_panel(hand.csi300_codes(), "2014-10-01", verbose=False)
    tradable = xs.tradable_mask(P)
    feats = ml.cs_features(P)
    print(f"面板 {P['close'].shape}, 特征 {len(feats)} 个")

    rows, imps, alphas = [], {}, {}
    for m in args.models.split(","):
        t0 = time.time()
        max_rows = 60_000 if m == "mlp" else 200_000           # 神经网络训练慢, 少抽一点样本
        alphas[m], imps[m] = ml.ml_alpha(P, m, args.horizon, features=feats, max_rows=max_rows)
        print(f"  {m}: {time.time() - t0:.0f}s")

    start = min(a.first_valid_index() for a in alphas.values())
    for m, a in alphas.items():
        rows.append(evaluate(m, a, P, tradable, start, args.horizon, args.cost))
    rows.append(evaluate("手写 alpha (对照)", hand.compute_alpha(P), P, tradable, start, args.horizon, args.cost))
    tab = pd.DataFrame(rows)

    imp_tab = pd.DataFrame({m: s for m, s in imps.items() if not s.empty})
    imp_tab.index.name = "特征"
    imp_tab = imp_tab.reset_index()

    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"\n样本外区间: {start.date()} 起, 预测 {args.horizon} 天, 单边成本 {args.cost * 1e4:.0f} bp\n")
    print(tab.round(3).to_string(index=False))
    print("\n最后一次训练的特征重要性 (树: 增益占比; ridge: 标准化系数)")
    print(imp_tab.round(3).to_string(index=False))

    out = REPORTS_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    L = [f"# 机器学习截面 alpha: 沪深300 (预测 {args.horizon} 天)", "",
         f"- 滚动训练: 每次用过去 756 天, 每 63 天重训, 隔离期 {args.horizon} 天; 超参数事先固定",
         f"- 全部是样本外预测, 区间 {start.date()} 起; 多空组合, 次日开盘成交, 每 {args.horizon} 天调仓, "
         f"单边成本 {args.cost * 1e4:.0f} bp",
         "- 股票池为当前沪深300 成分股 (幸存者偏差, 见 xs_alpha 报告)", "",
         *md_table(tab, {"IC均值": "{:.4f}", "盈亏平衡成本bp": "{:.1f}", "日均换手%": "{:.1f}"}), "",
         "## 特征重要性 (最后一次训练)", "", *md_table(imp_tab, {c: "{:.3f}" for c in imp_tab.columns})]
    path = out / f"xs_ml_h{args.horizon}_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n报告: {path}")


if __name__ == "__main__":
    main()
