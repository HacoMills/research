"""
第 5 层: 稳定性检验
===================
回答"这个结果是不是碰巧的":

  1. 信号对比     同一批品种换不同信号 (唐奇安 / TSMOM / 均线), 以及它们之间的相关性
  2. 过滤器对比   不过滤 / GARCH / VIX 高 / VIX 低
  3. 参数网格     信号参数扫一遍, 看夏普是一片高原 (稳) 还是一根尖刺 (过拟合)
  4. 样本内外     样本内挑最好的参数, 看它在样本外还剩多少
  5. 分年度       每年的收益和夏普, 赚钱年份占比
  6. 自助法       按 20 天分块重抽日收益 1000 次, 得到夏普的 95% 置信区间
  7. 成本敏感性   滑点 ×0 / ×1 / ×2 / ×4
  8. 信号 × 过滤器组合普查   每个信号 (可展开多组时间窗口) 配每个过滤器跑一遍,
                            按样本内挑组合、看样本外是否还成立, 并分时期看夏普

所有检验只返回表格数据 (DataFrame / dict), 由报告层 (report.py) 负责排版和画图.
"""

import itertools
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .filters import parse_filter
from .signals import parse_signal


def sharpe(r: pd.Series, dpy: float) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * math.sqrt(dpy)) if len(r) > 2 and r.std() > 0 else np.nan


def summary(r: pd.Series, dpy: float) -> Dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / dpy
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return {"年化收益%": cagr * 100, "年化波动%": r.std() * math.sqrt(dpy) * 100,
            "夏普": sharpe(r, dpy), "最大回撤%": dd * 100, "Calmar": cagr / abs(dd) if dd < 0 else 0}


def split_date(r: pd.Series, oos_start: Optional[str]) -> pd.Timestamp:
    """样本外起点: 配置里给了就用, 否则取样本 70% 处."""
    if oos_start:
        return pd.Timestamp(oos_start)
    return r.index[int(len(r) * 0.7)]


def _is_oos(r, split, dpy):
    return sharpe(r[r.index < split], dpy), sharpe(r[r.index >= split], dpy)


# ──────────────────────────────────────────────────

def compare(runner, base_pf, items: List[str], kind: str, base_signal, base_filter, split) -> Dict:
    """kind='signal': 换信号, 过滤器不变; kind='filter': 换过滤器, 信号不变."""
    rows, series = [], {}
    for spec in items:
        sig = parse_signal(spec) if kind == "signal" else base_signal
        flt = parse_filter(spec) if kind == "filter" else base_filter
        label = sig.label() if kind == "signal" else flt.label()
        print(f"  {kind}: {label}")
        pf = base_pf if (sig.label(), flt.label()) == (base_pf.signal, base_pf.filter) \
            else runner.run(sig, flt, verbose=False)
        r = pf.main
        if len(r) < 2:
            continue
        is_s, oos_s = _is_oos(r, split, pf.dpy)
        row = {kind: label, **summary(r, pf.dpy), "样本内夏普": is_s, "样本外夏普": oos_s,
               "交易次数": sum(x["n_trades"] for x in pf.results.values())}
        if pf.by_class.shape[1] > 1:                     # 分资产类别的夏普 (未缩放的类别收益)
            for cls in pf.by_class.columns:
                row[f"夏普·{cls}"] = sharpe(pf.by_class[cls].dropna(), pf.dpy)
        if kind == "filter":
            row["被挡掉的入场"] = sum(x.get("blocked", 0) for x in pf.results.values())
        rows.append(row)
        series[label] = r
    table = pd.DataFrame(rows)
    corr = pd.DataFrame(series).corr() if len(series) > 1 else None
    return {"table": table, "corr": corr, "returns": series}


def param_grid(runner, base_signal, base_filter, grid: List[List[float]], split) -> Optional[Dict]:
    """grid: 每个参数的取值列表, 如 donchian 的 [[20,40,60,100],[10,20,30]]."""
    if not grid:
        return None
    rows = []
    for params in itertools.product(*grid):
        name = base_signal.name
        if name == "donchian" and len(params) == 2 and params[1] > params[0]:
            continue                                     # 出场通道比入场长, 没意义
        if name == "ma" and len(params) == 2 and params[0] >= params[1]:
            continue                                     # 快线要短于慢线
        sig = base_signal.with_params(*params)
        pf = runner.run(sig, base_filter, verbose=False)
        r = pf.main
        if len(r) < 2:
            continue
        is_s, oos_s = _is_oos(r, split, pf.dpy)
        rows.append({"params": tuple(params), "label": sig.label(), "夏普": sharpe(r, pf.dpy),
                     "样本内夏普": is_s, "样本外夏普": oos_s, **{f"p{i+1}": p for i, p in enumerate(params)}})
        print(f"  {sig.label():<24} 夏普 {rows[-1]['夏普']:.2f}  (样本内 {is_s:.2f} / 样本外 {oos_s:.2f})")
    if not rows:
        return None
    t = pd.DataFrame(rows)
    best_is = t.loc[t["样本内夏普"].idxmax()]
    base_row = t[t["label"] == base_signal.label()]
    return {
        "table": t, "n_params": len(grid),
        "best_is": best_is,
        "base": base_row.iloc[0] if len(base_row) else None,
        "oos_median": float(t["样本外夏普"].median()),
        "share_positive": float((t["夏普"] > 0).mean()),
    }


def yearly(r: pd.Series, dpy: float) -> pd.DataFrame:
    rows = []
    for y, g in r.groupby(r.index.year):
        if len(g) < 20:
            continue
        rows.append({"年份": y, "收益%": ((1 + g).prod() - 1) * 100, "夏普": sharpe(g, dpy),
                     "最大回撤%": (((1 + g).cumprod() / (1 + g).cumprod().cummax()) - 1).min() * 100})
    return pd.DataFrame(rows)


def block_bootstrap(r: pd.Series, dpy: float, n: int = 1000, block: int = 20, seed: int = 0) -> Dict:
    """按连续 block 天分块有放回重抽 (保留短期自相关和波动聚集), 返回夏普分布."""
    x = r.dropna().to_numpy()
    T = len(x)
    if T < block * 5:
        return {}
    rng = np.random.default_rng(seed)
    k = int(math.ceil(T / block))
    out = np.empty(n)
    for b in range(n):
        starts = rng.integers(0, T - block + 1, size=k)
        s = np.concatenate([x[i:i + block] for i in starts])[:T]
        out[b] = s.mean() / s.std() * math.sqrt(dpy) if s.std() > 0 else 0
    return {"sharpe": sharpe(r, dpy), "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)), "p_le_0": float((out <= 0).mean()),
            "n": n, "block": block, "draws": out}


def cost_sensitivity(runner, base_signal, base_filter, mults=(0, 1, 2, 4)) -> pd.DataFrame:
    rows = []
    for m in mults:
        print(f"  滑点 ×{m:g}")
        pf = runner.run(base_signal, base_filter, slip_mult=m, verbose=False)
        rows.append({"滑点倍数": f"×{m:g}", **summary(pf.main, pf.dpy)})
    return pd.DataFrame(rows)


def expand_windows(signals: List[str], windows: Dict[str, List[str]]) -> List[str]:
    """
    [robustness.windows] 里每个信号列出几组参数 (时间窗口), 展开成具体信号:
      windows = {"donchian": ["20,10", "60,20"], "tsmom": [63, 252]}
      → ["donchian:20,10", "donchian:60,20", "tsmom:63", "tsmom:252"]
    再接上 signals 里手写的 (如混合信号), 去重并保持顺序.
    """
    out = [f"{name}:{p}" for name, params in (windows or {}).items() for p in params]
    out += list(signals or [])
    seen, uniq = set(), []
    for s in out:
        key = s.replace(" ", "")
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


def _periods(r: pd.Series, bounds: Optional[List[str]]) -> List[tuple]:
    """分段: bounds 是分界日期, 如 ["2019-01-01", "2022-01-01"] → 三段. 没给就按年."""
    start, end = r.index[0], r.index[-1]
    if bounds:
        inner = [pd.Timestamp(b) for b in bounds if start < pd.Timestamp(b) <= end]
    else:
        inner = [pd.Timestamp(f"{y}-01-01") for y in sorted(set(r.index.year))[1:]]
    cuts = [start] + inner + [end + pd.Timedelta(days=1)]
    out = []
    for k, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
        last = (b - pd.Timedelta(days=1)).year
        if bounds and k == 0:
            label = f"至{last}"          # 第一段起点因信号预热长短而不同, 标签只写终点, 保证各组合列名一致
        else:
            label = f"{a.year}" if a.year == last else f"{a.year}-{last % 100:02d}"
        out.append((label, a, b))
    return out


def combo_scan(runner, signals: List[str], filters: List[str], split,
               periods: Optional[List[str]] = None) -> Optional[Dict]:
    """
    信号 × 过滤器全组合. 试的组合越多, 最好的那个越可能只是运气 (多重检验),
    所以按样本内夏普排序挑组合, 再看它们的样本外夏普.
    另外每个组合分时期算夏普 (periods 分界; 不给就按年), 看它是不是只在某一段赚钱.
    """
    rows = []
    total = len(signals) * len(filters)
    for i, sig_spec in enumerate(signals):
        for j, flt_spec in enumerate(filters):
            sig, flt = parse_signal(sig_spec), parse_filter(flt_spec)
            print(f"  [{i * len(filters) + j + 1}/{total}] {sig.label()} | {flt.label()}")
            pf = runner.run(sig, flt, verbose=False)
            r = pf.main
            if len(r) < 2:
                continue
            is_s, oos_s = _is_oos(r, split, pf.dpy)
            row = {"信号": sig.label(), "过滤器": flt.label(), "夏普": sharpe(r, pf.dpy),
                   "样本内夏普": is_s, "样本外夏普": oos_s,
                   "最大回撤%": summary(r, pf.dpy).get("最大回撤%", np.nan),
                   "交易次数": sum(x["n_trades"] for x in pf.results.values())}
            per = {lab: sharpe(r[(r.index >= a) & (r.index < b)], pf.dpy) for lab, a, b in _periods(r, periods)}
            vals = [v for v in per.values() if pd.notna(v)]
            row["最差时期夏普"] = min(vals) if vals else np.nan
            row["盈利时期占比%"] = 100 * float(np.mean([v > 0 for v in vals])) if vals else np.nan
            row.update({f"夏普 {k}": v for k, v in per.items()})
            rows.append(row)
    if not rows:
        return None
    t = pd.DataFrame(rows)
    top = t.sort_values("样本内夏普", ascending=False).head(5)
    return {"table": t, "n": len(t), "top_is": top,
            "period_cols": [c for c in t.columns if c.startswith("夏普 ")],
            "oos_median": float(t["样本外夏普"].median()),
            "oos_of_top": float(top["样本外夏普"].mean()),
            "is_of_top": float(top["样本内夏普"].mean())}


def run_all(runner, base_pf, base_signal, base_filter, rcfg: dict) -> Dict:
    """按配置 [robustness] 跑全部检验."""
    r = base_pf.main
    split = split_date(r, rcfg.get("oos_start"))
    out = {"split": split}
    if rcfg.get("signals"):
        print("\n[稳定性] 信号对比")
        out["signals"] = compare(runner, base_pf, rcfg["signals"], "signal", base_signal, base_filter, split)
    if rcfg.get("filters"):
        print("\n[稳定性] 过滤器对比")
        out["filters"] = compare(runner, base_pf, rcfg["filters"], "filter", base_signal, base_filter, split)
    grid = rcfg.get("grid", {}).get(base_signal.name)
    if grid:
        print(f"\n[稳定性] 参数网格 ({base_signal.name})")
        out["grid"] = param_grid(runner, base_signal, base_filter, grid, split)
    combo_signals = expand_windows(rcfg.get("signals"), rcfg.get("windows"))
    if rcfg.get("combos") and combo_signals and rcfg.get("filters"):
        print(f"\n[稳定性] 信号 × 过滤器组合普查 ({len(combo_signals)} 个信号 × {len(rcfg['filters'])} 个过滤器)")
        out["combos"] = combo_scan(runner, combo_signals, rcfg["filters"], split, rcfg.get("periods"))
    out["yearly"] = yearly(r, base_pf.dpy)
    n_boot = int(rcfg.get("bootstrap", 1000))
    if n_boot > 0:
        out["bootstrap"] = block_bootstrap(r, base_pf.dpy, n=n_boot, block=int(rcfg.get("block_days", 20)))
    if rcfg.get("cost_multipliers"):
        print("\n[稳定性] 成本敏感性")
        out["costs"] = cost_sensitivity(runner, base_signal, base_filter, rcfg["cost_multipliers"])
    return out
