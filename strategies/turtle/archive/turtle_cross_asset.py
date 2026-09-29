#!/usr/bin/env python3
"""
海龟交易系统 — 跨资产组合 (股指 / 国债 / 商品 / 汇率 / 可选加密货币)
=====================================================================
参考 Moskowitz, Ooi & Pedersen (2012): 在多个资产类别上各自跑趋势策略, 再分散组合.

组合方式 (两层风险均衡):
  1. 品种内: 海龟按 1% 资金 / ATR 定仓位, 每个品种承担的风险大致相同
  2. 类别间: 默认先在每个资产类别内等权, 再把各类别等权 (--weighting class);
     否则商品有十几个品种, 会压倒只有几个品种的汇率和国债.
     也可以所有品种直接等权 (--weighting instrument).
  每天只用当时已经有数据的品种 (如 IM 2022 年、TL 2023 年才上市).

用法:
  python turtle_cross_asset.py                          # 默认: 股指 + 国债 + 商品 + 汇率, 系统2 (60/20)
  python turtle_cross_asset.py --with-crypto            # 再加 BTC、ETH
  python turtle_cross_asset.py --system 1               # 用系统1 (20/10)
  python turtle_cross_asset.py --classes 商品,汇率      # 只跑部分类别
  python turtle_cross_asset.py --garch                  # 加 GARCH 过滤 (滚动估计)
  python turtle_cross_asset.py --weighting instrument   # 所有品种直接等权

报告: quant/reports/turtle/turtle_cross_asset_report.md
图片: quant/figures/turtle/turtle_portfolio_cross_asset.png 等
"""

import argparse
import math
import os
import sys
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_here = Path(__file__).resolve().parent
_project_root = next(p for p in _here.parents if (p / "data" / "paths.py").exists())
for _p in (str(_here), str(_project_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data import market_data
from data.paths import figures, reports
from turtle_engine import MarketSpec, run_backtest
from turtle_multi_asset import FUTURE, spec_for_symbol
from turtle_timeframe_backtest import (SYSTEMS, compute_garch_filter_rolling,
                                       fetch_or_load, resample_ohlcv)
import turtle_analytics as ta


# ──────────────────────────────────────────────────
# 品种池
# ──────────────────────────────────────────────────

UNIVERSE = {
    "股指": [("future", s) for s in ["IF", "IH", "IC", "IM"]],
    "国债": [("future", s) for s in ["T", "TF", "TS", "TL"]],
    "商品": [("future", s) for s in ["RB", "I", "CU", "AL", "AU", "AG", "SC",
                                    "TA", "MA", "M", "Y", "CF", "SR"]],
    "汇率": [("fx", s) for s in ["USDCNH", "EURUSD", "USDJPY", "AUDUSD", "GBPUSD"]],
}
CRYPTO_CLASS = {"加密": [("crypto", "BTC"), ("crypto", "ETH")]}
CRYPTO_START = "2022-01-01"   # 与 turtle_timeframe_backtest 的缓存一致, 直接复用已下载的 15 分钟数据

# 各市场的交易规则
FUTURE_SPEC = replace(FUTURE, initial_capital=10_000_000)   # 股指/国债一手很贵, 本金给足, 避免整手取整扭曲仓位
FX_SPEC = MarketSpec(name="fx", timeframes=[("1d", None, 1)], days_per_year=260,
                     initial_capital=1_000_000, allow_short=True,
                     open_fee=0.00002, close_fee=0.00002, slippage=0.0001)
CRYPTO_DAILY_SPEC = MarketSpec(name="crypto", timeframes=[("1d", None, 1)], days_per_year=365.25,
                               initial_capital=1_000_000, allow_short=True,
                               open_fee=0.0005, close_fee=0.0005, slippage=0.0003)

# 压力时期 (日期为大致区间)
STRESS_PERIODS = [
    ("2015 股灾", "2015-06-15", "2015-09-15"),
    ("2016 熔断", "2016-01-01", "2016-01-29"),
    ("2018 贸易摩擦", "2018-03-22", "2018-12-31"),
    ("2020 疫情冲击", "2020-01-20", "2020-03-23"),
    ("2022 全球加息", "2022-01-01", "2022-10-31"),
    ("2024 年初小盘股下跌", "2024-01-02", "2024-02-05"),
]


# ──────────────────────────────────────────────────
# 数据与单品种回测
# ──────────────────────────────────────────────────

def load(market, symbol, args):
    if market == "future":
        return market_data.load_future_continuous(symbol, start_date=args.start,
                                                  adjust=args.futures_adjust, verbose=False)
    if market == "fx":
        return market_data.load_fx_daily(symbol, start_date=args.start)
    if market == "crypto":
        df = fetch_or_load(symbol=f"{symbol}/USDT:USDT", base_tf="15m", start_date=CRYPTO_START)
        return None if df is None or df.empty else resample_ohlcv(df, "1D")
    raise ValueError(market)


def spec_for(market, symbol, df):
    if market == "future":
        return spec_for_symbol(FUTURE_SPEC, symbol, df, {})
    return FX_SPEC if market == "fx" else CRYPTO_DAILY_SPEC


def backtest_one(market, symbol, args, entry_d, exit_d):
    df = load(market, symbol, args)
    if df is None or df.empty:
        return None, None
    spec = spec_for(market, symbol, df)
    vf = None
    if args.garch:
        vf = compute_garch_filter_rolling(df, lookback=20, min_train=int(spec.days_per_year),
                                          refit_every=21, verbose=False)
    return run_backtest(df, entry_d, exit_d, 1, spec, vol_filter=vf), df


# ──────────────────────────────────────────────────
# 组合
# ──────────────────────────────────────────────────

def span_fill(R: pd.DataFrame) -> pd.DataFrame:
    """每个品种在"开始交易 ~ 最后一天"之间, 没有数据的日子 (如对方市场休市) 记为 0 收益."""
    R = R.copy()
    for c in R.columns:
        s = R[c]
        a, b = s.first_valid_index(), s.last_valid_index()
        if a is not None:
            R.loc[a:b, c] = s.loc[a:b].fillna(0.0)
    return R


def combine(returns: dict, weighting: str):
    """returns: {类别: {品种: 日收益}} → (组合日收益, 各类别日收益 DataFrame, 全部品种 DataFrame)."""
    all_idx = sorted(set().union(*[set(r.index) for cls in returns.values() for r in cls.values()]))
    idx = pd.DatetimeIndex(all_idx)
    by_class, all_inst = {}, {}
    for cls, d in returns.items():
        if not d:
            continue
        R = span_fill(pd.DataFrame(d).reindex(idx))
        by_class[cls] = R.mean(axis=1, skipna=True)
        all_inst.update({f"{cls}:{s}": R[s] for s in R.columns})
    C = pd.DataFrame(by_class)
    I = pd.DataFrame(all_inst)
    port = (C if weighting == "class" else I).mean(axis=1, skipna=True).dropna()
    return port, C.loc[port.index], I.loc[port.index]


def vol_scale(r: pd.Series, target: float, dpy: float, com: int = 60, max_lev: float = 10.0):
    """
    事前波动率缩放 (参考 TSMOM 论文): 杠杆_t = 目标波动 / σ_{t-1}.
    σ 用指数加权 (质心 com 天) 估计, 只用 t-1 及以前的数据 (shift(1)), 没有前视偏差.
    只用收益不为 0 的日子估计 σ: 海龟空仓日收益为 0, 若计入, 空仓一段时间后 σ 会被低估,
    再次入场时杠杆会暴涨.
    返回 (缩放后的日收益, 杠杆序列).
    """
    active = r.where(r != 0)
    sigma = active.ewm(com=com, min_periods=com, ignore_na=True).std().ffill() * math.sqrt(dpy)
    lev = (target / sigma).clip(upper=max_lev).shift(1)
    out = (r * lev).dropna()
    return out, lev.reindex(out.index)


def obs_per_year(r: pd.Series) -> float:
    years = (r.index[-1] - r.index[0]).days / 365.25
    return len(r) / years if years > 0 else 252


def within_cross_corr(I: pd.DataFrame):
    """类内 / 跨类 品种两两相关性的平均值 (参考 TSMOM 论文 Table 4)."""
    corr = I.corr()
    cls = [c.split(":")[0] for c in corr.columns]
    within, cross = [], []
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            v = corr.iat[i, j]
            if not np.isnan(v):
                (within if cls[i] == cls[j] else cross).append(v)
    return (np.mean(within) if within else np.nan), (np.mean(cross) if cross else np.nan)


# ──────────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────────

def write_report(args, sys_name, results, port, C, I, bench, dpy, failed,
                 port_scaled=None, lev=None):
    out_dir = str(reports("turtle"))
    fig_dir = str(figures("turtle"))
    L = []
    L.append("# 海龟交易系统 — 跨资产组合回测报告\n")
    L.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    L.append(f"> 系统: {sys_name} | 起始: {args.start} | 组合方式: "
             + ("各类别内等权, 再类别间等权" if args.weighting == "class" else "所有品种等权")
             + (" | GARCH 过滤 (滚动估计)" if args.garch else "") + "  ")
    L.append(f"> 期货: 新浪单月合约自拼复权连续 ({args.futures_adjust}), 每品种本金 "
             f"{FUTURE_SPEC.initial_capital:,.0f}, 滑点 {FUTURE_SPEC.slippage*1e4:.0f}‱  ")
    L.append(f"> 汇率: 东方财富日线, 滑点 {FX_SPEC.slippage*1e4:.0f}‱, 不含利差 (carry)  ")
    L.append("> 权益逐根盯市; 止损跳空按开盘价; 信号用复权价, 盈亏按真实价格  ")
    L.append("> 波动率缩放: "
             + (f"单品种先缩放到 {args.instrument_vol:.0%} (论文做法); " if args.instrument_vol else
                "单品种用海龟 1%/ATR 仓位; ")
             + (f"组合缩放到 {args.target_vol:.0%} 目标波动 (事前 EWMA 估计, 杠杆上限 10 倍)"
                if port_scaled is not None else "组合不缩放") + "\n")
    L.append("---\n")

    # 1. 品种明细
    L.append("## 一、各品种表现\n")
    L.append("| 类别 | 品种 | 起始 | 交易 | 年化% | 年化波动% | 夏普 | PF | 最大回撤% | Calmar | 持仓占比% |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cls, d in results.items():
        for sym, r in d.items():
            eq = r["equity"].dropna()
            start = eq.index[0].date() if len(eq) else "—"
            L.append(f"| {cls} | {sym} | {start} | {r['n_trades']} | {r['annual_return_pct']:+.1f} "
                     f"| {r['ann_vol_pct']:.1f} | {r['sharpe']:.2f} | {r['profit_factor']:.2f} "
                     f"| {r['max_drawdown_pct']:.1f} | {r['calmar']:.2f} | {r['exposure_pct']:.0f} |")
    if failed:
        L.append(f"\n数据获取失败, 未参与: {', '.join(failed)}\n")
    L.append("")

    # 2. 类别表现 + 相关性
    L.append("---\n")
    L.append("## 二、各资产类别表现 (类别内品种等权)\n")
    L.append("| 类别 | 品种数 | 年化收益% | 年化波动% | 夏普 | 最大回撤% | Calmar |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for cls in C.columns:
        s = ta.stats_from_returns(C[cls].dropna(), dpy)
        if s:
            L.append(f"| {cls} | {len(results[cls])} | {s['cagr_pct']:+.1f} | {s['vol_pct']:.1f} "
                     f"| {s['sharpe']:.2f} | {s['max_dd_pct']:.1f} | {s['calmar']:.2f} |")
    s = ta.stats_from_returns(port, dpy)
    L.append(f"| **组合 (全样本)** | {I.shape[1]} | **{s['cagr_pct']:+.1f}** | **{s['vol_pct']:.1f}** "
             f"| **{s['sharpe']:.2f}** | **{s['max_dd_pct']:.1f}** | **{s['calmar']:.2f}** |")
    # 各类别开始时间不同 (如期货只有 2018 年之后的数据), 早期组合只由少数类别构成;
    # 单独统计"全部类别都有数据之后"的区间, 这才是真正的跨资产组合
    full_start = max(C[c].first_valid_index() for c in C.columns)
    s_full = ta.stats_from_returns(port.loc[full_start:], dpy)
    if s_full:
        L.append(f"| **组合 (全部类别到齐后, {full_start.date()} 起)** | {I.shape[1]} "
                 f"| **{s_full['cagr_pct']:+.1f}** | **{s_full['vol_pct']:.1f}** | **{s_full['sharpe']:.2f}** "
                 f"| **{s_full['max_dd_pct']:.1f}** | **{s_full['calmar']:.2f}** |")
    L.append("\n各类别开始时间: " + ", ".join(
        f"{c} {C[c].first_valid_index().date()}" for c in C.columns)
        + "。开始时间不同时, 早期的\"组合\"只由已有数据的类别构成。\n")
    L.append("\n**类别之间的相关性** (策略日收益):\n")
    corr = C.corr()
    L.append("| | " + " | ".join(corr.columns) + " |")
    L.append("|---|" + "---:|" * len(corr.columns))
    for r_name, row in corr.iterrows():
        L.append(f"| {r_name} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |")
    w, c = within_cross_corr(I)
    L.append(f"\n品种两两相关性平均: **类内 {w:.2f}**, **跨类 {c:.2f}**。"
             "跨类相关性越低, 跨资产分散化降低组合波动的效果越好 (对照 TSMOM 论文 Table 4)。\n")

    # 2.5 目标波动率
    if port_scaled is not None:
        s_sc = ta.stats_from_returns(port_scaled, dpy)
        L.append("---\n")
        L.append(f"## 组合目标波动率: {args.target_vol:.0%}\n")
        L.append("组合原本的波动率很低 (品种多且相关性低), 期货和永续合约只需保证金, "
                 "实际中会加杠杆把组合调到目标波动率。夏普基本不变, 收益和回撤同比例放大。\n")
        L.append("| 组合 | 年化收益% | 年化波动% | 夏普 | 最大回撤% | Calmar | 平均杠杆 | 最高杠杆 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        L.append(f"| 原始 | {s['cagr_pct']:+.1f} | {s['vol_pct']:.1f} | {s['sharpe']:.2f} "
                 f"| {s['max_dd_pct']:.1f} | {s['calmar']:.2f} | 1.0 | 1.0 |")
        L.append(f"| 缩放到 {args.target_vol:.0%} | {s_sc['cagr_pct']:+.1f} | {s_sc['vol_pct']:.1f} "
                 f"| {s_sc['sharpe']:.2f} | {s_sc['max_dd_pct']:.1f} | {s_sc['calmar']:.2f} "
                 f"| {lev.mean():.1f} | {lev.max():.1f} |")
        L.append(f"\n缩放从 {port_scaled.index[0].date()} 开始 (前 60 个交易日用于估计波动率)。"
                 "以下压力时期和组合分析均使用缩放后的组合。\n")
        port = port_scaled

    # 3. 压力时期
    L.append("---\n")
    L.append("## 三、压力时期表现\n")
    L.append("| 时期 | 区间 | 海龟组合 | 等权买入持有 | 股指类 (海龟) | 当时有数据的类别 |")
    L.append("|---|---|---:|---:|---:|---|")
    for name, a, b in STRESS_PERIODS:
        p = port.loc[a:b]
        if len(p) < 5:
            continue
        cum = lambda x: (1 + x.dropna()).prod() - 1
        eq_idx = C["股指"].loc[a:b].dropna() if "股指" in C else pd.Series(dtype=float)
        active = ", ".join(c for c in C.columns if C[c].loc[a:b].notna().any())
        L.append(f"| {name} | {a} ~ {b} | {cum(p)*100:+.1f}% | {cum(bench.loc[a:b])*100:+.1f}% | "
                 + (f"{cum(eq_idx)*100:+.1f}%" if len(eq_idx) else "— (尚无数据)") + f" | {active} |")
    L.append("\n日期为大致区间, 仅用于观察趋势策略在危机中的表现。\n")

    # 4. 组合与风险分析 (复用 turtle_analytics)
    L.append("---\n")
    L.append("## 四、组合表现与风险特征\n")
    pf = {"returns": port, "by_symbol": C.loc[port.index[0]:], "stats": ta.stats_from_returns(port, dpy),
          "avg_corr": c, "n_symbols": C.shape[1],
          "grey_note": f"grey: {C.shape[1]} asset classes",
          "unit_label": f"{C.shape[1]} 个资产类别、{I.shape[1]} 个品种",
          "corr_label": "跨类别品种两两相关性平均"}
    L += ta.analysis_section(pf, bench, dpy, out_dir, fig_dir, "cross_asset", f"跨资产 {sys_name}")

    path = os.path.join(out_dir, "turtle_cross_asset_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n报告已保存: {path}")
    return path


# ──────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="海龟系统 — 跨资产组合")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--system", type=int, choices=[1, 2], default=2, help="1=20/10, 2=60/20 (默认)")
    p.add_argument("--classes", default="", help="只跑部分类别, 如 商品,汇率 (默认全部)")
    p.add_argument("--with-crypto", action="store_true", help="加入 BTC、ETH (日线)")
    p.add_argument("--weighting", choices=["class", "instrument"], default="class")
    p.add_argument("--futures-adjust", choices=["diff", "ratio"], default="diff")
    p.add_argument("--garch", action="store_true", help="GARCH 过滤 (滚动估计)")
    p.add_argument("--target-vol", type=float, default=0.10,
                   help="组合目标年化波动率 (默认 0.10 = 10%%; 0 = 不缩放)")
    p.add_argument("--instrument-vol", type=float, default=0.0,
                   help="论文做法: 每个品种先缩放到该波动率再组合, 如 0.40 (默认不启用)")
    args = p.parse_args()

    sys_name, entry_d, exit_d = SYSTEMS[args.system - 1]
    universe = dict(UNIVERSE)
    if args.with_crypto:
        universe.update(CRYPTO_CLASS)
    if args.classes:
        keep = [c.strip() for c in args.classes.split(",")]
        if args.with_crypto:
            keep.append("加密")             # 指定了 --with-crypto 就一定包含加密类
        universe = {k: v for k, v in universe.items() if k in keep}

    n = sum(len(v) for v in universe.values())
    print(f"\n{'='*80}\n海龟交易系统 — 跨资产组合 | {sys_name} | {len(universe)} 个类别, {n} 个品种"
          f"\n{'='*80}")

    results, returns, prices, failed = {}, {}, {}, []
    for cls, items in universe.items():
        print(f"\n[{cls}]")
        results[cls], returns[cls], prices[cls] = {}, {}, {}
        for market, sym in items:
            try:
                r, df = backtest_one(market, sym, args, entry_d, exit_d)
            except Exception as e:
                r, df = None, None
                print(f"  [!] {sym} 出错: {e}")
            if r is None or r["n_trades"] == 0:
                failed.append(sym)
                if r is not None:
                    print(f"  {sym:<7} 没有成交 (数据太短或资金不足), 跳过")
                continue
            results[cls][sym] = r
            returns[cls][sym] = r["daily_returns"]
            prices[cls][sym] = df            # 含 raw_close/scale, 基准按真实价格算收益
            print(f"  {sym:<7} 交易:{r['n_trades']:>4}  年化:{r['annual_return_pct']:>+6.1f}%"
                  f"  夏普:{r['sharpe']:>5.2f}  回撤:{r['max_drawdown_pct']:>6.1f}%")
    results = {k: v for k, v in results.items() if v}
    returns = {k: v for k, v in returns.items() if v}
    if not returns:
        print("\n[!] 没有成功回测的品种")
        sys.exit(1)

    if args.instrument_vol > 0:          # 论文做法: 单品种先缩放到相同的波动率
        returns = {cls: {s: vol_scale(r, args.instrument_vol, obs_per_year(r))[0] for s, r in d.items()}
                   for cls, d in returns.items()}
    port, C, I = combine(returns, args.weighting)

    # 基准: 同样的品种和权重, 买入持有 (汇率为持有货币对本身)
    bench_ret = {}
    for cls, px in prices.items():
        if px:
            bench_ret[cls] = {s: ta.price_returns(v) for s, v in px.items()}
    bench, _, _ = combine(bench_ret, args.weighting)
    bench = bench.reindex(port.index).fillna(0.0)

    dpy = obs_per_year(port)
    s = ta.stats_from_returns(port, dpy)
    print(f"\n组合: 年化 {s['cagr_pct']:+.1f}%  波动 {s['vol_pct']:.1f}%  夏普 {s['sharpe']:.2f}"
          f"  最大回撤 {s['max_dd_pct']:.1f}%")
    port_scaled = lev = None
    if args.target_vol > 0:
        port_scaled, lev = vol_scale(port, args.target_vol, dpy)
        s2 = ta.stats_from_returns(port_scaled, dpy)
        print(f"缩放到 {args.target_vol:.0%} 波动: 年化 {s2['cagr_pct']:+.1f}%  波动 {s2['vol_pct']:.1f}%"
              f"  夏普 {s2['sharpe']:.2f}  最大回撤 {s2['max_dd_pct']:.1f}%  平均杠杆 {lev.mean():.1f}")
    write_report(args, sys_name, results, port, C, I, bench, dpy, failed, port_scaled, lev)


if __name__ == "__main__":
    main()
