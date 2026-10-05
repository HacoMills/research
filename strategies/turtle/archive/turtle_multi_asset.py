#!/usr/bin/env python3
"""
海龟交易系统 — 多市场版 (加密货币 / A股 / 国内期货)
=====================================================
在 turtle_timeframe_backtest.py 的基础上增加 A股和期货支持.
回测逻辑与原版保持一致 (收盘突破入场、2N 止损、唐奇安通道出场、
1% 风险/ATR 仓位、可选 GARCH 过滤), 只按各市场的交易规则做必要调整:

                    加密货币        A股                 国内期货
  K线周期           15min~1d        日线                日线
  能否做空          可以            不可以 (只做多)      可以
  最小交易单位      不限 (可小数)    100 股 (1 手)        1 张合约
  合约乘数          1               1                   按品种 (螺纹 10 吨/手 等)
  手续费            0.05% 双边       万 2.5 双边 +        万 1 双边
                                    卖出印花税 0.05%
  资金约束          无              持仓市值 ≤ 账户权益   无 (未建模保证金)
  涨跌停            无              收盘涨停不能买入,     未建模
                                    收盘跌停不能卖出
  年化天数          365.25          252                 252

  A股 T+1: 日线上"今天收盘买入 → 最早明天卖出", 原回测循环天然满足.

前视偏差处理:
  1. GARCH 默认滚动重估, 只用过去的数据拟合参数 (原版全样本拟合可用 --garch-mode insample)
  2. 期货连续合约: 换月只看当天收盘已知的持仓量, 次日生效
  3. 复权: 信号用复权价, 手数/手续费/盈亏换算回当时真实价格,
     以后的换月、分红不会改变过去任何一笔交易

用法:
  # 加密货币 (与原版完全一致, 数据走原来的 OKX 缓存)
  python turtle_multi_asset.py --market crypto --symbols BTC,ETH

  # A股 (默认读 quant/data/securities 下的 CSMAR 日度文件)
  python turtle_multi_asset.py --market stock --symbols 600519,000001,300750
  python turtle_multi_asset.py --market stock --symbols 600519 --source akshare   # 改用 AKShare

  # 国内期货 (默认: 新浪单月合约按持仓量换月, 自拼差值复权连续)
  python turtle_multi_asset.py --market future --symbols RB,CU,AU,M,IF
  python turtle_multi_asset.py --market future --symbols RB --futures-adjust ratio
  python turtle_multi_asset.py --market future --symbols RB --futures-source main   # 新浪未复权主力

  # GARCH 默认滚动重估 (无前视偏差); 想复现原版全样本拟合:
  python turtle_multi_asset.py --market crypto --symbols BTC --garch --garch-mode insample

  # 用自己从 Wind/CSMAR 导出的 CSV (文件名 = 代码, 如 600519.csv / RB.csv)
  python turtle_multi_asset.py --market future --source csv --csv-dir ./my_data --symbols RB,CU

  # GARCH 对比
  python turtle_multi_asset.py --market future --symbols RB,CU,AU --garch-compare
"""

import sys
import os
import math
import argparse
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 项目路径: turtle/ 用于导入原版回测模块, 根目录用于导入 data 包 ──
_here = Path(__file__).resolve().parent
_project_root = next(p for p in _here.parents if (p / "data" / "paths.py").exists())   # quant 根目录
for _p in (str(_here), str(_project_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 复用原版的指标、GARCH 过滤、数据加载, 保证逻辑完全一致
from turtle_timeframe_backtest import (
    calc_atr, calc_donchian, compute_garch_filter, resample_ohlcv,
    fetch_or_load, TIMEFRAMES as CRYPTO_TIMEFRAMES, SYSTEMS,
    INITIAL_CAPITAL as CRYPTO_CAPITAL, RISK_PCT, STOP_MULT,
    ATR_PERIOD_DAYS, COMMISSION_PCT, REPORTS_DIR,
    compute_garch_filter_rolling,        # 滚动版 GARCH 统一放在原版文件里, 两边共用
)
from data import market_data
from data.paths import figures as figures_dir
from turtle_engine import MarketSpec, run_backtest
import turtle_analytics as ta


# ──────────────────────────────────────────────────
# 市场规则
# ──────────────────────────────────────────────────

# MarketSpec 定义在 turtle_engine.py (各脚本共用)

CRYPTO = MarketSpec(
    name="crypto", timeframes=CRYPTO_TIMEFRAMES, days_per_year=365.25,
    initial_capital=CRYPTO_CAPITAL, allow_short=True,
    open_fee=COMMISSION_PCT, close_fee=COMMISSION_PCT,
    slippage=0.0003,                 # 3 个基点
)

STOCK = MarketSpec(
    name="stock", timeframes=[("1d", None, 1)], days_per_year=252,
    initial_capital=1_000_000, allow_short=False,
    open_fee=0.00025, close_fee=0.00025, sell_tax=0.0005,
    lot_size=100, multiplier=1, cash_only=True,
    slippage=0.001,                  # 10 个基点 (个股流动性差异大)
)

FUTURE = MarketSpec(
    name="future", timeframes=[("1d", None, 1)], days_per_year=252,
    initial_capital=2_000_000, allow_short=True,
    open_fee=0.0001, close_fee=0.0001,
    lot_size=1, multiplier=1,        # 乘数按品种在运行时填入
    slippage=0.0003,                 # 3 个基点, 约 1 个最小变动价位
)

# 常见期货品种合约乘数 (每手多少单位). 交易所可能调整, 使用前请核对最新合约规格.
FUTURE_MULTIPLIERS = {
    # 上期所 / 上期能源
    "CU": 5, "AL": 5, "ZN": 5, "PB": 5, "NI": 1, "SN": 1, "AU": 1000, "AG": 15,
    "RB": 10, "HC": 10, "SS": 5, "BU": 10, "RU": 10, "FU": 10, "SP": 10,
    "SC": 1000, "LU": 10, "NR": 10, "AO": 20, "BR": 5, "EC": 50,
    # 大商所
    "I": 100, "J": 100, "JM": 60, "M": 10, "Y": 10, "P": 10, "A": 10, "B": 10,
    "C": 10, "CS": 10, "L": 5, "V": 5, "PP": 5, "EG": 10, "EB": 5, "PG": 20,
    "JD": 10, "LH": 16,
    # 郑商所
    "TA": 5, "MA": 10, "SR": 10, "CF": 5, "OI": 10, "RM": 10, "FG": 20, "SA": 20,
    "AP": 10, "CJ": 5, "UR": 20, "SF": 5, "SM": 5, "PK": 5, "PF": 5, "PX": 5,
    "SH": 30,
    # 广期所
    "SI": 5, "LC": 1,
    # 中金所
    "IF": 300, "IH": 300, "IC": 200, "IM": 200, "T": 10000, "TF": 10000,
    "TS": 20000, "TL": 10000,
}

DEFAULT_SYMBOLS = {
    "crypto": ["BTC", "ETH"],
    "stock": ["600519", "000001", "601318", "000858", "300750"],
    "future": ["RB", "I", "CU", "AL", "AU", "AG", "SC", "TA", "MA",
               "M", "Y", "CF", "SR", "IF", "T"],
}


def stock_limit_pct(code: str, index: pd.DatetimeIndex) -> pd.Series:
    """
    按板块给出每天的涨跌停幅度 (%).
    主板 10%; 科创板 (688/689) 20%; 创业板 (300/301) 2020-08-24 起 20%, 之前 10%;
    北交所 (8/4/92 开头) 30%. ST 股 (5%) 无法从代码判断, 未处理.
    """
    s = pd.Series(10.0, index=index)
    if code.startswith(("688", "689")):
        s[:] = 20.0
    elif code.startswith(("300", "301")):
        s[index >= pd.Timestamp("2020-08-24")] = 20.0
    elif code.startswith(("8", "4", "92")):
        s[:] = 30.0
    return s


# ──────────────────────────────────────────────────
# GARCH 过滤: 滚动估计版 (消除前视偏差)
# ──────────────────────────────────────────────────
#
# 原版 compute_garch_filter 用全样本一次性拟合 GARCH(1,1), 2022 年的过滤信号
# 用到了 2022~2026 年全部数据估出的 α、β, 属于前视偏差.
#
# 滚动版做法 (扩展窗口, 定期重估):
#   - 前 min_train 根K线只用来训练, 这段时间不过滤 (与基线相同, 便于公平对比)
#   - 之后每隔 refit_every 根K线, 只用"此前"的收益率重新拟合一次参数
#   - 两次重估之间, 用最近一次的参数按 GARCH 递推公式逐根计算条件方差:
#         σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
#     σ_t 只依赖 t-1 及以前的收益, 在第 t 根K线收盘决策时已知.
#   - 过滤规则与原版相同: σ_t > 过去 lookback 根 σ 的均值 → 波动率扩张 → 允许入场

# compute_garch_filter_rolling 定义在 turtle_timeframe_backtest.py, 上面已导入


# 回测引擎: turtle_engine.run_backtest (逐根盯市、夏普、止损跳空、滑点)


# ──────────────────────────────────────────────────
# 数据加载 + 单品种回测
# ──────────────────────────────────────────────────

def load_symbol(market: str, symbol: str, args) -> Optional[pd.DataFrame]:
    if market == "crypto":
        pair = symbol if "/" in symbol else f"{symbol.upper()}/USDT:USDT"
        return fetch_or_load(symbol=pair, base_tf="15m", start_date=args.start)
    if args.source == "csmar":
        df = market_data.load_csmar_daily(symbol, start_date=args.start)
        if df.attrs.get("close_only"):
            print("    (CSMAR 文件只有收盘价: 通道和 ATR 按收盘价计算)")
        return df
    if args.source == "csv":
        return market_data.load_csv(market_data.find_csv(args.csv_dir, symbol),
                                    start_date=args.start)
    if market == "stock":
        return market_data.load_stock_daily(symbol, start_date=args.start,
                                            adjust=args.adjust, refresh=args.refresh)
    if args.futures_source == "main":
        return market_data.load_future_daily(symbol, start_date=args.start,
                                             refresh=args.refresh)
    return market_data.load_future_continuous(symbol, start_date=args.start,
                                              adjust=args.futures_adjust,
                                              refresh=args.refresh)


def spec_for_symbol(base: MarketSpec, symbol: str, df: pd.DataFrame,
                    mult_override: Dict[str, float]) -> MarketSpec:
    if base.name == "stock":
        return replace(base, limit_pct=stock_limit_pct(symbol, df.index))
    if base.name == "future":
        prod = symbol.upper().rstrip("0")
        mult = mult_override.get(prod, FUTURE_MULTIPLIERS.get(prod))
        if mult is None:
            print(f"    [!] 未知品种 {prod} 的合约乘数, 按 1 处理; "
                  f"可用 --multiplier {prod}=10 指定")
            mult = 1
        return replace(base, multiplier=mult)
    return base


def run_single_symbol(market: str, symbol: str, base_spec: MarketSpec, args,
                      use_garch: bool = False, df_raw: pd.DataFrame = None):
    if df_raw is None:
        df_raw = load_symbol(market, symbol, args)
    if df_raw is None or df_raw.empty:
        return None, None

    spec = spec_for_symbol(base_spec, symbol, df_raw, args.mult_override)
    results = {}

    for tf_name, tf_rule, bars_day in spec.timeframes:
        df_tf = df_raw.copy() if tf_rule is None else resample_ohlcv(df_raw, tf_rule)
        print(f"  ▸ {tf_name}  ({len(df_tf)} 根K线"
              + (f", 合约乘数 {spec.multiplier:g}" if market == "future" else "") + ")")

        garch_filter = None
        if use_garch:
            garch_lookback = max(ATR_PERIOD_DAYS * bars_day, 20)
            if args.garch_mode == "insample":        # 原版: 全样本拟合 (有前视偏差)
                garch_filter = compute_garch_filter(df_tf, lookback=garch_lookback)
            else:                                    # 默认: 滚动重估
                garch_filter = compute_garch_filter_rolling(
                    df_tf, lookback=garch_lookback,
                    min_train=int(spec.days_per_year * bars_day),          # 训练 1 年
                    refit_every=max(int(args.garch_refit_days * bars_day), 1))

        for sys_name, entry_d, exit_d in SYSTEMS:
            need = max(entry_d, exit_d, ATR_PERIOD_DAYS) * bars_day + 10
            if len(df_tf) < need:
                print(f"    {sys_name}: 数据不足, 跳过")
                continue
            r = run_backtest(df_tf, entry_d, exit_d, bars_day, spec,
                             vol_filter=garch_filter)
            results[(tf_name, sys_name)] = r
            extra = ""
            if r.get("skip_size"):
                extra += f"  资金不足跳过:{r['skip_size']}"
            if r.get("skip_limit"):
                extra += f"  涨跌停受阻:{r['skip_limit']}"
            print(f"    {sys_name}  交易:{r['n_trades']:>4}"
                  f"  年化:{r['annual_return_pct']:>+6.1f}%"
                  f"  夏普:{r['sharpe']:>5.2f}"
                  f"  PF:{r['profit_factor']:>5.2f}"
                  f"  回撤:{r['max_drawdown_pct']:>7.1f}%"
                  f"  Calmar:{r['calmar']:>6.3f}{extra}")
    return results, df_raw


# ──────────────────────────────────────────────────
# 报告
# ──────────────────────────────────────────────────

_ROW_HEAD = ("| 品种 | 周期 | 系统 | 交易 | 总收益% | 年化% | 年化波动% | 夏普 | 胜率% | PF "
             "| 回撤% | Calmar | 持仓占比% |")
_ROW_SEP = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"


def _row(sym, tf, sn, r):
    return (f"| {sym} | {tf} | {sn} | {r['n_trades']} | {r['total_return_pct']:+.1f} "
            f"| {r['annual_return_pct']:+.1f} | {r['ann_vol_pct']:.1f} | {r['sharpe']:.2f} "
            f"| {r['win_rate']:.1f} | {r['profit_factor']:.2f} | {r['max_drawdown_pct']:.1f} "
            f"| {r['calmar']:.3f} | {r['exposure_pct']:.0f} |")


def _summary_block(L, grand, spec, title):
    L(f"## {title}\n")
    for tf_name, _, _ in spec.timeframes:
        for sys_name, _, _ in SYSTEMS:
            # 没有成交的品种 (如资金不足买不起 1 手) 不计入横向统计
            rows = [(s, grand[s][(tf_name, sys_name)]) for s in sorted(grand)
                    if (tf_name, sys_name) in grand[s]
                    and grand[s][(tf_name, sys_name)]["n_trades"] > 0]
            if not rows:
                continue
            rows.sort(key=lambda x: x[1]["annual_return_pct"], reverse=True)
            L(f"### {tf_name} × {sys_name} ({len(rows)} 个品种)\n")
            L(_ROW_HEAD)
            L(_ROW_SEP)
            for s, r in rows:
                L(_row(s, tf_name, sys_name, r))
            ann = [r["annual_return_pct"] for _, r in rows]
            sh = [r["sharpe"] for _, r in rows]
            pf = [r["profit_factor"] for _, r in rows]
            dd = [r["max_drawdown_pct"] for _, r in rows]
            cal = [r["calmar"] for _, r in rows]
            n_pos = sum(a > 0 for a in ann)
            L(f"| **中位数** | | | | | {np.median(ann):+.1f} | | {np.median(sh):.2f} | "
              f"| {np.median(pf):.2f} | {np.median(dd):.1f} | {np.median(cal):.3f} | |")
            L(f"| **均值** | | | | | {np.mean(ann):+.1f} | | {np.mean(sh):.2f} | "
              f"| {np.mean(pf):.2f} | {np.mean(dd):.1f} | {np.mean(cal):.3f} | |")
            L(f"\n盈利品种: {n_pos}/{len(rows)} ({n_pos/len(rows)*100:.0f}%)\n")


def write_report(market, spec, grand, grand_garch, failed, args, out_dir, prices=None):
    lines = []
    L = lines.append
    names = {"crypto": "加密货币", "stock": "A股", "future": "国内期货"}
    L(f"# 海龟交易系统 — {names[market]}回测报告\n")
    L(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    L(f"> 回测区间: {args.start} ~ 今 | 数据来源: "
      f"{'OKX' if market == 'crypto' else args.source}  ")
    L(f"> 初始资金: {spec.initial_capital:,.0f} | 仓位: 1% 资金/ATR | 止损: 2N  ")
    L(f"> 手续费: 开 {spec.open_fee*1e4:.1f}‱ / 平 {spec.close_fee*1e4:.1f}‱"
      + (f" / 卖出印花税 {spec.sell_tax*1e4:.1f}‱" if spec.sell_tax else "")
      + f" | 滑点: {spec.slippage*1e4:.1f}‱/次"
      + f" | 做空: {'允许' if spec.allow_short else '不允许'}"
      + f" | 年化天数: {spec.days_per_year:g}  ")
    L("> 权益逐根盯市 (回撤、夏普含持仓浮亏); 止损跳空时按开盘价成交\n")
    if market == "stock" and args.source == "csmar":
        L("> A股数据: CSMAR 日个股回报率 (TRD_Dalyr); 信号用 Dretwd 累乘的复权价, "
          "手数/费用用不复权收盘价 Clsprc  ")
        L("> ⚠️ 文件只有收盘价 (无开高低) 时, 唐奇安通道与 ATR 均按收盘价计算, "
          "止损也只在收盘时检查。\n")
    if market == "future" and args.source == "akshare":
        if args.futures_source == "main":
            L("> ⚠️ 新浪主力连续为直接拼接合约, 换月跳空可能误触发突破, 结果仅供参考。\n")
        else:
            L(f"> 期货数据: 新浪单月合约按持仓量换月自拼 | 复权: {args.futures_adjust}  ")
            L("> 未计入换月时平旧开新的额外手续费与滑点。\n")
    if grand_garch or args.garch:
        L(f"> GARCH 模式: {'滚动重估 (无前视偏差), 每 %g 天重估, 首年仅训练不过滤' % args.garch_refit_days if args.garch_mode == 'rolling' else '全样本拟合 (原版, 有前视偏差)'}\n")
    L("---\n")

    L("## 一、各品种明细\n")
    L(_ROW_HEAD)
    L(_ROW_SEP)
    for s in sorted(grand):
        for (tf, sn), r in grand[s].items():
            L(_row(s, tf, sn, r))
    skipped = [(s, k, r) for s in sorted(grand) for k, r in grand[s].items()
               if r.get("skip_size") or r.get("skip_limit")]
    if skipped:
        L("\n**信号受阻统计** (资金不足买不起 1 手 / 涨跌停无法成交):\n")
        for s, (tf, sn), r in skipped:
            L(f"- {s} {tf} {sn}: 资金不足 {r.get('skip_size', 0)} 次, "
              f"涨跌停受阻 {r.get('skip_limit', 0)} 次")
    L("")

    L("---\n")
    _summary_block(L, grand, spec, "二、跨品种横向对比" + ("（基线）" if grand_garch else ""))

    if grand_garch:
        L("---\n")
        _summary_block(L, grand_garch, spec, "三、跨品种横向对比（GARCH 过滤）")
        deltas = []
        for s in grand:
            for k, rb in grand[s].items():
                rg = grand_garch.get(s, {}).get(k)
                if rg:
                    deltas.append(rg["calmar"] - rb["calmar"])
        if deltas:
            n_up = sum(d > 0 for d in deltas)
            L("### GARCH 影响汇总\n")
            L(f"- 对比组数: {len(deltas)}")
            L(f"- 改善: {n_up} ({n_up/len(deltas)*100:.0f}%), 恶化: {len(deltas)-n_up}")
            L(f"- 平均 Calmar 变化: {np.mean(deltas):+.3f}\n")

    # ── 组合与风险分析 ──
    L("---\n")
    L("## 组合层面 (各品种等权, 参考 TSMOM 论文的 diversified 组合)\n")
    combos = [(tf, sn) for tf, _, _ in spec.timeframes for sn, _, _ in SYSTEMS]
    rows = []
    for label, g in (("基线", grand), ("GARCH", grand_garch)):
        if not g:
            continue
        for tf, sn in combos:
            pf = ta.build_portfolio({s: g[s].get((tf, sn)) for s in g}, spec.days_per_year)
            rows.append((f"{label} {tf} {sn}", pf))
    L("\n".join(ta.portfolio_summary_table(rows)) + "\n")

    main_tf = "4h" if market == "crypto" else "1d"
    main_sys = "系统2 (60/20)"
    bench = None
    for label, g, suffix in (("基线", grand, ""), ("GARCH", grand_garch, "_garch")):
        if not g:
            continue
        pf = ta.build_portfolio({s: g[s].get((main_tf, main_sys)) for s in g}, spec.days_per_year)
        if pf is None:
            continue
        if bench is None:
            bench = ta.build_benchmark(prices or {}, pf["returns"].index)
        L(f"### {label}: {main_tf} × {main_sys}\n")
        L("\n".join(ta.analysis_section(pf, bench, spec.days_per_year, out_dir,
                                         str(figures_dir("turtle")), f"{market}{suffix}",
                                         f"{label} {main_tf} {main_sys}")))

    if failed:
        L(f"\n跳过 (数据获取失败): {', '.join(failed)}\n")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"turtle_multi_asset_report_{market}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存: {path}")
    return path


# ──────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────

def parse_mult(s: str) -> Dict[str, float]:
    out = {}
    for part in filter(None, (p.strip() for p in s.split(","))):
        k, v = part.split("=")
        out[k.strip().upper()] = float(v)
    return out


def main():
    p = argparse.ArgumentParser(description="海龟系统 — 多市场回测")
    p.add_argument("--market", choices=["crypto", "stock", "future"], default="crypto")
    p.add_argument("--symbols", default="",
                   help="逗号分隔: 加密 BTC,ETH / A股 600519,000001 / 期货 RB,CU")
    p.add_argument("--start", default=None,
                   help="起始日期 (默认: 加密 2022-01-01, A股/期货 2015-01-01)")
    p.add_argument("--source", choices=["csmar", "akshare", "csv"], default=None,
                   help="数据来源: csmar=data/securities 下的 CSMAR 日度文件 (A股默认) / "
                        "akshare (期货默认) / csv")
    p.add_argument("--csv-dir", default="", help="--source csv 时的文件目录")
    p.add_argument("--adjust", default="hfq", choices=["hfq", "qfq", ""],
                   help="A股复权方式: hfq 后复权 (默认) / qfq 前复权 / '' 不复权")
    p.add_argument("--capital", type=float, default=0, help="覆盖默认初始资金")
    p.add_argument("--slippage", type=float, default=None,
                   help="覆盖默认滑点 (比例, 如 0.0005 = 5 个基点; 0 = 不计滑点)")
    p.add_argument("--multiplier", default="", help="覆盖期货乘数, 如 RB=10,XX=5")
    p.add_argument("--futures-source", choices=["contracts", "main"], default="contracts",
                   help="期货: contracts=单月合约自拼复权连续 (默认) / main=新浪主力连续 (未复权)")
    p.add_argument("--futures-adjust", choices=["diff", "ratio", "none"], default="diff",
                   help="期货复权: diff 差值 (默认) / ratio 比例 / none 不复权")
    p.add_argument("--garch", action="store_true", help="启用 GARCH 过滤")
    p.add_argument("--garch-mode", choices=["rolling", "insample"], default="rolling",
                   help="rolling=滚动重估, 无前视偏差 (默认) / insample=原版全样本拟合")
    p.add_argument("--garch-refit-days", type=float, default=21,
                   help="滚动模式下每隔多少天重估一次参数 (默认 21)")
    p.add_argument("--garch-compare", action="store_true", help="同时跑有/无 GARCH")
    p.add_argument("--refresh", action="store_true", help="A股/期货: 忽略缓存重新拉取")
    args = p.parse_args()

    market = args.market
    if args.start is None:
        args.start = "2022-01-01" if market == "crypto" else "2015-01-01"
    if args.source is None:
        args.source = "csmar" if market == "stock" else "akshare"
    if args.source == "csmar" and market != "stock":
        p.error("--source csmar 只适用于 --market stock")
    if args.source == "csv" and not args.csv_dir:
        p.error("--source csv 需要同时指定 --csv-dir")
    args.mult_override = parse_mult(args.multiplier)

    spec = {"crypto": CRYPTO, "stock": STOCK, "future": FUTURE}[market]
    if args.capital > 0:
        spec = replace(spec, initial_capital=args.capital)
    if args.slippage is not None:
        spec = replace(spec, slippage=args.slippage)

    symbols = ([s.strip() for s in args.symbols.split(",") if s.strip()]
               or DEFAULT_SYMBOLS[market])

    print(f"\n{'='*80}")
    print(f"海龟交易系统 — {market}  |  {len(symbols)} 个品种: {', '.join(symbols)}")
    print(f"初始资金 {spec.initial_capital:,.0f}  |  1% 风险/ATR  |  2N 止损  |  起始 {args.start}")
    print(f"{'='*80}")

    grand, grand_garch, failed = {}, {}, []
    prices = {}          # 各品种复权收盘价, 用来算"等权买入持有"基准
    for idx, sym in enumerate(symbols, 1):
        print(f"\n[{idx}/{len(symbols)}] ── {sym} ──")
        try:
            if args.garch_compare:
                print("  --- 基线 ---")
                r, df = run_single_symbol(market, sym, spec, args, use_garch=False)
                if r is None:
                    failed.append(sym)
                    continue
                grand[sym] = r
                prices[sym] = df        # 含 raw_close/scale, 基准按真实价格算收益
                print("  --- GARCH ---")
                rg, _ = run_single_symbol(market, sym, spec, args, use_garch=True, df_raw=df)
                grand_garch[sym] = rg
            else:
                r, df = run_single_symbol(market, sym, spec, args, use_garch=args.garch)
                if r is None:
                    failed.append(sym)
                    print(f"  [!] {sym} 无数据, 跳过")
                    continue
                grand[sym] = r
                prices[sym] = df        # 含 raw_close/scale, 基准按真实价格算收益
        except Exception as e:
            failed.append(sym)
            print(f"  [!] {sym} 出错: {e}")

    if not grand:
        print("\n[!] 没有成功回测的品种")
        sys.exit(1)

    write_report(market, spec, grand, grand_garch, failed, args, str(REPORTS_DIR), prices)
    print(f"\n完成: {len(grand)} 个品种成功, {len(failed)} 个跳过。")


if __name__ == "__main__":
    main()
