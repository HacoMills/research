#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  海龟交易系统 — 多品种回测 & 选品工具

  功能:
    1. 通过 data/ 模块自动拉取数据 (带本地缓存)
    2. 多空独立参数, 随意调整
    3. 前 N 品种横向对比, 按 Calmar 排名选品

  安装依赖:
    pip install ccxt matplotlib pandas numpy

  使用:
    修改下方 ★ 用户参数 ★ 区域 → python turtle_screener.py
    首次运行自动拉数据并缓存到 data/cache/okx/
    改参数后重跑秒出结果 (不重新拉数据)
    加 --refresh 强制刷新数据
═══════════════════════════════════════════════════════════════
"""

import sys, os, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# 将项目根目录加入 sys.path, 以便 import data 模块
_project_root = str(next(p for p in Path(__file__).resolve().parents if (p / "data" / "paths.py").exists()))   # 往上找含 data/paths.py 的文件夹 = quant 根目录 (挪位置也不怕)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.fetch_data import (
    create_exchange, fetch_or_cache, fetch_batch,
    fetch_top_symbols, short_name, DEFAULT_SYMBOLS, CACHE_DIR,
)

# ════════════════════════════════════════════════════════════════
#  ★ 用户参数 — 在这里调整 ★
# ════════════════════════════════════════════════════════════════

# ─── 多头 ───
LONG_ENTRY  = 60          # 入场通道 (根K线)
LONG_EXIT   = 20          # 出场通道 (根K线)
LONG_STOP   = 2.0         # 止损倍数 (×ATR)

# ─── 空头 ───
SHORT_ENTRY = 60          # 入场通道 (根K线)
SHORT_EXIT  = 20          # 出场通道 (根K线)
SHORT_STOP  = 2.0         # 止损倍数 (×ATR)

# ─── 通用 ───
ATR_PERIOD  = 20          # ATR 周期 (根K线)
RISK_PCT    = 0.01        # 单笔风险 1%
CAPITAL     = 10000       # 初始资金 USDT
FEE_RATE    = 0.0005      # 手续费 0.05% (taker)

# ─── 数据 ───
TIMEFRAME   = '4h'        # K线周期: 15m / 1h / 4h / 1d
START_DATE  = '2022-01-01'
EXCHANGE_ID = 'okx'       # 交易所: okx / binance / bybit

# 品种列表: None → 自动获取成交量 Top N; 或手动指定列表覆盖
SYMBOLS = None            # None = 动态获取
TOP_N   = 20              # 动态获取的数量


# ════════════════════════════════════════════════════════════════
# 数据获取: 已迁移到 data/fetch_data.py
# 通过顶部 from data.fetch_data import ... 导入
# ════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════
#  回测引擎
# ════════════════════════════════════════════════════════════════

def compute_atr(high, low, close, period):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def run_backtest(df, long_entry, long_exit, long_stop,
                 short_entry, short_exit, short_stop,
                 atr_period, risk_pct, capital, fee_rate):
    """
    海龟回测, 多空独立参数.
    返回 (equity_curve, trades_list)
    """
    n = len(df)
    cl = df['close'].values
    hi = df['high'].values
    lo = df['low'].values

    atr = compute_atr(df['high'], df['low'], df['close'], atr_period).values

    le_ch = df['high'].rolling(long_entry).max().shift(1).values    # 多头入场
    lx_ch = df['low'].rolling(long_exit).min().shift(1).values      # 多头出场
    se_ch = df['low'].rolling(short_entry).min().shift(1).values    # 空头入场
    sx_ch = df['high'].rolling(short_exit).max().shift(1).values    # 空头出场

    equity = float(capital)
    pos = 0.0        # 持仓数量 (始终 >= 0)
    ep = 0.0         # 入场价
    sp = 0.0         # 止损价
    dire = 0         # 1=多, -1=空, 0=空仓
    e_bar = 0        # 入场 bar 索引
    e_fee = 0.0      # 入场手续费

    eq_curve = np.full(n, capital, dtype=float)
    trades = []
    warmup = max(long_entry, short_entry, atr_period) + 1

    for i in range(1, n):
        # ── 逐根 Mark-to-Market ──
        if dire == 1:
            equity += pos * (cl[i] - cl[i - 1])
        elif dire == -1:
            equity += pos * (cl[i - 1] - cl[i])

        if i < warmup or np.isnan(atr[i]) or atr[i] <= 0:
            eq_curve[i] = equity
            continue

        # ── 出场判断 ──
        exited = False
        if dire == 1:
            hit_exit = cl[i] < lx_ch[i] if not np.isnan(lx_ch[i]) else False
            hit_stop = cl[i] < sp
            if hit_exit or hit_stop:
                x_fee = pos * cl[i] * fee_rate
                equity -= x_fee
                gross = pos * (cl[i] - ep)
                trades.append(dict(
                    sym='', dire='LONG', ep=ep, xp=cl[i],
                    pnl=gross - e_fee - x_fee,
                    reason='止损' if hit_stop else '通道',
                    entry_t=df.index[e_bar], exit_t=df.index[i],
                    bars=i - e_bar,
                ))
                pos, dire = 0.0, 0
                exited = True

        elif dire == -1:
            hit_exit = cl[i] > sx_ch[i] if not np.isnan(sx_ch[i]) else False
            hit_stop = cl[i] > sp
            if hit_exit or hit_stop:
                x_fee = pos * cl[i] * fee_rate
                equity -= x_fee
                gross = pos * (ep - cl[i])
                trades.append(dict(
                    sym='', dire='SHORT', ep=ep, xp=cl[i],
                    pnl=gross - e_fee - x_fee,
                    reason='止损' if hit_stop else '通道',
                    entry_t=df.index[e_bar], exit_t=df.index[i],
                    bars=i - e_bar,
                ))
                pos, dire = 0.0, 0
                exited = True

        # ── 入场判断 (空仓时) ──
        if dire == 0:
            if not np.isnan(le_ch[i]) and cl[i] > le_ch[i]:
                # 做多
                qty = (equity * risk_pct) / atr[i]
                fee = qty * cl[i] * fee_rate
                equity -= fee
                pos, ep, sp = qty, cl[i], cl[i] - long_stop * atr[i]
                dire, e_bar, e_fee = 1, i, fee

            elif not np.isnan(se_ch[i]) and cl[i] < se_ch[i]:
                # 做空
                qty = (equity * risk_pct) / atr[i]
                fee = qty * cl[i] * fee_rate
                equity -= fee
                pos, ep, sp = qty, cl[i], cl[i] + short_stop * atr[i]
                dire, e_bar, e_fee = -1, i, fee

        eq_curve[i] = equity

    # 收尾: 未平仓按最后价格计算 (标记为 open)
    if dire != 0:
        gross = pos * (cl[-1] - ep) if dire == 1 else pos * (ep - cl[-1])
        trades.append(dict(
            sym='', dire='LONG' if dire == 1 else 'SHORT',
            ep=ep, xp=cl[-1], pnl=gross - e_fee,
            reason='未平仓', entry_t=df.index[e_bar], exit_t=df.index[-1],
            bars=n - 1 - e_bar,
        ))

    return eq_curve, trades


# ════════════════════════════════════════════════════════════════
#  绩效计算
# ════════════════════════════════════════════════════════════════

def bars_per_year(tf):
    """根据周期估算年化用的每年K线数."""
    m = {'1m': 525600, '5m': 105120, '15m': 35040, '30m': 17520,
         '1h': 8760, '2h': 4380, '4h': 2190, '6h': 1460,
         '8h': 1095, '12h': 730, '1d': 365, '3d': 122, '1w': 52}
    return m.get(tf, 2190)


def calc_metrics(eq, trades, capital, tf='4h'):
    """计算完整绩效指标."""
    final = eq[-1]
    ret = (final / capital - 1) * 100

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()

    bpy = bars_per_year(tf)
    years = len(eq) / bpy if bpy else 1
    ann_ret = ((final / capital) ** (1 / years) - 1) * 100 if years > 0 else 0
    calmar = (ann_ret / 100) / abs(max_dd / 100) if max_dd != 0 else 0

    rets = np.diff(eq) / eq[:-1]
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(bpy)
    else:
        sharpe = 0

    n_tr = len(trades)
    if n_tr > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        wr = len(wins) / n_tr * 100
        gp = sum(t['pnl'] for t in wins)
        gl = abs(sum(t['pnl'] for t in losses)) or 1e-9
        pf = gp / gl
        avg_w = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_l = np.mean([abs(t['pnl']) for t in losses]) if losses else 1e-9
        payoff = avg_w / avg_l
        long_pnl = sum(t['pnl'] for t in trades if t['dire'] == 'LONG')
        short_pnl = sum(t['pnl'] for t in trades if t['dire'] == 'SHORT')
        n_long = sum(1 for t in trades if t['dire'] == 'LONG')
        n_short = sum(1 for t in trades if t['dire'] == 'SHORT')
    else:
        wr = pf = payoff = long_pnl = short_pnl = 0
        n_long = n_short = 0

    return dict(
        total_return=ret, annual_return=ann_ret, max_dd=max_dd,
        sharpe=sharpe, calmar=calmar, profit_factor=pf,
        win_rate=wr, payoff=payoff, n_trades=n_tr,
        n_long=n_long, n_short=n_short,
        long_pnl=long_pnl, short_pnl=short_pnl,
        final_equity=final,
    )


# ════════════════════════════════════════════════════════════════
#  可视化
# ════════════════════════════════════════════════════════════════

def setup_font():
    """尝试设置中文字体."""
    import matplotlib.font_manager as fm
    candidates = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei',
                  'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB']
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams['font.sans-serif'] = [c, 'DejaVu Sans']
            break
    plt.rcParams['axes.unicode_minus'] = False


def plot_comparison(results, output_path):
    """生成4面板对比图."""
    setup_font()

    # 按 Calmar 排序
    ranked = sorted(results, key=lambda r: r['metrics']['calmar'], reverse=True)
    names = [r['name'] for r in ranked]
    metrics_list = [r['metrics'] for r in ranked]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'海龟选品对比  |  多头 {LONG_ENTRY}/{LONG_EXIT}  空头 {SHORT_ENTRY}/{SHORT_EXIT}'
                 f'  |  ATR {ATR_PERIOD}  |  {TIMEFRAME}', fontsize=14, fontweight='bold')

    # ── 1) Top 5 权益曲线 (归一化 100) ──
    ax = axes[0, 0]
    top5 = ranked[:5]
    for r in top5:
        eq_norm = r['eq'] / r['eq'][0] * 100
        ax.plot(r['dates'], eq_norm, label=f"{r['name']} ({r['metrics']['total_return']:+.0f}%)", linewidth=1.5)
    ax.set_title('Top 5 权益曲线 (归一化)')
    ax.set_ylabel('净值')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 2) 总收益率 (水平柱状) ──
    ax = axes[0, 1]
    rets = [m['total_return'] for m in metrics_list]
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in rets]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, rets, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('总收益率 (%)')
    ax.set_title('总收益率排名')
    ax.invert_yaxis()
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # ── 3) Calmar Ratio (水平柱状) ──
    ax = axes[1, 0]
    calmars = [m['calmar'] for m in metrics_list]
    colors = ['#3498db' if c > 0 else '#e74c3c' for c in calmars]
    ax.barh(y_pos, calmars, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Calmar Ratio')
    ax.set_title('Calmar 排名 (年化收益 / 最大回撤)')
    ax.invert_yaxis()
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # ── 4) 收益 vs 回撤 散点 ──
    ax = axes[1, 1]
    for r in ranked:
        m = r['metrics']
        c = '#2ecc71' if m['total_return'] > 0 else '#e74c3c'
        ax.scatter(abs(m['max_dd']), m['total_return'], s=80, c=c, edgecolors='white', zorder=5)
        ax.annotate(r['name'], (abs(m['max_dd']), m['total_return']),
                    fontsize=8, ha='left', va='bottom', xytext=(4, 4),
                    textcoords='offset points')
    ax.set_xlabel('最大回撤 (%)')
    ax.set_ylabel('总收益率 (%)')
    ax.set_title('收益 vs 回撤')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n图表已保存: {output_path}')


def print_ranking(results, tf):
    """打印排名表格."""
    ranked = sorted(results, key=lambda r: r['metrics']['calmar'], reverse=True)

    print('\n' + '═' * 110)
    print(f'  海龟选品排名  |  多头 {LONG_ENTRY}/{LONG_EXIT}  空头 {SHORT_ENTRY}/{SHORT_EXIT}'
          f'  |  ATR {ATR_PERIOD}  |  {tf}  |  风险 {RISK_PCT*100:.1f}%')
    print('═' * 110)
    header = f'{"排名":>4s}  {"品种":>6s}  {"总收益%":>8s}  {"年化%":>7s}  {"最大回撤%":>9s}  ' \
             f'{"Calmar":>7s}  {"Sharpe":>7s}  {"PF":>5s}  {"胜率%":>6s}  ' \
             f'{"交易数":>6s}  {"多/空":>7s}  {"多PnL":>8s}  {"空PnL":>8s}'
    print(header)
    print('─' * 110)

    for rank, r in enumerate(ranked, 1):
        m = r['metrics']
        ls = f"{m['n_long']}/{m['n_short']}"
        print(f'{rank:>4d}  {r["name"]:>6s}  {m["total_return"]:>+8.1f}  {m["annual_return"]:>+7.1f}'
              f'  {m["max_dd"]:>9.1f}  {m["calmar"]:>7.2f}  {m["sharpe"]:>7.2f}'
              f'  {m["profit_factor"]:>5.2f}  {m["win_rate"]:>6.1f}  {m["n_trades"]:>6d}'
              f'  {ls:>7s}  {m["long_pnl"]:>+8.0f}  {m["short_pnl"]:>+8.0f}')

    print('═' * 110)

    # 简要结论
    best = ranked[0]
    worst = ranked[-1]
    profitable = [r for r in ranked if r['metrics']['total_return'] > 0]
    print(f'\n  ✓ 盈利品种: {len(profitable)}/{len(ranked)}')
    print(f'  ★ 最佳: {best["name"]}  (Calmar {best["metrics"]["calmar"]:.2f},'
          f' 收益 {best["metrics"]["total_return"]:+.1f}%, 回撤 {best["metrics"]["max_dd"]:.1f}%)')
    print(f'  ✗ 最差: {worst["name"]}  (Calmar {worst["metrics"]["calmar"]:.2f},'
          f' 收益 {worst["metrics"]["total_return"]:+.1f}%, 回撤 {worst["metrics"]["max_dd"]:.1f}%)')
    print()


# ════════════════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════════════════

def main():
    force_refresh = '--refresh' in sys.argv

    print('╔══════════════════════════════════════════════════════╗')
    print('║     海龟交易系统 — 多品种回测 & 选品工具            ║')
    print('╚══════════════════════════════════════════════════════╝')

    # ── 0. 解析品种列表 ──
    global SYMBOLS
    if SYMBOLS is None:
        print('\n[ 获取品种 ]')
        SYMBOLS = fetch_top_symbols(exchange_id=EXCHANGE_ID, top_n=TOP_N)

    print(f'\n参数: 多头 {LONG_ENTRY}/{LONG_EXIT}/{LONG_STOP}N'
          f'  空头 {SHORT_ENTRY}/{SHORT_EXIT}/{SHORT_STOP}N'
          f'  ATR {ATR_PERIOD}  风险 {RISK_PCT*100:.1f}%  {TIMEFRAME}')
    print(f'品种: {len(SYMBOLS)} 个\n')

    # ── 1. 获取数据 (通过 data/ 模块) ──
    print('[ 数据获取 ]')
    min_bars = max(LONG_ENTRY, SHORT_ENTRY, ATR_PERIOD) + 50
    datasets = fetch_batch(
        symbols=SYMBOLS,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        exchange_id=EXCHANGE_ID,
        force=force_refresh,
    )

    if not datasets:
        print('未获取到任何数据, 请检查网络或交易所设置')
        return

    # 过滤数据不足的品种
    datasets = {sym: df for sym, df in datasets.items() if len(df) > min_bars}

    # ── 2. 回测 ──
    print('\n[ 回测运行 ]')
    results = []
    for sym, df in datasets.items():
        name = short_name(sym)
        print(f'  {name:>6s} 回测中...', end='', flush=True)
        eq, trades = run_backtest(
            df, LONG_ENTRY, LONG_EXIT, LONG_STOP,
            SHORT_ENTRY, SHORT_EXIT, SHORT_STOP,
            ATR_PERIOD, RISK_PCT, CAPITAL, FEE_RATE,
        )
        for t in trades:
            t['sym'] = name
        m = calc_metrics(eq, trades, CAPITAL, TIMEFRAME)
        results.append(dict(name=name, sym=sym, eq=eq, trades=trades,
                            metrics=m, dates=df.index))
        print(f' 收益 {m["total_return"]:+.1f}%  回撤 {m["max_dd"]:.1f}%'
              f'  Calmar {m["calmar"]:.2f}  交易 {m["n_trades"]}笔')

    # ── 3. 排名输出 ──
    print_ranking(results, TIMEFRAME)

    # ── 4. 图表 ──
    from data.paths import figures, reports
    fig_dir, rep_dir = figures('turtle'), reports('turtle')   # quant/figures/turtle, quant/reports/turtle
    output_img = fig_dir / 'turtle_screener_results.png'
    plot_comparison(results, str(output_img))

    # ── 5. 保存 CSV ──
    output_csv = rep_dir / 'turtle_screener_results.csv'
    rows = []
    for r in results:
        m = r['metrics']
        m_copy = m.copy()
        m_copy['symbol'] = r['name']
        rows.append(m_copy)
    pd.DataFrame(rows).to_csv(str(output_csv), index=False)
    print(f'指标已保存: {output_csv}')

    # ── 6. 保存交易明细 ──
    all_trades = []
    for r in results:
        all_trades.extend(r['trades'])
    if all_trades:
        trades_csv = rep_dir / 'turtle_screener_trades.csv'
        pd.DataFrame(all_trades).to_csv(str(trades_csv), index=False)
        print(f'交易明细: {trades_csv}')

    print('\n完成!')


if __name__ == '__main__':
    main()
