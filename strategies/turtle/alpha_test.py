#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  海龟策略 Alpha 检验
  ------------------------------------------------------------
  输出:
    1. 策略 vs BTC buy&hold vs 等权指数
    2. CAPM 回归: alpha, beta, t值, p值
    3. 多因子回归 (BTC + 市场等权指数)
    4. 多空归因
    5. 逐年分解
    6. 单笔交易 t 检验
    7. 蒙特卡洛置换检验 (随机入场对照)
═══════════════════════════════════════════════════════════════
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_project_root = str(next(p for p in Path(__file__).resolve().parents if (p / "data" / "paths.py").exists()))   # quant 根目录
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.fetch_data import (
    create_exchange, fetch_or_cache, fetch_batch,
    fetch_top_symbols, short_name, DEFAULT_SYMBOLS,
)

# ════════════════════════════════════════════════════════════
#  ★ 参数: 与 turtle_screener.py 保持一致 ★
# ════════════════════════════════════════════════════════════
LONG_ENTRY, LONG_EXIT, LONG_STOP   = 60, 20, 2.0
SHORT_ENTRY, SHORT_EXIT, SHORT_STOP = 60, 20, 2.0
ATR_PERIOD = 20
RISK_PCT   = 0.01
CAPITAL    = 10000
FEE_RATE   = 0.0005
SLIPPAGE   = 0.0005         # 新增: 滑点
TIMEFRAME  = '4h'
START_DATE = '2022-01-01'
EXCHANGE_ID = 'okx'
TOP_N      = 20
SYMBOLS    = None
RISK_FREE  = 0.04           # 无风险年化利率 (用于 CAPM)
BARS_PER_YEAR = 2190        # 4h


# ════════════════════════════════════════════════════════════
#  回测引擎 (与 screener 相同, 但加了滑点)
# ════════════════════════════════════════════════════════════
def compute_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def run_backtest(df, fee_rate=FEE_RATE, slippage=SLIPPAGE):
    n = len(df)
    cl, hi, lo = df['close'].values, df['high'].values, df['low'].values
    atr = compute_atr(df['high'], df['low'], df['close'], ATR_PERIOD).values

    le_ch = df['high'].rolling(LONG_ENTRY).max().shift(1).values
    lx_ch = df['low'].rolling(LONG_EXIT).min().shift(1).values
    se_ch = df['low'].rolling(SHORT_ENTRY).min().shift(1).values
    sx_ch = df['high'].rolling(SHORT_EXIT).max().shift(1).values

    equity = float(CAPITAL)
    pos, ep, sp, dire, e_bar, e_fee = 0.0, 0.0, 0.0, 0, 0, 0.0
    eq_curve = np.full(n, CAPITAL, dtype=float)
    trades = []
    warmup = max(LONG_ENTRY, SHORT_ENTRY, ATR_PERIOD) + 1

    for i in range(1, n):
        # Mark-to-Market
        if dire == 1:
            equity += pos * (cl[i] - cl[i - 1])
        elif dire == -1:
            equity += pos * (cl[i - 1] - cl[i])

        if i < warmup or np.isnan(atr[i]) or atr[i] <= 0:
            eq_curve[i] = equity
            continue

        exited = False
        # ── 出场: 盘中触发 ──
        if dire == 1:
            hit_exit = (not np.isnan(lx_ch[i])) and lo[i] < lx_ch[i]
            hit_stop = lo[i] < sp
            if hit_exit or hit_stop:
                raw_x = min(lo[i], sp) if hit_stop else lx_ch[i]
                xp = raw_x * (1 - slippage)
                x_fee = pos * xp * fee_rate
                equity -= x_fee
                gross = pos * (xp - ep)
                trades.append(dict(
                    dire='LONG', ep=ep, xp=xp, pnl=gross - e_fee - x_fee,
                    reason='止损' if hit_stop else '通道',
                    entry_t=df.index[e_bar], exit_t=df.index[i], bars=i - e_bar))
                pos, dire, exited = 0.0, 0, True

        elif dire == -1:
            hit_exit = (not np.isnan(sx_ch[i])) and hi[i] > sx_ch[i]
            hit_stop = hi[i] > sp
            if hit_exit or hit_stop:
                raw_x = max(hi[i], sp) if hit_stop else sx_ch[i]
                xp = raw_x * (1 + slippage)
                x_fee = pos * xp * fee_rate
                equity -= x_fee
                gross = pos * (ep - xp)
                trades.append(dict(
                    dire='SHORT', ep=ep, xp=xp, pnl=gross - e_fee - x_fee,
                    reason='止损' if hit_stop else '通道',
                    entry_t=df.index[e_bar], exit_t=df.index[i], bars=i - e_bar))
                pos, dire, exited = 0.0, 0, True

        # ── 入场 ──
        if dire == 0:
            if not np.isnan(le_ch[i]) and cl[i] > le_ch[i]:
                ep_raw = cl[i] * (1 + slippage)
                stop_dist = LONG_STOP * atr[i]
                qty = (equity * RISK_PCT) / stop_dist   # ★ 修正: 除以止损距离
                fee = qty * ep_raw * fee_rate
                equity -= fee
                pos, ep, sp = qty, ep_raw, ep_raw - stop_dist
                dire, e_bar, e_fee = 1, i, fee
            elif not np.isnan(se_ch[i]) and cl[i] < se_ch[i]:
                ep_raw = cl[i] * (1 - slippage)
                stop_dist = SHORT_STOP * atr[i]
                qty = (equity * RISK_PCT) / stop_dist
                fee = qty * ep_raw * fee_rate
                equity -= fee
                pos, ep, sp = qty, ep_raw, ep_raw + stop_dist
                dire, e_bar, e_fee = -1, i, fee

        eq_curve[i] = equity

    # 收尾
    if dire != 0:
        xp = cl[-1]
        gross = pos * (xp - ep) if dire == 1 else pos * (ep - xp)
        x_fee = pos * xp * fee_rate
        trades.append(dict(
            dire='LONG' if dire == 1 else 'SHORT', ep=ep, xp=xp,
            pnl=gross - e_fee - x_fee, reason='未平仓',
            entry_t=df.index[e_bar], exit_t=df.index[-1], bars=n - 1 - e_bar))

    return eq_curve, trades


# ════════════════════════════════════════════════════════════
#  统计工具
# ════════════════════════════════════════════════════════════
def ols(y, X):
    """带截距的最小二乘, 返回 (params, t_values, r2, resid)."""
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return beta, tvals, r2, resid


def p_value_two_sided(t, df):
    """无 scipy 时的 t 分布近似 p 值 (用正态近似, n 大时足够)."""
    try:
        from scipy import stats
        return 2 * (1 - stats.t.cdf(abs(t), df))
    except ImportError:
        # 正态近似
        import math
        return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def perf_stats(eq):
    """给定权益曲线, 返回总收益/年化/最大回撤/Sharpe."""
    eq = np.asarray(eq, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    total = eq[-1] / eq[0] - 1
    years = len(eq) / BARS_PER_YEAR
    ann = (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = dd.min()
    sharpe = (rets.mean() / rets.std() * np.sqrt(BARS_PER_YEAR)) if rets.std() > 0 else 0
    return dict(total=total, ann=ann, mdd=mdd, sharpe=sharpe,
                rets=rets, years=years)


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════
def main():
    print('╔════════════════════════════════════════════════════╗')
    print('║   海龟策略 Alpha 检验                              ║')
    print('╚════════════════════════════════════════════════════╝')

    # ── 1. 数据 ──
    global SYMBOLS
    if SYMBOLS is None:
        print('\n[ 获取品种 ]')
        SYMBOLS = fetch_top_symbols(exchange_id=EXCHANGE_ID, top_n=TOP_N)

    print(f'\n[ 数据获取 ] {len(SYMBOLS)} 品种, {TIMEFRAME}, {START_DATE} 起')
    datasets = fetch_batch(
        symbols=SYMBOLS, timeframe=TIMEFRAME,
        start_date=START_DATE, exchange_id=EXCHANGE_ID,
    )
    datasets = {s: df for s, df in datasets.items() if len(df) > 200}

    # ── 2. 逐币回测 ──
    print('\n[ 回测 ]')
    results = {}
    for sym, df in datasets.items():
        name = short_name(sym)
        eq, trades = run_backtest(df)
        results[name] = dict(sym=sym, df=df, eq=eq, trades=trades)

    # ── 3. 基准: BTC buy&hold, 等权指数 ──
    print('\n[ 基准构建 ]')
    # 统一到公共时间轴
    common_idx = None
    for r in results.values():
        common_idx = r['df'].index if common_idx is None else common_idx.intersection(r['df'].index)
    common_idx = common_idx.sort_values()

    btc_sym = next((s for s in datasets if short_name(s) == 'BTC'), None)
    if btc_sym is None:
        print('  警告: 未找到 BTC, 用第一个币作市场基准')
        btc_sym = list(datasets.keys())[0]
    btc_close = datasets[btc_sym]['close'].reindex(common_idx).ffill()
    btc_ret = btc_close.pct_change().fillna(0).values

    # 等权指数: 所有品种收益等权
    all_rets = []
    for r in results.values():
        c = r['df']['close'].reindex(common_idx).ffill()
        all_rets.append(c.pct_change().fillna(0).values)
    mkt_ret = np.mean(all_rets, axis=0)

    # ── 4. 策略组合权益: 等权分配合并 ──
    # 每个币的权益归一到 1, 求平均
    eqs_norm = []
    for r in results.values():
        eq = r['eq']
        # 保证长度与 common_idx 一致 (用 reindex)
        s = pd.Series(eq, index=r['df'].index).reindex(common_idx).ffill().bfill()
        eqs_norm.append((s / s.iloc[0]).values)
    port_eq = np.mean(eqs_norm, axis=0) * CAPITAL
    port_stats = perf_stats(port_eq)

    btc_stats = perf_stats(btc_close.values)
    mkt_stats = perf_stats(np.cumprod(1 + mkt_ret))

    print(f'\n  {"组合":<12s}  总收益 {port_stats["total"]*100:+7.1f}%  '
          f'年化 {port_stats["ann"]*100:+6.1f}%  '
          f'回撤 {port_stats["mdd"]*100:6.1f}%  '
          f'Sharpe {port_stats["sharpe"]:.2f}')
    print(f'  {"BTC B&H":<12s}  总收益 {btc_stats["total"]*100:+7.1f}%  '
          f'年化 {btc_stats["ann"]*100:+6.1f}%  '
          f'回撤 {btc_stats["mdd"]*100:6.1f}%  '
          f'Sharpe {btc_stats["sharpe"]:.2f}')
    print(f'  {"等权指数":<12s}  总收益 {mkt_stats["total"]*100:+7.1f}%  '
          f'年化 {mkt_stats["ann"]*100:+6.1f}%  '
          f'回撤 {mkt_stats["mdd"]*100:6.1f}%  '
          f'Sharpe {mkt_stats["sharpe"]:.2f}')

    # ── 5. CAPM 回归: R_p - Rf = alpha + beta*(Rm - Rf) + e ──
    print('\n[ CAPM 回归 ]')
    port_ret = np.diff(port_eq) / port_eq[:-1]
    port_ret = np.nan_to_num(port_ret)

    rf_per_bar = RISK_FREE / BARS_PER_YEAR
    excess_p = port_ret - rf_per_bar
    excess_m = btc_ret[1:] - rf_per_bar

    # 对齐长度
    L = min(len(excess_p), len(excess_m))
    excess_p, excess_m = excess_p[:L], excess_m[:L]

    beta, tvals, r2, resid = ols(excess_p, excess_m.reshape(-1, 1))
    alpha_per_bar, beta_val = beta[0], beta[1]
    t_alpha, t_beta = tvals[0], tvals[1]
    df_resid = L - 2
    p_alpha = p_value_two_sided(t_alpha, df_resid)

    # 年化 alpha
    alpha_ann = alpha_per_bar * BARS_PER_YEAR

    print(f'  对 BTC 单因子:')
    print(f'    Alpha (年化) = {alpha_ann*100:+.2f}%   t = {t_alpha:+.2f}   p = {p_alpha:.4f}')
    print(f'    Beta         = {beta_val:.3f}        t = {t_beta:+.2f}')
    print(f'    R²           = {r2:.3f}')

    if p_alpha < 0.05 and alpha_ann > 0:
        print('    → Alpha 在 5% 水平显著为正')
    elif p_alpha < 0.05 and alpha_ann < 0:
        print('    → Alpha 显著为负, 策略跑输 BTC 风险调整后收益')
    else:
        print('    → Alpha 不显著 (无法拒绝 alpha=0)')

    # ── 6. 双因子: BTC + 等权指数 ──
    print('\n[ 双因子回归: BTC + 市场等权 ]')
    excess_m2 = mkt_ret[1:][:L] - rf_per_bar
    X2 = np.column_stack([excess_m, excess_m2])
    beta2, tvals2, r2_2, _ = ols(excess_p, X2)
    alpha2_ann = beta2[0] * BARS_PER_YEAR
    t_alpha2 = tvals2[0]
    p_alpha2 = p_value_two_sided(t_alpha2, L - 3)

    print(f'    Alpha (年化) = {alpha2_ann*100:+.2f}%   t = {t_alpha2:+.2f}   p = {p_alpha2:.4f}')
    print(f'    Beta_BTC     = {beta2[1]:.3f}')
    print(f'    Beta_MKT     = {beta2[2]:.3f}')
    print(f'    R²           = {r2_2:.3f}')

    # ── 7. 多空归因 ──
    print('\n[ 多空归因 ]')
    all_trades = []
    for name, r in results.items():
        for t in r['trades']:
            t['sym'] = name
            all_trades.append(t)
    tdf = pd.DataFrame(all_trades)

    if len(tdf):
        for dire in ['LONG', 'SHORT']:
            sub = tdf[tdf['dire'] == dire]
            if len(sub):
                wr = (sub['pnl'] > 0).mean() * 100
                print(f'  {dire:<5s}  笔数 {len(sub):>3d}  总PnL {sub["pnl"].sum():>+9.0f}  '
                      f'胜率 {wr:5.1f}%  均值 {sub["pnl"].mean():>+8.2f}')
        # 多空占比
        long_pnl = tdf[tdf['dire'] == 'LONG']['pnl'].sum()
        short_pnl = tdf[tdf['dire'] == 'SHORT']['pnl'].sum()
        total_pnl = long_pnl + short_pnl
        if abs(total_pnl) > 1e-9:
            print(f'  多头贡献 {long_pnl/total_pnl*100:+.1f}%   空头贡献 {short_pnl/total_pnl*100:+.1f}%')

    # ── 8. 逐年分解 ──
    print('\n[ 逐年分解 (组合) ]')
    port_series = pd.Series(port_eq, index=common_idx)
    yearly = port_series.resample('YE').last()
    yearly_ret = yearly.pct_change().fillna(yearly.iloc[0] / CAPITAL - 1)
    # 更稳的方式: 按自然年分组
    yearly_ret = port_series.groupby(port_series.index.year).apply(
        lambda s: s.iloc[-1] / s.iloc[0] - 1)

    btc_series = btc_close / btc_close.iloc[0] * CAPITAL
    btc_yearly = btc_series.groupby(btc_series.index.year).apply(
        lambda s: s.iloc[-1] / s.iloc[0] - 1)

    print(f'  {"年份":<6s}  {"策略":>10s}  {"BTC B&H":>10s}  {"超额":>10s}')
    for yr in sorted(set(yearly_ret.index) | set(btc_yearly.index)):
        p = yearly_ret.get(yr, np.nan) * 100
        b = btc_yearly.get(yr, np.nan) * 100
        diff = p - b if not (np.isnan(p) or np.isnan(b)) else np.nan
        print(f'  {yr:<6d}  {p:>+9.1f}%  {b:>+9.1f}%  {diff:>+9.1f}%')

    # ── 9. 单笔交易 t 检验 ──
    print('\n[ 单笔交易 t 检验 ]')
    if len(tdf):
        pnls = tdf['pnl'].values
        # 收益率 (相对入场时权益, 这里用名义仓位收益)
        mean_pnl = pnls.mean()
        std_pnl = pnls.std(ddof=1)
        n = len(pnls)
        t_stat = mean_pnl / (std_pnl / np.sqrt(n)) if std_pnl > 0 else 0
        p_val = p_value_two_sided(t_stat, n - 1)
        print(f'  均值 {mean_pnl:+.2f}  标准差 {std_pnl:.2f}  n={n}')
        print(f'  t = {t_stat:+.2f}   p = {p_val:.4f}')
        if p_val < 0.05:
            print('  → 单笔收益均值显著不为 0')
        else:
            print('  → 单笔收益均值不显著 (可能是运气)')

    # ── 10. 蒙特卡洛置换检验 ──
    print('\n[ 蒙特卡洛: 随机入场对照 ]')
    n_sim = 200
    btc_df = datasets[btc_sym]
    rand_ret = []
    np.random.seed(42)
    for _ in range(n_sim):
        # 随机方向, 随机入场点, 持有一段时间
        n_bars = len(btc_df)
        n_trades_sim = len(tdf) if len(tdf) else 100
        hold = int(n_bars / n_trades_sim)
        pnl_sum = 0
        for _ in range(n_trades_sim):
            i = np.random.randint(ATR_PERIOD + 2, n_bars - hold - 1)
            d = np.random.choice([1, -1])
            entry = btc_df['close'].iloc[i]
            exit_ = btc_df['close'].iloc[i + hold]
            ret = (exit_ - entry) / entry if d == 1 else (entry - exit_) / entry
            pnl_sum += ret * (CAPITAL * RISK_PCT * 100)  # 粗略
        rand_ret.append(pnl_sum)
    rand_ret = np.array(rand_ret)

    strat_total = port_stats['total'] * CAPITAL
    pctile = (rand_ret < strat_total).mean() * 100
    print(f'  随机策略 PnL 分布: 均值 {rand_ret.mean():+.0f}  '
          f'std {rand_ret.std():.0f}  '
          f'5% {np.percentile(rand_ret,5):+.0f}  95% {np.percentile(rand_ret,95):+.0f}')
    print(f'  实际策略 PnL: {strat_total:+.0f}')
    print(f'  分位数: {pctile:.1f}%')
    if pctile > 95:
        print('  → 策略显著优于随机入场 (>95%)')
    elif pctile < 5:
        print('  → 策略显著差于随机入场')
    else:
        print('  → 策略与随机入场无显著差异')

    # ── 11. 结论 ──
    print('\n' + '═' * 60)
    print('  结论')
    print('═' * 60)
    excess_total = port_stats['total'] - btc_stats['total']
    print(f'  策略总收益      : {port_stats["total"]*100:+.1f}%')
    print(f'  BTC B&H 总收益  : {btc_stats["total"]*100:+.1f}%')
    print(f'  超额 (总收益差) : {excess_total*100:+.1f}%')
    print(f'  年化 Alpha (CAPM): {alpha_ann*100:+.2f}%  (p={p_alpha:.3f})')
    print(f'  Beta (对 BTC)   : {beta_val:.3f}')
    print()
    if p_alpha < 0.05 and alpha_ann > 0 and excess_total > 0:
        print('  ★ 有统计显著的 alpha')
    elif excess_total > 0 and p_alpha >= 0.05:
        print('  ⚠ 总收益跑赢 BTC, 但 alpha 不显著 — 可能是 beta 或运气')
    elif excess_total <= 0:
        print('  ✗ 未跑赢 BTC, 无 alpha 证据')
    else:
        print('  ? 证据不足, 需更多样本外测试')
    print()


if __name__ == '__main__':
    main()