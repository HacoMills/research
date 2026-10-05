#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  海龟交易系统 — 实时信号扫描 & 持仓跟踪

  功能:
    1. 拉取最新K线, 计算唐奇安通道 + ATR
    2. 扫描多空入场 / 出场 / 止损信号
    3. 本地持仓记录, 跟踪浮盈与出场条件
    4. 建议仓位大小 (基于 ATR 和风险比例)

  使用:
    python turtle_signal.py                  # 扫描信号
    python turtle_signal.py --watch 300      # 每5分钟自动刷新
    python turtle_signal.py --status         # 查看持仓
    python turtle_signal.py --record BTC long 63200   # 记录开仓
    python turtle_signal.py --close  BTC 65000        # 记录平仓
    python turtle_signal.py --symbols BTC ETH --tf 1h # 指定品种/周期
═══════════════════════════════════════════════════════════════
"""

import sys, os, json, time, warnings, argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── 项目路径 ──
_project_root = str(next(p for p in Path(__file__).resolve().parents if (p / "data" / "paths.py").exists()))   # 往上找含 data/paths.py 的文件夹 = quant 根目录 (挪位置也不怕)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.fetch_data import (
    create_exchange, fetch_ohlcv, fetch_top_symbols,
    short_name, DEFAULT_SYMBOLS,
    fetch_account_positions, fetch_account_balance, has_auth_config,
)

# ════════════════════════════════════════════════════════════════
#  ★ 参数 — 在这里调整 (与 screener 保持一致) ★
# ════════════════════════════════════════════════════════════════

LONG_ENTRY  = 60          # 多头入场通道
LONG_EXIT   = 20          # 多头出场通道
LONG_STOP   = 2.0         # 多头止损 ×ATR

SHORT_ENTRY = 60          # 空头入场通道
SHORT_EXIT  = 20          # 空头出场通道
SHORT_STOP  = 2.0         # 空头止损 ×ATR

ATR_PERIOD  = 20          # ATR 周期
RISK_PCT    = 0.01        # 单笔风险 1%
CAPITAL     = 10000       # 账户资金 USDT
FEE_RATE    = 0.0005      # 手续费 0.05%

TIMEFRAME   = '4h'
EXCHANGE_ID = 'okx'
SYMBOLS     = None        # None → 自动获取成交量 Top N; 或手动指定列表
TOP_N       = 20          # 动态获取的数量

NEAR_PCT    = 2.0         # 距通道 <2% 视为"接近"

# ── 持仓文件 ──
POSITIONS_FILE = Path(__file__).resolve().parent / 'turtle_positions.json'


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════

def compute_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


TF_MINUTES = {'1m':1, '5m':5, '15m':15, '30m':30, '1h':60,
              '2h':120, '4h':240, '6h':360, '8h':480, '12h':720,
              '1d':1440, '3d':4320, '1w':10080}


def calc_start_date(tf, bars_needed):
    """根据周期和所需K线数推算起始日期."""
    mins = TF_MINUTES.get(tf, 240)
    delta = timedelta(minutes=mins * bars_needed * 1.5)
    return (datetime.now(timezone.utc) - delta).strftime('%Y-%m-%d')


def fmt_price(p):
    if p >= 1000:   return f'{p:,.0f}'
    if p >= 1:      return f'{p:,.2f}'
    if p >= 0.01:   return f'{p:,.4f}'
    return f'{p:,.6f}'


# ════════════════════════════════════════════════════════════════
#  持仓管理 (JSON 文件)
# ════════════════════════════════════════════════════════════════

def load_positions():
    if POSITIONS_FILE.exists():
        try:
            return json.load(open(POSITIONS_FILE))
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_positions(pos):
    with open(POSITIONS_FILE, 'w') as f:
        json.dump(pos, f, indent=2, ensure_ascii=False, default=str)


def add_position(symbol, direction, price, stop, qty, atr):
    positions = load_positions()
    positions[symbol] = dict(
        direction=direction, entry_price=price,
        stop_loss=stop, quantity=qty, atr=atr,
        entry_time=datetime.now(timezone.utc).isoformat(),
    )
    save_positions(positions)
    return positions[symbol]


def remove_position(symbol, exit_price=None):
    positions = load_positions()
    if symbol not in positions:
        return None
    pos = positions.pop(symbol)
    save_positions(positions)
    if exit_price is not None:
        if pos['direction'] == 'long':
            pos['pnl'] = pos['quantity'] * (exit_price - pos['entry_price'])
        else:
            pos['pnl'] = pos['quantity'] * (pos['entry_price'] - exit_price)
        pos['exit_price'] = exit_price
    return pos


# ════════════════════════════════════════════════════════════════
#  信号扫描核心
# ════════════════════════════════════════════════════════════════

def scan_symbol(df, symbol, positions):
    """扫描单个品种, 返回信号字典."""
    name = short_name(symbol)

    atr_s  = compute_atr(df['high'], df['low'], df['close'], ATR_PERIOD)
    up_l   = df['high'].rolling(LONG_ENTRY).max().shift(1)    # 多头入场线
    lo_l   = df['low'].rolling(LONG_EXIT).min().shift(1)      # 多头出场线
    lo_s   = df['low'].rolling(SHORT_ENTRY).min().shift(1)    # 空头入场线
    up_s   = df['high'].rolling(SHORT_EXIT).max().shift(1)    # 空头出场线

    price    = df['close'].iloc[-1]
    atr_val  = atr_s.iloc[-1]
    ch_up    = up_l.iloc[-1]
    ch_lo    = lo_s.iloc[-1]
    exit_lo  = lo_l.iloc[-1]      # 多头出场通道
    exit_up  = up_s.iloc[-1]      # 空头出场通道
    bar_time = df.index[-1]

    if pd.isna(atr_val) or pd.isna(ch_up):
        return dict(name=name, symbol=symbol, price=price, bar_time=bar_time,
                    signal='数据不足', signal_type='none',
                    upper=0, lower=0, atr=0)

    base = dict(name=name, symbol=symbol, price=price, bar_time=bar_time,
                upper=ch_up, lower=ch_lo, atr=atr_val,
                exit_upper=exit_up, exit_lower=exit_lo,
                signal='', signal_type='none', detail='',
                suggested_qty=0, suggested_stop=0, risk_usdt=0)

    pos = positions.get(symbol)

    # ── 有持仓 → 检查出场 ──
    if pos:
        d = pos['direction']
        if d == 'long':
            if price <= pos['stop_loss']:
                base.update(signal='⚠ 多头止损', signal_type='exit_stop',
                    detail=f"价格 {fmt_price(price)} ≤ 止损 {fmt_price(pos['stop_loss'])}")
            elif price < exit_lo:
                base.update(signal='◀ 多头出场', signal_type='exit_channel',
                    detail=f"价格 {fmt_price(price)} < {LONG_EXIT}周期低点 {fmt_price(exit_lo)}")
            else:
                pnl = pos['quantity'] * (price - pos['entry_price'])
                pct = (price / pos['entry_price'] - 1) * 100
                exit_dist = (price - exit_lo) / price * 100
                base.update(signal=f"● 持多 {pct:+.1f}%", signal_type='holding_long',
                    detail=f"入场 {fmt_price(pos['entry_price'])}  "
                           f"止损 {fmt_price(pos['stop_loss'])}  "
                           f"出场线 {fmt_price(exit_lo)} (距 {exit_dist:.1f}%)  "
                           f"浮盈 {pnl:+,.0f} USDT")
        else:  # short
            if price >= pos['stop_loss']:
                base.update(signal='⚠ 空头止损', signal_type='exit_stop',
                    detail=f"价格 {fmt_price(price)} ≥ 止损 {fmt_price(pos['stop_loss'])}")
            elif price > exit_up:
                base.update(signal='◀ 空头出场', signal_type='exit_channel',
                    detail=f"价格 {fmt_price(price)} > {SHORT_EXIT}周期高点 {fmt_price(exit_up)}")
            else:
                pnl = pos['quantity'] * (pos['entry_price'] - price)
                pct = (pos['entry_price'] / price - 1) * 100
                exit_dist = (exit_up - price) / price * 100
                base.update(signal=f"● 持空 {pct:+.1f}%", signal_type='holding_short',
                    detail=f"入场 {fmt_price(pos['entry_price'])}  "
                           f"止损 {fmt_price(pos['stop_loss'])}  "
                           f"出场线 {fmt_price(exit_up)} (距 {exit_dist:.1f}%)  "
                           f"浮盈 {pnl:+,.0f} USDT")
        return base

    # ── 无持仓 → 检查入场 ──
    if price > ch_up:
        qty  = (CAPITAL * RISK_PCT) / atr_val
        stop = price - LONG_STOP * atr_val
        base.update(signal='★ 做多信号', signal_type='entry_long',
            suggested_qty=qty, suggested_stop=stop,
            risk_usdt=CAPITAL * RISK_PCT,
            detail=f"价格 {fmt_price(price)} > {LONG_ENTRY}周期高点 {fmt_price(ch_up)}")

    elif price < ch_lo:
        qty  = (CAPITAL * RISK_PCT) / atr_val
        stop = price + SHORT_STOP * atr_val
        base.update(signal='★ 做空信号', signal_type='entry_short',
            suggested_qty=qty, suggested_stop=stop,
            risk_usdt=CAPITAL * RISK_PCT,
            detail=f"价格 {fmt_price(price)} < {SHORT_ENTRY}周期低点 {fmt_price(ch_lo)}")

    else:
        dist_up = (ch_up - price) / price * 100
        dist_lo = (price - ch_lo) / price * 100
        if dist_up < NEAR_PCT:
            base.update(signal=f'△ 接近上轨 {dist_up:.1f}%', signal_type='near_upper',
                detail=f"距多头入场 {fmt_price(ch_up)} 差 {dist_up:.1f}%")
        elif dist_lo < NEAR_PCT:
            base.update(signal=f'▽ 接近下轨 {dist_lo:.1f}%', signal_type='near_lower',
                detail=f"距空头入场 {fmt_price(ch_lo)} 差 {dist_lo:.1f}%")
        else:
            base.update(signal='— 观望', signal_type='neutral',
                detail=f"距上轨 {dist_up:.1f}% / 距下轨 {dist_lo:.1f}%")

    return base


# ════════════════════════════════════════════════════════════════
#  输出
# ════════════════════════════════════════════════════════════════

def print_results(signals, scan_time):
    print()
    print('╔═══════════════════════════════════════════════════════════════════════╗')
    print(f'║  海龟信号扫描  |  {TIMEFRAME}  |  {scan_time}')
    print(f'║  多头 {LONG_ENTRY}入/{LONG_EXIT}出/{LONG_STOP:.0f}N止损'
          f'  空头 {SHORT_ENTRY}入/{SHORT_EXIT}出/{SHORT_STOP:.0f}N止损'
          f'  通道出场  风险 {RISK_PCT*100:.1f}%  资金 {CAPITAL:,} USDT')
    print('╚═══════════════════════════════════════════════════════════════════════╝')

    # ── 总览表 ──
    print(f'\n  {"品种":>6s}  {"价格":>10s}  {"入场上轨":>10s}  '
          f'{"入场下轨":>10s}  {"多出场":>10s}  {"空出场":>10s}  '
          f'{"ATR":>9s}  {"信号":<18s}')
    print('  ' + '─' * 92)

    for s in signals:
        if s.get('upper', 0) == 0:
            print(f'  {s["name"]:>6s}  {fmt_price(s["price"]):>10s}  '
                  f'{"—":>10s}  {"—":>10s}  {"—":>10s}  {"—":>10s}  '
                  f'{"—":>9s}  {s["signal"]}')
            continue
        exit_lo_str = fmt_price(s.get('exit_lower', 0)) if s.get('exit_lower') else '—'
        exit_up_str = fmt_price(s.get('exit_upper', 0)) if s.get('exit_upper') else '—'
        print(f'  {s["name"]:>6s}  {fmt_price(s["price"]):>10s}  '
              f'{fmt_price(s["upper"]):>10s}  {fmt_price(s["lower"]):>10s}  '
              f'{exit_lo_str:>10s}  {exit_up_str:>10s}  '
              f'{fmt_price(s["atr"]):>9s}  {s["signal"]}')

    # ── 分类汇总 ──
    entries  = [s for s in signals if s['signal_type'].startswith('entry_')]
    exits    = [s for s in signals if s['signal_type'].startswith('exit_')]
    holdings = [s for s in signals if s['signal_type'].startswith('holding_')]
    nears    = [s for s in signals if s['signal_type'].startswith('near_')]

    if entries:
        print(f'\n  {"★ 入场信号 ":═<38s}')
        for s in entries:
            dire_cn = '做多' if 'long' in s['signal_type'] else '做空'
            stop_n  = LONG_STOP if 'long' in s['signal_type'] else SHORT_STOP
            val     = s['suggested_qty'] * s['price']
            # 通道出场线
            if 'long' in s['signal_type']:
                exit_line = s.get('exit_lower', 0)
                exit_label = f'{LONG_EXIT}周期低点'
            else:
                exit_line = s.get('exit_upper', 0)
                exit_label = f'{SHORT_EXIT}周期高点'
            exit_dist = abs(s['price'] - exit_line) / s['price'] * 100 if s['price'] else 0
            print(f'  {s["name"]} — {dire_cn}')
            print(f'    {s["detail"]}')
            print(f'    建议仓位: {s["suggested_qty"]:.4f} {s["name"]}'
                  f'  (≈ {val:,.0f} USDT)')
            print(f'    止损价:   {fmt_price(s["suggested_stop"])}'
                  f'  ({stop_n:.0f}×ATR = {fmt_price(s["atr"] * stop_n)})')
            print(f'    出场线:   {fmt_price(exit_line)}'
                  f'  ({exit_label}, 距 {exit_dist:.1f}%)')
            print(f'    风险金额: {s["risk_usdt"]:.0f} USDT ({RISK_PCT*100:.1f}%)')
            print(f'    → 确认开仓后执行: python turtle_signal.py '
                  f'--record {s["name"]} {dire_cn[1:]} {s["price"]:.2f}')

    if exits:
        print(f'\n  {"⚠ 出场信号 ":═<38s}')
        for s in exits:
            print(f'  {s["name"]}: {s["signal"]}')
            print(f'    {s["detail"]}')
            print(f'    → 已平仓后执行: python turtle_signal.py '
                  f'--close {s["name"]} <平仓价>')

    if holdings:
        print(f'\n  {"● 当前持仓 ":═<38s}')
        for s in holdings:
            print(f'  {s["name"]}: {s["signal"]}')
            print(f'    {s["detail"]}')

    if nears:
        print(f'\n  {"△ 接近触发 ":─<38s}')
        for s in nears:
            print(f'    {s["name"]}: {s["detail"]}')

    n_total = len(signals)
    n_other = n_total - len(entries) - len(exits) - len(holdings) - len(nears)
    print(f'\n  汇总: 入场 {len(entries)} | 出场 {len(exits)} |'
          f' 持仓 {len(holdings)} | 接近 {len(nears)} | 观望 {n_other}')
    print()


def print_positions():
    positions = load_positions()

    # ── OKX 实盘持仓 ──
    okx_pos = []
    balance = None
    if has_auth_config():
        print('\n  连接 OKX 账户 ...', end='', flush=True)
        try:
            auth_ex = create_exchange(EXCHANGE_ID, need_auth=True)
            okx_pos = fetch_account_positions(auth_ex)
            balance = fetch_account_balance(auth_ex)
            print(' ✓')
        except Exception as e:
            print(f' ✗ {e}')

    if okx_pos:
        print(f'\n  {"═" * 60}')
        print('  OKX 实盘持仓')
        print(f'  {"═" * 60}')
        for p in okx_pos:
            print(f'\n  {p["name"]}  {p["side"].upper()}  ×{p["leverage"]:.0f}')
            print(f'    入场价:     {fmt_price(p["entry_price"]):>12s}    '
                  f'标记价: {fmt_price(p["mark_price"]):>12s}')
            qty = p['contracts'] * p['contract_size']
            print(f'    持仓量:     {qty:.4f} {p["name"]}'
                  f'  ({p["contracts"]:.0f} 张)')
            print(f'    未实现盈亏: {p["unrealized_pnl"]:>+12,.2f} USDT'
                  f'  ({p["percentage"]:+.2f}%)')
            print(f'    强平价:     {fmt_price(p["liq_price"]):>12s}    '
                  f'保证金: {p["margin"]:,.2f} USDT')
        if balance and balance['total'] > 0:
            print(f'\n  ── 账户余额 ──')
            print(f'    总额: {balance["total"]:,.2f} USDT'
                  f'   可用: {balance["free"]:,.2f}'
                  f'   已用: {balance["used"]:,.2f}')
    elif has_auth_config():
        print('\n  OKX 实盘: 当前无持仓')

    # ── 本地跟踪记录 ──
    if positions:
        print(f'\n  {"═" * 60}')
        print('  本地跟踪持仓')
        print(f'  {"═" * 60}')
        for sym, p in positions.items():
            name = short_name(sym)
            print(f'\n  {name}  {p["direction"].upper()}')
            print(f'    入场价: {fmt_price(p["entry_price"]):>12s}    '
                  f'止损价: {fmt_price(p["stop_loss"]):>12s}')
            print(f'    ATR:    {fmt_price(p["atr"]):>12s}    '
                  f'出场:   通道({LONG_EXIT}/{SHORT_EXIT}周期)')
            print(f'    数  量: {p["quantity"]:.4f}')
            print(f'    入场时间: {p["entry_time"][:19]}')

    if not okx_pos and not positions:
        print('\n  当前无持仓记录。')
        print('  用 --record <品种> <long|short> <价格> 记录开仓')
        if not has_auth_config():
            print('  或在 turtle/ 下创建 config.json 连接 OKX 账户查看实盘')

    print()


# ════════════════════════════════════════════════════════════════
#  主程序
# ════════════════════════════════════════════════════════════════

def run_scan():
    """执行一次扫描, 返回信号列表."""
    global SYMBOLS
    if SYMBOLS is None:
        print('  [ 获取品种 ]')
        SYMBOLS = fetch_top_symbols(exchange_id=EXCHANGE_ID, top_n=TOP_N)

    bars_needed = max(LONG_ENTRY, SHORT_ENTRY, ATR_PERIOD) + 10
    start_date  = calc_start_date(TIMEFRAME, bars_needed)

    # ── 获取持仓: 本地 + OKX 实盘 ──
    positions = load_positions()
    okx_real  = {}

    if has_auth_config():
        print('  [ 获取 OKX 实盘持仓 ]')
        try:
            auth_ex  = create_exchange(EXCHANGE_ID, need_auth=True)
            real_pos = fetch_account_positions(auth_ex)
            for p in real_pos:
                okx_real[p['symbol']] = p
                # 实盘品种加入扫描列表
                if p['symbol'] not in SYMBOLS:
                    SYMBOLS.append(p['symbol'])
            if real_pos:
                print(f'    实盘: {", ".join(p["name"] for p in real_pos)}')
            else:
                print('    实盘: 无持仓')
        except Exception as e:
            print(f'    ✗ 获取实盘失败: {e}')

    print(f'  拉取最新数据 ({len(SYMBOLS)} 品种, {TIMEFRAME}) ...')
    exchange = create_exchange(EXCHANGE_ID)
    signals  = []

    for sym in SYMBOLS:
        name = short_name(sym)
        try:
            df = fetch_ohlcv(exchange, sym, TIMEFRAME, start_date)
            if df is None or len(df) < bars_needed:
                signals.append(dict(name=name, symbol=sym, price=0,
                                    signal='数据不足', signal_type='none',
                                    upper=0, lower=0, atr=0, bar_time=None))
                continue

            # 实盘持仓 → 合并为本地 positions 格式 (本地记录优先)
            if sym in okx_real and sym not in positions:
                rp = okx_real[sym]
                atr_val = compute_atr(df['high'], df['low'],
                                      df['close'], ATR_PERIOD).iloc[-1]
                dire = rp['side']
                if dire == 'long':
                    stop = rp['entry_price'] - LONG_STOP * atr_val
                else:
                    stop = rp['entry_price'] + SHORT_STOP * atr_val
                positions[sym] = dict(
                    direction=dire,
                    entry_price=rp['entry_price'],
                    stop_loss=stop,
                    quantity=rp['contracts'] * rp['contract_size'],
                    atr=atr_val,
                    source='okx',
                )

            signals.append(scan_symbol(df, sym, positions))
            print(f'    {name} ✓', end='  ', flush=True)
        except Exception as e:
            print(f'    {name} ✗ {e}')
            signals.append(dict(name=name, symbol=sym, price=0,
                                signal='获取失败', signal_type='none',
                                upper=0, lower=0, atr=0, bar_time=None))
    print()

    scan_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    print_results(signals, scan_time)
    return signals


def main():
    parser = argparse.ArgumentParser(description='海龟信号扫描')
    parser.add_argument('--watch',   type=int,   metavar='SEC',
                        help='自动刷新间隔 (秒), 如 300')
    parser.add_argument('--status',  action='store_true',
                        help='查看持仓记录')
    parser.add_argument('--record',  nargs=3,
                        metavar=('SYM', 'DIR', 'PRICE'),
                        help='记录开仓: --record ETH long 3420')
    parser.add_argument('--close',   nargs=2,
                        metavar=('SYM', 'PRICE'),
                        help='记录平仓: --close ETH 3500')
    parser.add_argument('--symbols', nargs='+', metavar='S',
                        help='指定品种, 如 BTC ETH SOL')
    parser.add_argument('--tf',      type=str,  metavar='TF',
                        help='K线周期, 如 15m 1h 4h 1d')
    args = parser.parse_args()

    global SYMBOLS, TIMEFRAME
    if args.symbols:
        SYMBOLS = [f'{s.upper()}/USDT:USDT' if '/' not in s else s
                   for s in args.symbols]
    if args.tf:
        TIMEFRAME = args.tf

    # ── 查看持仓 ──
    if args.status:
        print_positions()
        return

    # ── 记录开仓 ──
    if args.record:
        sym_in, dire_in, price_in = args.record
        symbol = f'{sym_in.upper()}/USDT:USDT' if '/' not in sym_in else sym_in
        price  = float(price_in)
        dire   = 'long' if dire_in.lower() in ('long', '多', 'l') else 'short'

        # 拉当前 ATR
        exchange = create_exchange(EXCHANGE_ID)
        sd = calc_start_date(TIMEFRAME, ATR_PERIOD + 5)
        df = fetch_ohlcv(exchange, symbol, TIMEFRAME, sd)
        atr_val = compute_atr(df['high'], df['low'], df['close'], ATR_PERIOD).iloc[-1]

        if dire == 'long':
            stop = price - LONG_STOP * atr_val
        else:
            stop = price + SHORT_STOP * atr_val
        qty = (CAPITAL * RISK_PCT) / atr_val

        add_position(symbol, dire, price, stop, qty, atr_val)
        name = short_name(symbol)
        print(f'\n  ✓ 已记录 {name} {dire.upper()}')
        print(f'    入场: {fmt_price(price)}   止损: {fmt_price(stop)}')
        print(f'    数量: {qty:.4f}  ATR: {fmt_price(atr_val)}'
              f'  风险: {CAPITAL * RISK_PCT:.0f} USDT'
              f'  出场: {LONG_EXIT if dire=="long" else SHORT_EXIT}周期通道\n')
        return

    # ── 记录平仓 ──
    if args.close:
        sym_in, price_in = args.close
        symbol = f'{sym_in.upper()}/USDT:USDT' if '/' not in sym_in else sym_in
        price  = float(price_in)
        pos = remove_position(symbol, price)
        if pos:
            name = short_name(symbol)
            print(f'\n  ✓ 已平仓 {name} {pos["direction"].upper()}')
            print(f'    入场: {fmt_price(pos["entry_price"])}'
                  f'  出场: {fmt_price(price)}'
                  f'  盈亏: {pos.get("pnl", 0):+,.0f} USDT\n')
        else:
            print(f'\n  ✗ 未找到 {sym_in.upper()} 的持仓记录\n')
        return

    # ── 扫描 ──
    if args.watch:
        print(f'  自动刷新模式: 每 {args.watch} 秒 (Ctrl+C 退出)\n')
        try:
            while True:
                run_scan()
                for remaining in range(args.watch, 0, -1):
                    print(f'\r  下次扫描: {remaining}s ...', end='', flush=True)
                    time.sleep(1)
                print('\r' + ' ' * 40 + '\r', end='')
        except KeyboardInterrupt:
            print('\n\n  已停止自动扫描\n')
    else:
        run_scan()


if __name__ == '__main__':
    main()