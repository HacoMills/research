#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  海龟交易系统 — Streamlit 仪表盘 + Telegram 通知

  功能:
    1. 信号总览表 (带颜色标记)
    2. OKX 实盘持仓 + 账户余额
    3. K 线图 + 唐奇安通道叠加
    4. 每 4 小时自动刷新
    5. 有入场/出场/止损信号时推送 Telegram

  使用:
    streamlit run turtle_dashboard.py

  Telegram 配置 (config.json):
    {
      "apiKey": "...",
      "secret": "...",
      "password": "...",
      "proxy": "http://127.0.0.1:29290",
      "telegram_token": "your_bot_token",
      "telegram_chat_id": "your_chat_id"
    }
═══════════════════════════════════════════════════════════════
"""

import sys, os, json, time, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

# ── 项目路径 ──
_project_root = str(next(p for p in Path(__file__).resolve().parents if (p / "data" / "paths.py").exists()))   # 往上找含 data/paths.py 的文件夹 = quant 根目录 (挪位置也不怕)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.fetch_data import (
    create_exchange, fetch_ohlcv, fetch_top_symbols,
    short_name, DEFAULT_SYMBOLS,
    fetch_account_positions, fetch_account_balance, has_auth_config,
    _find_config_json,
)

# ════════════════════════════════════════════════════════════════
#  参数 (与 turtle_signal.py 保持一致)
# ════════════════════════════════════════════════════════════════

LONG_ENTRY  = 60
LONG_EXIT   = 20
LONG_STOP   = 2.0
SHORT_ENTRY = 60
SHORT_EXIT  = 20
SHORT_STOP  = 2.0
ATR_PERIOD  = 20
RISK_PCT    = 0.01
CAPITAL     = 10000
TIMEFRAME   = '4h'
EXCHANGE_ID = 'okx'
TOP_N       = 20
NEAR_PCT    = 2.0

REFRESH_SECONDS = 4 * 3600   # 4 小时自动刷新

TF_MINUTES = {'1m':1, '5m':5, '15m':15, '30m':30, '1h':60,
              '2h':120, '4h':240, '6h':360, '8h':480, '12h':720,
              '1d':1440, '3d':4320, '1w':10080}

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


def calc_start_date(tf, bars_needed):
    mins = TF_MINUTES.get(tf, 240)
    delta = timedelta(minutes=mins * bars_needed * 1.5)
    return (datetime.now(timezone.utc) - delta).strftime('%Y-%m-%d')


def fmt_price(p):
    if p >= 1000:   return f'{p:,.0f}'
    if p >= 1:      return f'{p:,.2f}'
    if p >= 0.01:   return f'{p:,.4f}'
    return f'{p:,.6f}'


# ════════════════════════════════════════════════════════════════
#  Telegram 通知
# ════════════════════════════════════════════════════════════════

def send_telegram(message):
    """通过 Telegram Bot 发送消息."""
    cfg = _find_config_json()
    token   = cfg.get('telegram_token', '')
    chat_id = cfg.get('telegram_chat_id', '')
    if not token or not chat_id:
        return False

    import urllib.request
    import ssl

    url = f'https://api.telegram.org/bot{token}/sendMessage'
    data = json.dumps({
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML',
    }).encode('utf-8')

    req = urllib.request.Request(url, data=data, headers={
        'Content-Type': 'application/json',
    })

    proxy = cfg.get('proxy', '')
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({
            'http': proxy, 'https': proxy,
        }))
    handlers.append(urllib.request.HTTPSHandler(
        context=ssl.create_default_context()))
    opener = urllib.request.build_opener(*handlers)

    try:
        resp = opener.open(req, timeout=10)
        return resp.status == 200
    except Exception as e:
        st.warning(f'Telegram 发送失败: {e}')
        return False


def has_telegram_config():
    cfg = _find_config_json()
    return bool(cfg.get('telegram_token') and cfg.get('telegram_chat_id'))


def notify_signals(signals):
    """有重要信号时推送 Telegram."""
    important = [s for s in signals
                 if s['signal_type'].startswith(('entry_', 'exit_'))]
    if not important:
        return

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [f'🐢 <b>海龟信号提醒</b>  {now}\n']

    for s in important:
        emoji = '🟢' if 'entry_long' in s['signal_type'] else \
                '🔴' if 'entry_short' in s['signal_type'] else \
                '⚠️'
        lines.append(f'{emoji} <b>{s["name"]}</b>  {s["signal"]}')
        lines.append(f'    价格: {fmt_price(s["price"])}  ATR: {fmt_price(s["atr"])}')
        if s.get('suggested_stop'):
            lines.append(f'    止损: {fmt_price(s["suggested_stop"])}')
        if s.get('detail'):
            lines.append(f'    {s["detail"]}')
        lines.append('')

    send_telegram('\n'.join(lines))


# ════════════════════════════════════════════════════════════════
#  数据扫描 (返回结构化数据)
# ════════════════════════════════════════════════════════════════

def scan_symbol(df, symbol, positions):
    """扫描单个品种, 返回信号字典."""
    name = short_name(symbol)
    atr_s  = compute_atr(df['high'], df['low'], df['close'], ATR_PERIOD)
    up_l   = df['high'].rolling(LONG_ENTRY).max().shift(1)
    lo_l   = df['low'].rolling(LONG_EXIT).min().shift(1)
    lo_s   = df['low'].rolling(SHORT_ENTRY).min().shift(1)
    up_s   = df['high'].rolling(SHORT_EXIT).max().shift(1)

    price    = df['close'].iloc[-1]
    atr_val  = atr_s.iloc[-1]
    ch_up    = up_l.iloc[-1]
    ch_lo    = lo_s.iloc[-1]
    exit_lo  = lo_l.iloc[-1]
    exit_up  = up_s.iloc[-1]

    if pd.isna(atr_val) or pd.isna(ch_up):
        return dict(name=name, symbol=symbol, price=price,
                    signal='数据不足', signal_type='none',
                    upper=0, lower=0, atr=0,
                    exit_lower=0, exit_upper=0,
                    df=df, atr_series=atr_s,
                    up_l=up_l, lo_l=lo_l, lo_s=lo_s, up_s=up_s)

    base = dict(name=name, symbol=symbol, price=price,
                upper=ch_up, lower=ch_lo, atr=atr_val,
                exit_upper=exit_up, exit_lower=exit_lo,
                signal='', signal_type='none', detail='',
                suggested_qty=0, suggested_stop=0, risk_usdt=0,
                df=df, atr_series=atr_s,
                up_l=up_l, lo_l=lo_l, lo_s=lo_s, up_s=up_s)

    pos = positions.get(symbol)

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
        else:
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


@st.cache_data(ttl=REFRESH_SECONDS)
def run_full_scan():
    """执行完整扫描, 返回 (signals, positions_info, balance, scan_time)."""
    symbols = fetch_top_symbols(exchange_id=EXCHANGE_ID, top_n=TOP_N)

    bars_needed = max(LONG_ENTRY, SHORT_ENTRY, ATR_PERIOD) + 10
    start_date  = calc_start_date(TIMEFRAME, bars_needed)

    # 持仓数据
    positions = {}
    positions_file = Path(__file__).resolve().parent / 'turtle_positions.json'
    if positions_file.exists():
        try:
            positions = json.load(open(positions_file))
        except Exception:
            pass

    okx_real  = {}
    okx_pos   = []
    balance   = None

    if has_auth_config():
        try:
            auth_ex  = create_exchange(EXCHANGE_ID, need_auth=True)
            okx_pos  = fetch_account_positions(auth_ex)
            balance  = fetch_account_balance(auth_ex)
            for p in okx_pos:
                okx_real[p['symbol']] = p
                if p['symbol'] not in symbols:
                    symbols.append(p['symbol'])
        except Exception:
            pass

    exchange = create_exchange(EXCHANGE_ID)
    signals  = []

    progress = st.progress(0, text='拉取数据中...')
    for i, sym in enumerate(symbols):
        progress.progress((i + 1) / len(symbols),
                          text=f'扫描 {short_name(sym)} ({i+1}/{len(symbols)})')
        try:
            df = fetch_ohlcv(exchange, sym, TIMEFRAME, start_date)
            if df is None or len(df) < bars_needed:
                signals.append(dict(name=short_name(sym), symbol=sym, price=0,
                                    signal='数据不足', signal_type='none',
                                    upper=0, lower=0, atr=0,
                                    exit_lower=0, exit_upper=0,
                                    df=pd.DataFrame(), atr_series=pd.Series(),
                                    up_l=pd.Series(), lo_l=pd.Series(),
                                    lo_s=pd.Series(), up_s=pd.Series()))
                continue

            if sym in okx_real and sym not in positions:
                rp = okx_real[sym]
                atr_val = compute_atr(df['high'], df['low'],
                                      df['close'], ATR_PERIOD).iloc[-1]
                dire = rp['side']
                stop = (rp['entry_price'] - LONG_STOP * atr_val if dire == 'long'
                        else rp['entry_price'] + SHORT_STOP * atr_val)
                positions[sym] = dict(
                    direction=dire, entry_price=rp['entry_price'],
                    stop_loss=stop,
                    quantity=rp['contracts'] * rp['contract_size'],
                    atr=atr_val, source='okx',
                )

            signals.append(scan_symbol(df, sym, positions))
        except Exception:
            signals.append(dict(name=short_name(sym), symbol=sym, price=0,
                                signal='获取失败', signal_type='none',
                                upper=0, lower=0, atr=0,
                                exit_lower=0, exit_upper=0,
                                df=pd.DataFrame(), atr_series=pd.Series(),
                                up_l=pd.Series(), lo_l=pd.Series(),
                                lo_s=pd.Series(), up_s=pd.Series()))

    progress.empty()
    scan_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    # 推送 Telegram
    if has_telegram_config():
        notify_signals(signals)

    return signals, okx_pos, balance, scan_time


# ════════════════════════════════════════════════════════════════
#  K 线图 + 通道叠加
# ════════════════════════════════════════════════════════════════

def plot_kline_chart(signal_data):
    """用 plotly 画 K 线 + 唐奇安通道."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = signal_data.get('df')
    if df is None or df.empty:
        return None

    up_l = signal_data.get('up_l', pd.Series())
    lo_s = signal_data.get('lo_s', pd.Series())
    lo_l = signal_data.get('lo_l', pd.Series())
    up_s = signal_data.get('up_s', pd.Series())

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)

    # K 线
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='K线',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
    ), row=1, col=1)

    # 入场通道
    if not up_l.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=up_l, name=f'入场上轨({LONG_ENTRY})',
            line=dict(color='#2196F3', width=1.5),
        ), row=1, col=1)
    if not lo_s.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=lo_s, name=f'入场下轨({SHORT_ENTRY})',
            line=dict(color='#FF9800', width=1.5),
        ), row=1, col=1)

    # 出场通道
    if not lo_l.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=lo_l, name=f'多出场({LONG_EXIT})',
            line=dict(color='#2196F3', width=1, dash='dash'),
        ), row=1, col=1)
    if not up_s.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=up_s, name=f'空出场({SHORT_EXIT})',
            line=dict(color='#FF9800', width=1, dash='dash'),
        ), row=1, col=1)

    # 成交量
    if 'volume' in df.columns:
        colors = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df.index, y=df['volume'], name='成交量',
            marker_color=colors, opacity=0.5,
        ), row=2, col=1)

    name = signal_data['name']
    fig.update_layout(
        title=f'{name}/USDT  {TIMEFRAME}  唐奇安通道',
        height=500,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        margin=dict(l=50, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='成交量', row=2, col=1)

    return fig


# ════════════════════════════════════════════════════════════════
#  Streamlit 页面
# ════════════════════════════════════════════════════════════════

def signal_color(signal_type):
    """返回信号对应的颜色."""
    if signal_type.startswith('entry_'):   return '🟢'
    if signal_type.startswith('exit_'):    return '🔴'
    if signal_type.startswith('holding_'): return '🔵'
    if signal_type.startswith('near_'):    return '🟡'
    return '⚪'


def main():
    st.set_page_config(
        page_title='海龟信号仪表盘',
        page_icon='🐢',
        layout='wide',
    )

    st.title('🐢 海龟交易信号仪表盘')

    # ── 侧边栏 ──
    with st.sidebar:
        st.header('⚙️ 设置')
        st.markdown(f"""
        - **周期**: {TIMEFRAME}
        - **入场通道**: {LONG_ENTRY} / {SHORT_ENTRY}
        - **出场通道**: {LONG_EXIT} / {SHORT_EXIT}
        - **止损**: {LONG_STOP:.0f}N / {SHORT_STOP:.0f}N
        - **风险**: {RISK_PCT*100:.1f}%
        - **资金**: {CAPITAL:,} USDT
        - **品种数**: Top {TOP_N}
        """)

        st.divider()
        tg_status = '✅ 已配置' if has_telegram_config() else '❌ 未配置'
        st.markdown(f'**Telegram 通知**: {tg_status}')
        if not has_telegram_config():
            st.caption('在 config.json 添加 telegram_token 和 telegram_chat_id')

        st.divider()
        if st.button('🔄 立即刷新', use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.caption(f'自动刷新间隔: {REFRESH_SECONDS // 3600} 小时')

    # ── 数据扫描 ──
    signals, okx_pos, balance, scan_time = run_full_scan()

    # ── 顶部统计 ──
    entries  = [s for s in signals if s['signal_type'].startswith('entry_')]
    exits    = [s for s in signals if s['signal_type'].startswith('exit_')]
    holdings = [s for s in signals if s['signal_type'].startswith('holding_')]
    nears    = [s for s in signals if s['signal_type'].startswith('near_')]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('⏱ 扫描时间', scan_time)
    col2.metric('🟢 入场信号', len(entries))
    col3.metric('🔴 出场信号', len(exits))
    col4.metric('🔵 持仓', len(holdings))
    col5.metric('🟡 接近触发', len(nears))

    # ── 账户余额 ──
    if balance and balance['total'] > 0:
        st.divider()
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric('💰 账户总额', f"{balance['total']:,.2f} USDT")
        bc2.metric('可用', f"{balance['free']:,.2f} USDT")
        bc3.metric('已用', f"{balance['used']:,.2f} USDT")

    # ── 重要信号 (入场 / 出场) ──
    if entries or exits:
        st.divider()
        st.subheader('⚡ 重要信号')
        for s in entries + exits:
            emoji = signal_color(s['signal_type'])
            with st.expander(f"{emoji} {s['name']}  {s['signal']}", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric('价格', fmt_price(s['price']))
                c2.metric('ATR', fmt_price(s['atr']))
                if s.get('suggested_stop'):
                    c3.metric('建议止损', fmt_price(s['suggested_stop']))
                st.caption(s['detail'])
                if s.get('suggested_qty'):
                    val = s['suggested_qty'] * s['price']
                    st.info(f"建议仓位: {s['suggested_qty']:.4f} {s['name']}"
                            f"  (≈ {val:,.0f} USDT)  风险: {s['risk_usdt']:.0f} USDT")

    # ── OKX 持仓 ──
    if okx_pos:
        st.divider()
        st.subheader('📊 OKX 实盘持仓')
        for p in okx_pos:
            with st.expander(
                f"{'🟢' if p['side']=='long' else '🔴'} "
                f"{p['name']}  {p['side'].upper()}  ×{p['leverage']:.0f}  "
                f"{p['unrealized_pnl']:+,.2f} USDT ({p['percentage']:+.2f}%)",
                expanded=True
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric('入场价', fmt_price(p['entry_price']))
                c2.metric('标记价', fmt_price(p['mark_price']))
                qty = p['contracts'] * p['contract_size']
                c3.metric('持仓量', f"{qty:.4f}")
                c4.metric('保证金', f"{p['margin']:,.2f} USDT")

    # ── 信号总览表 ──
    st.divider()
    st.subheader('📋 信号总览')

    table_data = []
    for s in signals:
        row = {
            '状态': signal_color(s['signal_type']),
            '品种': s['name'],
            '价格': fmt_price(s['price']) if s['price'] > 0 else '—',
            '入场上轨': fmt_price(s['upper']) if s.get('upper', 0) > 0 else '—',
            '入场下轨': fmt_price(s['lower']) if s.get('lower', 0) > 0 else '—',
            '多出场': fmt_price(s.get('exit_lower', 0)) if s.get('exit_lower', 0) > 0 else '—',
            '空出场': fmt_price(s.get('exit_upper', 0)) if s.get('exit_upper', 0) > 0 else '—',
            'ATR': fmt_price(s['atr']) if s.get('atr', 0) > 0 else '—',
            '信号': s['signal'],
        }
        table_data.append(row)

    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True,
                 height=min(40 * len(table_data) + 38, 800))

    # ── K 线图 ──
    st.divider()
    st.subheader('📈 K 线图')

    valid_signals = [s for s in signals
                     if isinstance(s.get('df'), pd.DataFrame)
                     and not s['df'].empty]
    symbol_names  = [s['name'] for s in valid_signals]

    if symbol_names:
        # 默认显示有信号的品种
        default_idx = 0
        for i, s in enumerate(valid_signals):
            if s['signal_type'] not in ('neutral', 'none'):
                default_idx = i
                break

        selected = st.selectbox('选择品种', symbol_names, index=default_idx)
        sel_data = next(s for s in valid_signals if s['name'] == selected)

        fig = plot_kline_chart(sel_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # 显示详情
        if sel_data.get('detail'):
            st.info(f"**{sel_data['signal']}** — {sel_data['detail']}")

    # ── 自动刷新 ──
    time.sleep(REFRESH_SECONDS)
    st.cache_data.clear()
    st.rerun()


if __name__ == '__main__':
    main()
