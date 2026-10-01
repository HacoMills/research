#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  海龟交易系统 — Streamlit 仪表盘

  侧边栏可以选: 币种、K线周期、入场/出场通道天数、止损倍数、资金和风险、是否启用 GARCH 过滤.
  通道和 ATR 按"天"设置, 自动换算成 K线根数, 与回测框架 (python -m backtest) 一致:
    4h 周期的 60 天 = 360 根K线; 1h 周期的 60 天 = 1440 根.
    1. 信号总览 (入场/出场 + GARCH 过滤状态)
    2. GARCH 条件波动率曲线 + 波动率扩张/收缩标记
    3. K线图 + 唐奇安通道 + ATR
    4. 仓位计算 (unit_size, 止损价, 风险金额)
    5. Telegram 通知 (可选)
    6. 自动刷新 (侧边栏选 30 秒 / 1 分钟 / 5 分钟): 只拉最新几根K线, 不重新下载历史

  使用:
    streamlit run turtle_dashboard_garch.py

  依赖:
    pip install streamlit plotly arch ccxt pandas numpy
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
    create_exchange, fetch_ohlcv, short_name,
    fetch_account_positions, fetch_account_balance, has_auth_config,
    _find_config_json,
)

# ════════════════════════════════════════════════════════════════
#  参数
# ════════════════════════════════════════════════════════════════

# 侧边栏里可选的币 (也可以在侧边栏手动输入其他币)
COIN_CHOICES = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT', 'LTC',
                'BCH', 'TRX', 'UNI', 'ATOM', 'NEAR', 'SUI', 'BNB', 'FIL', 'ETC', 'OP', 'ARB', 'ZEC']
DEFAULT_COINS = ['BTC', 'ETH']

# 周期 → 每天K线根数 (加密货币 24 小时交易)
TF_BARS_PER_DAY = {'15m': 96, '30m': 48, '1h': 24, '4h': 6, '1d': 1}

# 默认参数 (单位: 天; 与回测框架的 donchian:60,20 一致)
DEFAULT_TF          = '4h'
DEFAULT_ENTRY_DAYS  = 60
DEFAULT_EXIT_DAYS   = 20
DEFAULT_STOP        = 2.0
DEFAULT_CAPITAL     = 10000
DEFAULT_RISK_PCT    = 0.01
ATR_DAYS            = 20      # ATR 周期 (天)
GARCH_DAYS          = 20      # GARCH 波动率均值窗口 (天)
EXCHANGE_ID = 'okx'
NEAR_PCT    = 2.0


def make_cfg(symbols, tf, entry_days, exit_days, stop, capital, risk_pct, use_garch):
    """侧边栏参数 → 运行配置 (天数换算成K线根数)."""
    bpd = TF_BARS_PER_DAY[tf]
    return dict(
        symbols=list(symbols), tf=tf, use_garch=use_garch,
        entry_days=entry_days, exit_days=exit_days,
        entry=max(int(entry_days * bpd), 2), exit=max(int(exit_days * bpd), 2),
        atr=max(int(ATR_DAYS * bpd), 14), garch_lookback=max(int(GARCH_DAYS * bpd), 20),
        stop=stop, capital=capital, risk_pct=risk_pct,
    )


CFG = make_cfg(['BTC/USDT:USDT', 'ETH/USDT:USDT'], DEFAULT_TF, DEFAULT_ENTRY_DAYS,
               DEFAULT_EXIT_DAYS, DEFAULT_STOP, DEFAULT_CAPITAL, DEFAULT_RISK_PCT, False)

REFRESH_CHOICES = {'30 秒': 30, '1 分钟': 60, '5 分钟': 300, '关闭': 0}
DEFAULT_REFRESH = '1 分钟'
CHART_BARS = 400              # K线图只画最近多少根 (画太多会拖慢页面)

TF_MINUTES = {'1m':1, '5m':5, '15m':15, '30m':30, '1h':60,
              '2h':120, '4h':240, '6h':360, '8h':480, '12h':720,
              '1d':1440, '3d':4320, '1w':10080}


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════

def compute_atr(high, low, close, period):
    """指数加权 ATR (与回测一致)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


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
#  GARCH(1,1) 波动率过滤器
# ════════════════════════════════════════════════════════════════

def compute_garch_volatility(df, lookback=120):
    """
    计算 GARCH(1,1) 条件波动率和过滤信号.

    返回 dict:
      - cond_vol: 条件波动率 Series
      - vol_ma: 波动率滚动均值 Series
      - expanding: 布尔 Series (True = 波动率扩张, 允许入场)
      - alpha: GARCH α 参数
      - beta: GARCH β 参数
      - omega: GARCH ω 参数
      - pct_on: 允许入场时间占比
    失败返回 None
    """
    try:
        from arch import arch_model
    except ImportError:
        st.error("需要安装 arch 库: pip install arch --break-system-packages")
        return None

    close = df['close'].dropna()
    if len(close) < 200:
        st.warning("数据不足 200 根 K 线, 无法拟合 GARCH")
        return None

    # 对数收益率 (×100, arch 库习惯)
    log_ret = (np.log(close / close.shift(1)) * 100).dropna()

    try:
        model = arch_model(log_ret, vol='Garch', p=1, q=1,
                           mean='Zero', rescale=False)
        res = model.fit(disp='off', show_warning=False)
    except Exception as e:
        st.warning(f"GARCH 拟合失败: {e}")
        return None

    # 条件波动率
    cond_vol = res.conditional_volatility

    # 滚动均值
    vol_ma = cond_vol.rolling(window=lookback).mean()

    # σ_t > σ̄ → 波动率扩张 → 允许入场
    expanding = cond_vol > vol_ma

    # 对齐回原始 df 的 index
    cond_vol_aligned = pd.Series(np.nan, index=df.index)
    cond_vol_aligned.loc[cond_vol.index] = cond_vol.values

    vol_ma_aligned = pd.Series(np.nan, index=df.index)
    vol_ma_aligned.loc[vol_ma.index] = vol_ma.values

    expanding_aligned = pd.Series(False, index=df.index)
    expanding_aligned.loc[expanding.index] = expanding.values

    alpha = res.params.get('alpha[1]', 0)
    beta  = res.params.get('beta[1]', 0)
    omega = res.params.get('omega', 0)
    pct_on = expanding.sum() / len(expanding) * 100

    return {
        'cond_vol': cond_vol_aligned,
        'vol_ma': vol_ma_aligned,
        'expanding': expanding_aligned,
        'alpha': alpha,
        'beta': beta,
        'omega': omega,
        'pct_on': pct_on,
    }


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

    import urllib.request, ssl

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


@st.cache_resource
def _notified():
    """已推送过的信号 {(币种, 周期): 信号类型}, 避免每次刷新重复推送."""
    return {}


def notify_signals(signals):
    """出现新的入场/出场信号时推送 Telegram (同一个信号只推一次)."""
    sent = _notified()
    important = []
    for s in signals:
        key = (s['symbol'], CFG['tf'])
        if s['signal_type'].startswith(('entry_', 'exit_')) and sent.get(key) != s['signal_type']:
            important.append(s)
        sent[key] = s['signal_type']
    if not important:
        return

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [f'🐢 <b>GARCH 海龟信号</b>  {now}\n']

    for s in important:
        emoji = '🟢' if 'entry_long' in s['signal_type'] else \
                '🔴' if 'entry_short' in s['signal_type'] else '⚠️'
        lines.append(f'{emoji} <b>{s["name"]}</b>  {s["signal"]}')
        lines.append(f'    价格: {fmt_price(s["price"])}  ATR: {fmt_price(s["atr"])}')
        garch_status = '✅ 允许' if s.get('garch_expanding') else '🚫 过滤'
        lines.append(f'    GARCH: {garch_status}  σ: {s.get("garch_vol", 0):.4f}')
        if s.get('suggested_stop'):
            lines.append(f'    止损: {fmt_price(s["suggested_stop"])}')
        if s.get('detail'):
            lines.append(f'    {s["detail"]}')
        lines.append('')

    send_telegram('\n'.join(lines))


# ════════════════════════════════════════════════════════════════
#  数据扫描
# ════════════════════════════════════════════════════════════════

def scan_symbol_garch(df, symbol, garch_info, positions):
    """扫描单个品种, 返回信号字典. CFG['use_garch'] 为 False 时突破直接算入场."""
    name = short_name(symbol)
    E, X, use_garch = CFG['entry'], CFG['exit'], CFG['use_garch']
    CAPITAL, RISK_PCT, STOP = CFG['capital'], CFG['risk_pct'], CFG['stop']
    atr_s  = compute_atr(df['high'], df['low'], df['close'], CFG['atr'])
    up_l   = df['high'].rolling(E).max().shift(1)
    lo_l   = df['low'].rolling(X).min().shift(1)
    lo_s   = df['low'].rolling(E).min().shift(1)
    up_s   = df['high'].rolling(X).max().shift(1)

    price    = df['close'].iloc[-1]
    atr_val  = atr_s.iloc[-1]
    ch_up    = up_l.iloc[-1]
    ch_lo    = lo_s.iloc[-1]
    exit_lo  = lo_l.iloc[-1]
    exit_up  = up_s.iloc[-1]

    # GARCH 当前值
    garch_expanding = False
    garch_vol = 0.0
    garch_vol_ma = 0.0
    if garch_info:
        garch_expanding = bool(garch_info['expanding'].iloc[-1])
        garch_vol = garch_info['cond_vol'].iloc[-1] if not pd.isna(garch_info['cond_vol'].iloc[-1]) else 0
        garch_vol_ma = garch_info['vol_ma'].iloc[-1] if not pd.isna(garch_info['vol_ma'].iloc[-1]) else 0

    if pd.isna(atr_val) or pd.isna(ch_up):
        return dict(name=name, symbol=symbol, price=price,
                    signal='数据不足', signal_type='none',
                    upper=0, lower=0, atr=0,
                    exit_lower=0, exit_upper=0,
                    garch_expanding=garch_expanding, garch_vol=garch_vol,
                    garch_vol_ma=garch_vol_ma,
                    df=df, atr_series=atr_s, garch_info=garch_info,
                    up_l=up_l, lo_l=lo_l, lo_s=lo_s, up_s=up_s)

    base = dict(name=name, symbol=symbol, price=price,
                upper=ch_up, lower=ch_lo, atr=atr_val,
                exit_upper=exit_up, exit_lower=exit_lo,
                signal='', signal_type='none', detail='',
                suggested_qty=0, suggested_stop=0, risk_usdt=0,
                garch_expanding=garch_expanding, garch_vol=garch_vol,
                garch_vol_ma=garch_vol_ma,
                df=df, atr_series=atr_s, garch_info=garch_info,
                up_l=up_l, lo_l=lo_l, lo_s=lo_s, up_s=up_s)

    pos = positions.get(symbol)

    # ── 持仓检查出场 ──
    if pos:
        d = pos['direction']
        if d == 'long':
            if price <= pos['stop_loss']:
                base.update(signal='⚠ 多头止损', signal_type='exit_stop',
                    detail=f"价格 {fmt_price(price)} ≤ 止损 {fmt_price(pos['stop_loss'])}")
            elif price < exit_lo:
                base.update(signal='◀ 多头出场', signal_type='exit_channel',
                    detail=f"价格 {fmt_price(price)} < {CFG['exit_days']}天低点 {fmt_price(exit_lo)}")
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
                    detail=f"价格 {fmt_price(price)} > {CFG['exit_days']}天高点 {fmt_price(exit_up)}")
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

    # ── 空仓: 检查入场信号 ──
    garch_ok = garch_expanding or not use_garch          # 不启用 GARCH 时不过滤
    garch_txt = (f" | GARCH 扩张 σ={garch_vol:.4f} > μ={garch_vol_ma:.4f}" if use_garch else "")
    if price > ch_up:
        qty  = (CAPITAL * RISK_PCT) / atr_val
        stop = price - STOP * atr_val
        brk = f"价格 {fmt_price(price)} > {CFG['entry_days']}天高点 {fmt_price(ch_up)}"
        if garch_ok:
            base.update(signal='★ 做多信号' + (' (GARCH ✅)' if use_garch else ''),
                signal_type='entry_long', suggested_qty=qty, suggested_stop=stop,
                risk_usdt=CAPITAL * RISK_PCT, detail=brk + garch_txt)
        else:
            base.update(signal='△ 突破但 GARCH 过滤 🚫', signal_type='near_upper',
                suggested_qty=qty, suggested_stop=stop, risk_usdt=CAPITAL * RISK_PCT,
                detail=brk + f" | 但 GARCH 收缩 σ={garch_vol:.4f} ≤ μ={garch_vol_ma:.4f} → 不入场")
    elif price < ch_lo:
        qty  = (CAPITAL * RISK_PCT) / atr_val
        stop = price + STOP * atr_val
        brk = f"价格 {fmt_price(price)} < {CFG['entry_days']}天低点 {fmt_price(ch_lo)}"
        if garch_ok:
            base.update(signal='★ 做空信号' + (' (GARCH ✅)' if use_garch else ''),
                signal_type='entry_short', suggested_qty=qty, suggested_stop=stop,
                risk_usdt=CAPITAL * RISK_PCT, detail=brk + garch_txt)
        else:
            base.update(signal='▽ 突破但 GARCH 过滤 🚫', signal_type='near_lower',
                suggested_qty=qty, suggested_stop=stop, risk_usdt=CAPITAL * RISK_PCT,
                detail=brk + f" | 但 GARCH 收缩 σ={garch_vol:.4f} ≤ μ={garch_vol_ma:.4f} → 不入场")
    else:
        dist_up = (ch_up - price) / price * 100
        dist_lo = (price - ch_lo) / price * 100
        garch_label = ('扩张 ✅' if garch_expanding else '收缩 🚫') if use_garch else '未启用'
        if dist_up < NEAR_PCT:
            base.update(signal=f'△ 接近上轨 {dist_up:.1f}%', signal_type='near_upper',
                detail=f"距多头入场 {fmt_price(ch_up)} 差 {dist_up:.1f}% | GARCH {garch_label}")
        elif dist_lo < NEAR_PCT:
            base.update(signal=f'▽ 接近下轨 {dist_lo:.1f}%', signal_type='near_lower',
                detail=f"距空头入场 {fmt_price(ch_lo)} 差 {dist_lo:.1f}% | GARCH {garch_label}")
        else:
            base.update(signal=f'— 观望 | GARCH {garch_label}', signal_type='neutral',
                detail=f"距上轨 {dist_up:.1f}% / 距下轨 {dist_lo:.1f}% | GARCH {garch_label}")

    return base


@st.cache_resource(show_spinner=False)
def get_exchange(auth=False):
    """交易所连接只建一次 (加载市场信息要几秒, 每次刷新都重建会很慢)."""
    ex = create_exchange(EXCHANGE_ID, need_auth=auth)
    try:
        ex.load_markets()
    except Exception:
        pass
    return ex


@st.cache_resource
def _bar_store():
    """内存里的K线缓存 {(币种, 周期): DataFrame}; 刷新时只拉最新几根."""
    return {}


def get_bars(symbol, tf, bars_needed):
    """第一次全量拉取, 之后只从最后一根K线开始拉 (覆盖未收盘的那根)."""
    store, key = _bar_store(), (symbol, tf)
    ex = get_exchange()
    old = store.get(key)
    if old is not None and len(old) >= bars_needed:
        since_ms = int(old.index[-1].timestamp() * 1000)
        new = fetch_ohlcv(ex, symbol, tf, since_ms=since_ms)
        df = old if new is None or new.empty else \
            pd.concat([old, new])[lambda d: ~d.index.duplicated(keep='last')].sort_index()
    else:
        df = fetch_ohlcv(ex, symbol, tf, calc_start_date(tf, bars_needed))
    if df is not None and not df.empty:
        df = df.iloc[-(bars_needed + 50):]
        store[key] = df
    return df


@st.cache_data(ttl=60, show_spinner=False)
def get_account():
    """OKX 持仓和余额 (需要 API Key), 1 分钟内不重复请求."""
    if not has_auth_config():
        return [], None
    try:
        ex = get_exchange(auth=True)
        return fetch_account_positions(ex), fetch_account_balance(ex)
    except Exception:
        return [], None


def _empty_signal(coin, sym, msg):
    return dict(name=coin, symbol=sym, price=0, signal=msg, signal_type='none',
                upper=0, lower=0, atr=0, exit_lower=0, exit_upper=0,
                garch_expanding=False, garch_vol=0, garch_vol_ma=0,
                df=pd.DataFrame(), atr_series=pd.Series(), garch_info=None,
                up_l=pd.Series(), lo_l=pd.Series(), lo_s=pd.Series(), up_s=pd.Series())


def run_full_scan():
    """按当前 CFG 扫描全部币种, 返回 (signals, okx持仓, 余额, 扫描时间)."""
    bars_needed = max(CFG['entry'], CFG['exit'], CFG['atr'], CFG['garch_lookback']) + 200
    stop = CFG['stop']

    positions = {}
    positions_file = Path(__file__).resolve().parent / 'turtle_positions.json'
    if positions_file.exists():
        try:
            positions = json.load(open(positions_file))
        except Exception:
            pass

    okx_pos, balance = get_account()
    okx_real = {p['symbol']: p for p in okx_pos}

    signals = []
    for sym in CFG['symbols']:
        coin = short_name(sym)
        try:
            df = get_bars(sym, CFG['tf'], bars_needed)
            if df is None or len(df) < bars_needed // 2:
                signals.append(_empty_signal(coin, sym, '数据不足 (可能是新上线的币)'))
                continue
            garch_info = compute_garch_volatility(df, lookback=CFG['garch_lookback']) if CFG['use_garch'] else None

            if sym in okx_real and sym not in positions:          # 同步 OKX 实盘持仓
                rp = okx_real[sym]
                atr_val = compute_atr(df['high'], df['low'], df['close'], CFG['atr']).iloc[-1]
                dire = rp['side']
                positions[sym] = dict(
                    direction=dire, entry_price=rp['entry_price'],
                    stop_loss=(rp['entry_price'] - stop * atr_val if dire == 'long'
                               else rp['entry_price'] + stop * atr_val),
                    quantity=rp['contracts'] * rp['contract_size'], atr=atr_val, source='okx')

            signals.append(scan_symbol_garch(df, sym, garch_info, positions))
        except Exception as e:
            signals.append(_empty_signal(coin, sym, f'获取失败: {e}'))

    scan_time = datetime.now().strftime('%H:%M:%S')
    if has_telegram_config():
        notify_signals(signals)
    return signals, okx_pos, balance, scan_time


# ════════════════════════════════════════════════════════════════
#  图表
# ════════════════════════════════════════════════════════════════

def plot_kline_garch(signal_data):
    """K线 + 唐奇安通道 + GARCH 波动率 + ATR 四合一图表."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = signal_data.get('df')
    if df is None or df.empty:
        return None
    df = df.iloc[-CHART_BARS:]                     # 只画最近一段, 页面更快

    up_l = signal_data.get('up_l', pd.Series()).reindex(df.index)
    lo_s = signal_data.get('lo_s', pd.Series()).reindex(df.index)
    lo_l = signal_data.get('lo_l', pd.Series()).reindex(df.index)
    up_s = signal_data.get('up_s', pd.Series()).reindex(df.index)
    atr_s = signal_data.get('atr_series', pd.Series()).reindex(df.index)
    garch = signal_data.get('garch_info')

    # 4 行子图: K线 | GARCH 波动率 | ATR | 成交量
    if garch:
        nrows, row_heights, titles = 4, [0.45, 0.2, 0.15, 0.2], ['', 'GARCH 条件波动率', 'ATR', '成交量']
    else:                                          # 未启用 GARCH: 不画波动率子图
        nrows, row_heights, titles = 3, [0.6, 0.18, 0.22], ['', 'ATR', '成交量']
    r_atr, r_vol = (3, 4) if garch else (2, 3)
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True,
                        row_heights=row_heights,
                        vertical_spacing=0.02,
                        subplot_titles=titles)

    # ── Row 1: K 线 + 通道 ──
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='K线',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
    ), row=1, col=1)

    if not up_l.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=up_l, name=f"入场上轨({CFG['entry_days']}天)",
            line=dict(color='#2196F3', width=1.5),
        ), row=1, col=1)
    if not lo_s.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=lo_s, name=f"入场下轨({CFG['entry_days']}天)",
            line=dict(color='#FF9800', width=1.5),
        ), row=1, col=1)
    if not lo_l.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=lo_l, name=f"多出场({CFG['exit_days']}天)",
            line=dict(color='#2196F3', width=1, dash='dash'),
        ), row=1, col=1)
    if not up_s.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=up_s, name=f"空出场({CFG['exit_days']}天)",
            line=dict(color='#FF9800', width=1, dash='dash'),
        ), row=1, col=1)

    # ── Row 2: GARCH 条件波动率 ──
    if garch:
        cond_vol = garch['cond_vol'].reindex(df.index)
        vol_ma = garch['vol_ma'].reindex(df.index)
        expanding = garch['expanding'].reindex(df.index).fillna(False).astype(bool)

        fig.add_trace(go.Scatter(
            x=df.index, y=cond_vol, name='σ_t (条件波动率)',
            line=dict(color='#E040FB', width=1.5),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=vol_ma, name=f'σ̄ (均值 {GARCH_DAYS}天)',
            line=dict(color='#FFD740', width=1.2, dash='dash'),
        ), row=2, col=1)

        # 扩张区间: 用填充面积标出 (逐段 add_vrect 在短周期上会非常慢)
        fig.add_trace(go.Scatter(
            x=df.index, y=cond_vol.where(expanding), name='扩张 (允许入场)',
            mode='none', fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.18)',
        ), row=2, col=1)

    # ── Row 3: ATR ──
    if not atr_s.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=atr_s, name=f'ATR({ATR_DAYS}天)',
            line=dict(color='#00BCD4', width=1.5),
            fill='tozeroy', fillcolor='rgba(0, 188, 212, 0.1)',
        ), row=r_atr, col=1)

    # ── Row 4: 成交量 ──
    if 'volume' in df.columns:
        colors = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(
            x=df.index, y=df['volume'], name='成交量',
            marker_color=colors, opacity=0.5,
        ), row=r_vol, col=1)

    name = signal_data['name']
    fig.update_layout(
        title=f"{name}/USDT  {CFG['tf']}  唐奇安 {CFG['entry_days']}/{CFG['exit_days']} 天"
              + ('  + GARCH 过滤' if CFG['use_garch'] else ''),
        height=850,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=10)),
        margin=dict(l=50, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text='价格', row=1, col=1)
    if garch:
        fig.update_yaxes(title_text='σ', row=2, col=1)
    fig.update_yaxes(title_text='ATR', row=r_atr, col=1)
    fig.update_yaxes(title_text='Vol', row=r_vol, col=1)

    return fig


# ════════════════════════════════════════════════════════════════
#  Streamlit 页面
# ════════════════════════════════════════════════════════════════

def signal_color(signal_type):
    if signal_type.startswith('entry_'):   return '🟢'
    if signal_type.startswith('exit_'):    return '🔴'
    if signal_type.startswith('holding_'): return '🔵'
    if signal_type.startswith('near_'):    return '🟡'
    return '⚪'


def main():
    st.set_page_config(
        page_title='海龟信号',
        page_icon='🐢',
        layout='wide',
    )

    st.title('🐢 海龟交易信号')

    # ── 侧边栏: 参数 ──
    with st.sidebar:
        st.header('⚙️ 参数')
        coins = st.multiselect('币种', COIN_CHOICES, default=DEFAULT_COINS, key='coins')
        extra = st.text_input('其他币 (逗号分隔, 如 PEPE,WIF)', key='extra_coins')
        coins = list(dict.fromkeys(coins + [c.strip().upper() for c in extra.split(',') if c.strip()]))
        tf = st.selectbox('K线周期', list(TF_BARS_PER_DAY), index=list(TF_BARS_PER_DAY).index(DEFAULT_TF), key='tf')

        c1, c2 = st.columns(2)
        entry_days = c1.number_input('入场通道 (天)', 5, 300, DEFAULT_ENTRY_DAYS, step=5, key='entry_days')
        exit_days = c2.number_input('出场通道 (天)', 2, 200, DEFAULT_EXIT_DAYS, step=5, key='exit_days')
        stop = st.number_input('止损 (几倍 ATR)', 0.5, 10.0, DEFAULT_STOP, step=0.5, key='stop')
        c3, c4 = st.columns(2)
        capital = c3.number_input('资金 (USDT)', 100, 10_000_000, DEFAULT_CAPITAL, step=1000, key='capital')
        risk_pct = c4.number_input('每笔风险 %', 0.1, 5.0, DEFAULT_RISK_PCT * 100, step=0.1, key='risk') / 100

        use_garch = st.toggle('启用 GARCH 过滤', value=False, key='use_garch',
                              help='开启后, 只在 GARCH 条件波动率高于过去 20 天均值时才提示入场. '
                                   '回测检验中 GARCH 过滤没有稳定改善结果, 默认关闭.')

        bpd = TF_BARS_PER_DAY[tf]
        st.caption(f"{tf} 周期: 入场 {entry_days} 天 = {int(entry_days * bpd)} 根K线, "
                   f"出场 {exit_days} 天 = {int(exit_days * bpd)} 根; ATR {ATR_DAYS} 天")
        if bpd >= 48 and entry_days >= 60:
            st.caption('⏳ 短周期 + 长通道需要拉取几千根K线, 第一次会慢一些')

        st.divider()
        tg_status = '✅ 已配置' if has_telegram_config() else '未配置 (不推送)'
        st.markdown(f'**Telegram 通知**: {tg_status}')

        refresh = st.selectbox('自动刷新', list(REFRESH_CHOICES),
                               index=list(REFRESH_CHOICES).index(DEFAULT_REFRESH), key='refresh',
                               help='刷新时只拉最新几根K线, 一般一两秒; 历史数据留在内存里')
        show_chart = st.toggle('显示K线图', value=True, key='show_chart')
        if st.button('🔄 立即刷新', use_container_width=True):
            st.rerun()

    if not coins:
        st.info('在左侧选择至少一个币种')
        return
    symbols = tuple(f'{c}/USDT:USDT' for c in coins)
    global CFG
    CFG = make_cfg(symbols, tf, entry_days, exit_days, stop, capital, risk_pct, use_garch)

    args = (symbols, tf, entry_days, exit_days, stop, capital, risk_pct, use_garch, show_chart)
    interval = REFRESH_CHOICES[refresh]
    if interval and hasattr(st, 'fragment'):
        # 只重跑信号区域, 侧边栏和页面其他部分不动
        st.fragment(run_every=interval)(render_signals)(*args)
    else:
        render_signals(*args)
        if interval:                                   # 旧版 Streamlit 没有 fragment: 倒计时后整页刷新
            countdown = st.empty()
            for remaining in range(interval, 0, -1):
                countdown.caption(f'⏱ {remaining} 秒后自动刷新')
                time.sleep(1)
            st.rerun()


def render_signals(symbols, tf, entry_days, exit_days, stop, capital, risk_pct, use_garch, show_chart):
    """信号区域: 扫描 + 汇总表 + 逐币种面板."""
    global CFG
    CFG = make_cfg(symbols, tf, entry_days, exit_days, stop, capital, risk_pct, use_garch)
    first_load = any((sym, tf) not in _bar_store() for sym in symbols)
    if first_load:
        with st.spinner('第一次加载: 下载历史K线...'):
            signals, okx_pos, balance, scan_time = run_full_scan()
    else:
        signals, okx_pos, balance, scan_time = run_full_scan()

    st.caption(f'⏱ 更新时间: {scan_time}')

    # ── 汇总表: 一眼看全部币 ──
    rows = []
    for s_ in signals:
        p_ = s_['price']
        rows.append({
            '': signal_color(s_['signal_type']), '币种': s_['name'], '信号': s_['signal'],
            '价格': fmt_price(p_) if p_ > 0 else '—',
            '距上轨': f"{(s_['upper'] - p_) / p_ * 100:+.1f}%" if p_ > 0 and s_.get('upper') else '—',
            '距下轨': f"{(s_['lower'] - p_) / p_ * 100:+.1f}%" if p_ > 0 and s_.get('lower') else '—',
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── 账户余额 ──
    if balance and balance['total'] > 0:
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric('💰 账户总额', f"{balance['total']:,.2f} USDT")
        bc2.metric('可用', f"{balance['free']:,.2f} USDT")
        bc3.metric('已用', f"{balance['used']:,.2f} USDT")
        st.divider()

    # ── 逐币种面板 ──
    for s in signals:
        emoji = signal_color(s['signal_type'])
        garch_badge = ('🟢 扩张' if s.get('garch_expanding') else '🔴 收缩') if CFG['use_garch'] else '— 未启用'

        st.subheader(f"{emoji} {s['name']}  —  {s['signal']}")

        # 核心指标行
        if CFG['use_garch']:
            c1, c2, c3, c4, c5 = st.columns(5)
            c3.metric('📈 GARCH σ', f"{s.get('garch_vol', 0):.4f}")
            c4.metric('📉 GARCH σ̄', f"{s.get('garch_vol_ma', 0):.4f}")
            c5.metric('🔬 GARCH 状态', garch_badge)
        else:
            c1, c2 = st.columns(2)
        c1.metric('💲 价格', fmt_price(s['price']) if s['price'] > 0 else '—')
        c2.metric('📊 ATR', fmt_price(s['atr']) if s.get('atr', 0) > 0 else '—')

        # 通道 + 仓位参数
        c6, c7, c8, c9 = st.columns(4)
        c6.metric('⬆ 入场上轨', fmt_price(s['upper']) if s.get('upper', 0) > 0 else '—')
        c7.metric('⬇ 入场下轨', fmt_price(s['lower']) if s.get('lower', 0) > 0 else '—')
        c8.metric('🔽 多出场', fmt_price(s.get('exit_lower', 0)) if s.get('exit_lower', 0) > 0 else '—')
        c9.metric('🔼 空出场', fmt_price(s.get('exit_upper', 0)) if s.get('exit_upper', 0) > 0 else '—')

        # 仓位建议 (入场信号时)
        if s.get('suggested_qty') and s['suggested_qty'] > 0:
            val = s['suggested_qty'] * s['price']
            st.info(
                f"**仓位**: {s['suggested_qty']:.6f} {s['name']}  "
                f"(≈ {val:,.0f} USDT)  |  "
                f"**止损**: {fmt_price(s['suggested_stop'])}  |  "
                f"**风险**: {s['risk_usdt']:.0f} USDT  |  "
                f"**止损距离**: {CFG['stop']:g}N = {CFG['stop'] * s['atr']:.2f}"
            )

        # GARCH 模型参数
        garch = s.get('garch_info')
        if garch:
            with st.expander(f'🔬 GARCH(1,1) 模型参数 — {s["name"]}'):
                gc1, gc2, gc3, gc4 = st.columns(4)
                gc1.metric('α (冲击系数)', f"{garch['alpha']:.4f}")
                gc2.metric('β (惯性系数)', f"{garch['beta']:.4f}")
                gc3.metric('ω (基准方差)', f"{garch['omega']:.6f}")
                gc4.metric('允许入场占比', f"{garch['pct_on']:.1f}%")

                st.caption(
                    f"σ²_t = {garch['omega']:.6f} + "
                    f"{garch['alpha']:.4f} × r²_{{t-1}} + "
                    f"{garch['beta']:.4f} × σ²_{{t-1}}  |  "
                    f"α+β = {garch['alpha'] + garch['beta']:.4f}  "
                    f"({'高持续性' if garch['alpha'] + garch['beta'] > 0.95 else '中等持续性'})"
                )

        # 信号详情
        if s.get('detail'):
            st.caption(s['detail'])

        # K 线图
        fig = plot_kline_garch(s) if show_chart else None
        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{s['symbol']}")

        st.divider()

    # ── OKX 实盘持仓 ──
    if okx_pos:
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



if __name__ == '__main__':
    main()
