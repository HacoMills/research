#!/usr/bin/env python3
"""
海龟交易系统 — 多时间框架回测
================================
自动从 OKX 拉取 15min K线, 重采样到: 15min / 30min / 1h / 4h / 1d
在每个时间框架上分别跑 系统1 (20/10天) 和 系统2 (60/20天)
仓位管理: 1% 资金 / ATR
止损: 2N

用法:
  python turtle_timeframe_backtest.py                    # 默认 BTC, 从2022起
  python turtle_timeframe_backtest.py --symbol ETH/USDT:USDT
  python turtle_timeframe_backtest.py --start 2023-01-01
  python turtle_timeframe_backtest.py --refresh          # 强制重新拉取数据
"""

import sys, time, pandas as pd #sys/time控制扒数据的频率，
import numpy as np
import warnings, os #减少报错 →23
from pathlib import Path
from typing import List, Dict, Tuple

warnings.filterwarnings("ignore")

# ── 项目路径 ──
_project_root = str(next(p for p in Path(__file__).resolve().parents if (p / "data" / "paths.py").exists())) #resolve转成绝对路径，.parent转到上一级文件夹，两个就是转到根目录 (FROM pathlib)
# 文件夹整理后多了一层: parents[0]=turtle, parents[1]=strategies, parents[2]=quant (data 在这里)
# 为了以后挪文件夹也不出错, 改成: 沿 parents 往上找, 第一个含 data/paths.py 的就是 quant 根目录

if _project_root not in sys.path:    #sys.path列表在文件里面找有没有叫xxx的文件夹
    sys.path.insert(0, _project_root) # <class 'list'>

# 路径统一在 quant/data/paths.py 里定义:
#   图片 → quant/figures/turtle/   报告 → quant/reports/turtle/   K线缓存 → quant/data/cache/okx/
from data.paths import FIGURES_DIR as _FIG_ROOT, REPORTS_DIR as _REP_ROOT, OKX_CACHE
FIGURES_DIR = _FIG_ROOT / "turtle"
REPORTS_DIR = _REP_ROOT / "turtle"

# 结构	写法	特点
# 列表 list	[a, b, c]	有序、可修改
# 元组 tuple	(a, b, c)	有序、不可修改
# 字典 dict	{"k": v}	键值对，按键取值
# 集合 set	{a, b, c}	无序、不

from data.fetch_data import create_exchange, fetch_ohlcv, fetch_top_symbols  # 在我自己的模块里面加载 这几个函数。那些库也是别人写好的函数，类什么的。


# 类：数据
# 实例：数据造出的具体对策。实例化交易所就是定义一个字典，然后把需要的参数提前填进去，有一些专属函数放在上面。 → 146 exchange = cls(config)   
# 属性：对象上面的数据
# 方法：函数，操作实例


#python 数据结构

#1. 基础类型：一个值
#整数 int、浮点数 float、字符串 str、布尔 bool（True/False）、None

#2. 容器：装多个值
#列表 list、元组 tuple、字典 dict、集合 set

#3. 函数：把一段操作打包，可以重复调用
#def calc_atr(df, period): ...

#4. 类与实例：把数据和函数打包
#类（模板）→ 实例（具体对象）→ 属性（数据）+ 方法（函数）

#5. 模块、包、库：把以上所有东西组织成文件
#模块（一个 .py）→ 包（文件夹）→ 库（发布出来的包）


# ──────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────

INITIAL_CAPITAL = 10_000.0
RISK_PCT = 0.01            # 1% 风险
STOP_MULT = 2.0            # 2N 止损
ATR_PERIOD_DAYS = 20       # ATR 周期 (天)
COMMISSION_PCT = 0.05 / 100

# 时间框架: (名称, pandas resample rule, 每天有多少根K线)
TIMEFRAMES = [
    ("15min",  None,   96),    # 原始数据，不需要 resample
    ("30min",  "30min", 48),
    ("1h",     "1h",   24),
    ("4h",     "4h",    6),
    ("1d",     "1D",    1),
]

# 海龟系统配置: (名称, 入场天数, 出场天数)
SYSTEMS = [
    ("系统1 (20/10)", 20, 10),
    ("系统2 (60/20)", 60, 20),
]


# ──────────────────────────────────────────────────
# 数据处理
# ──────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame: #:在函数定义里面写注释 | .提取属性
    df = pd.read_csv(path, parse_dates=["timestamp"]) # 把时间转成datatime类型
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    print(f"原始数据: {len(df)} 行, {df.index[0]} ~ {df.index[-1]}")
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """将K线数据重采样到更大的时间框架"""
    resampled = df.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum"
    }).dropna()
    return resampled


# ──────────────────────────────────────────────────
# 指标
# ──────────────────────────────────────────────────

def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


def calc_donchian(series_high: pd.Series, series_low: pd.Series, period: int):
    upper = series_high.rolling(period).max().shift(1)
    lower = series_low.rolling(period).min().shift(1)
    return upper, lower


# ──────────────────────────────────────────────────
# GARCH 波动率过滤器
# ──────────────────────────────────────────────────

def compute_garch_filter(df: pd.DataFrame, lookback: int = 20,
                         verbose: bool = True) -> pd.Series:
    """
    GARCH(1,1) 波动率过滤器 —— 全样本版 (有前视偏差, 仅用于对比).

    ⚠️ 用整段数据一次性拟合 α、β: 2022 年的过滤信号用到了 2022~至今全部数据
       估出的参数. 回测默认改用下面的 compute_garch_filter_rolling;
       想复现这个版本, 运行时加 --garch-mode insample.

    原理: 拟合 GARCH(1,1) 得到条件波动率 σ_t,
          当 σ_t > rolling_mean(σ, lookback) 时认为波动率在扩张,
          趋势策略在波动率扩张期入场更有优势.

    返回: 布尔 Series, True = 允许入场 (波动率扩张)
          失败时返回 None
    """
    try:
        from arch import arch_model
    except ImportError:
        if verbose:
            print("  [!] 需要安装 arch 库: pip install arch --break-system-packages")
        return None

    close = df["close"].dropna()
    if len(close) < 200:
        if verbose:
            print("  [!] 数据太短, 无法拟合 GARCH")
        return None

    # 对数收益率 (×100 缩放, arch 库习惯)
    log_ret = (np.log(close / close.shift(1)) * 100).dropna()

    try:
        model = arch_model(log_ret, vol="Garch", p=1, q=1,
                           mean="Zero", rescale=False)
        res = model.fit(disp="off", show_warning=False)
    except Exception as e:
        if verbose:
            print(f"  [!] GARCH 拟合失败: {e}")
        return None

    # 条件波动率
    cond_vol = res.conditional_volatility

    # 滚动均值 (与 ATR lookback 一致)
    vol_ma = cond_vol.rolling(window=lookback).mean()

    # σ_t > σ̄  →  波动率扩张  →  允许 Turtle 入场
    expanding = cond_vol > vol_ma

    # 对齐回原始 df 的 index, 缺失位置默认 False
    garch_filter = pd.Series(False, index=df.index)
    garch_filter.loc[expanding.index] = expanding.values

    if verbose:
        pct_on = garch_filter.sum() / len(garch_filter) * 100
        print(f"    GARCH 过滤: {pct_on:.1f}% 时间允许入场"
              f"  (α={res.params.get('alpha[1]', 0):.4f},"
              f" β={res.params.get('beta[1]', 0):.4f})")

    return garch_filter


def compute_garch_filter_rolling(df: pd.DataFrame, lookback: int,
                                 min_train: int, refit_every: int,
                                 verbose: bool = True) -> pd.Series:
    """
    GARCH(1,1) 波动率过滤器 —— 滚动估计版 (无前视偏差, 回测默认使用).

    做法 (扩展窗口, 定期重估):
      - 前 min_train 根K线只用来训练, 这段时间不过滤 (与基线相同, 便于公平对比)
      - 之后每隔 refit_every 根K线, 只用"此前"的收益率重新拟合一次 ω、α、β
      - 两次重估之间, 用最近一次的参数按递推公式逐根计算条件方差:
            σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
        σ_t 只依赖 t-1 及以前的收益, 在第 t 根K线收盘决策时已经知道
      - 过滤规则与原版相同: σ_t > 过去 lookback 根 σ 的均值 → 允许入场

    检验方法: 把数据截断到任意一天重新计算, 那天之前的过滤信号完全不变.
    """
    try:
        from arch import arch_model
    except ImportError:
        if verbose:
            print("  [!] 需要安装 arch 库: pip install arch")
        return None

    r = (np.log(df["close"] / df["close"].shift(1)) * 100).dropna()
    n = len(r)
    if n < min_train + lookback:
        if verbose:
            print("  [!] 数据太短, 无法滚动拟合 GARCH")
        return None

    rv = r.to_numpy()
    sigma2 = np.full(n, np.nan)
    params, s2_prev, n_fit, n_fail = None, None, 0, 0

    for start in range(min_train, n, refit_every):
        try:
            res = arch_model(r.iloc[:start], vol="Garch", p=1, q=1,        # 只用 start 之前的数据
                             mean="Zero", rescale=False).fit(disp="off", show_warning=False)
            params = (res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"])
            n_fit += 1
            if s2_prev is None:          # 第一段: 用训练期最后一天的条件方差起步
                s2_prev = float(res.conditional_volatility.iloc[-1]) ** 2
        except Exception:
            n_fail += 1
            if params is None:
                continue                 # 还没有可用参数, 这段不过滤
        omega, alpha, beta = params
        for t in range(start, min(start + refit_every, n)):
            s2_prev = omega + alpha * rv[t - 1] ** 2 + beta * s2_prev
            sigma2[t] = s2_prev

    cond_vol = pd.Series(np.sqrt(sigma2), index=r.index)
    vol_ma = cond_vol.rolling(lookback).mean()
    valid = cond_vol.notna() & vol_ma.notna()

    garch_filter = pd.Series(True, index=df.index)          # 训练期: 不过滤
    garch_filter.loc[valid[valid].index] = (cond_vol > vol_ma)[valid].values

    if verbose and params is not None:
        pct_on = garch_filter[valid.reindex(df.index, fill_value=False)].mean() * 100
        print(f"    GARCH(滚动) 过滤: 训练期后 {pct_on:.1f}% 时间允许入场"
              f"  (重估 {n_fit} 次" + (f", 失败 {n_fail} 次" if n_fail else "")
              + f", 最近 α={params[1]:.4f}, β={params[2]:.4f})")
    return garch_filter


# ──────────────────────────────────────────────────
# 回测引擎
# ──────────────────────────────────────────────────
#
# 现在的 run_backtest 调用共用引擎 turtle_engine.py (逐根盯市、夏普、止损跳空、滑点).
# 下面的 run_backtest_legacy 是旧版引擎, 保留作对照: 运行时加 --engine legacy 可复现旧结果.

SLIPPAGE = 0.0003          # 每次成交 3 个基点滑点 (新引擎)
USE_LEGACY = False         # 由 --engine legacy 打开


def run_backtest(df: pd.DataFrame, entry_days: int, exit_days: int,
                 bars_per_day: int, vol_filter: pd.Series = None) -> Dict:
    """海龟回测 (新引擎). 参数与旧版相同, 返回的结果多了夏普、年化波动等字段."""
    if USE_LEGACY:
        return run_backtest_legacy(df, entry_days, exit_days, bars_per_day, vol_filter)
    from turtle_engine import MarketSpec, run_backtest as engine_run
    spec = MarketSpec(name="crypto", timeframes=TIMEFRAMES, days_per_year=365.25,
                      initial_capital=INITIAL_CAPITAL, allow_short=True,
                      open_fee=COMMISSION_PCT, close_fee=COMMISSION_PCT,
                      slippage=SLIPPAGE)
    return engine_run(df, entry_days, exit_days, bars_per_day, spec, vol_filter=vol_filter)


def run_backtest_legacy(df: pd.DataFrame, entry_days: int, exit_days: int,
                        bars_per_day: int, vol_filter: pd.Series = None) -> Dict:
    """
    海龟回测 — 单Unit, 1%风险/ATR仓位 (旧版引擎, 仅作对照)
    ⚠️ 权益只在平仓时更新 (回撤偏小)、止损跳空按止损价成交、无滑点
    vol_filter: 可选的 GARCH 波动率过滤器 (布尔 Series, True=允许入场)
    """
    entry_bars = entry_days * bars_per_day
    exit_bars = exit_days * bars_per_day
    atr_bars = ATR_PERIOD_DAYS * bars_per_day

    # 日线特殊处理: 最小 ATR 周期不小于 14
    if bars_per_day == 1:
        atr_bars = max(atr_bars, 14)

    atr = calc_atr(df, atr_bars)
    entry_upper, entry_lower = calc_donchian(df["high"], df["low"], entry_bars)
    exit_upper, exit_lower = calc_donchian(df["high"], df["low"], exit_bars)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr_vals = atr.values
    n = len(df)

    equity = INITIAL_CAPITAL
    peak_equity = equity
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    qty = 0.0

    trades = []
    equity_curve = np.full(n, np.nan)

    warmup = max(entry_bars, exit_bars, atr_bars) + 2

    for i in range(warmup, n):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            equity_curve[i] = equity
            continue

        price = close[i]

        # ── 持仓检查出场 ──
        if position != 0:
            exit_signal = False
            exit_reason = ""
            exit_p = price

            if position == 1:
                if low[i] <= stop_price:
                    exit_p = stop_price
                    exit_reason = "止损"
                    exit_signal = True
                elif not np.isnan(exit_lower.iloc[i]) and price < exit_lower.iloc[i]:
                    exit_p = price
                    exit_reason = "通道"
                    exit_signal = True
            else:
                if high[i] >= stop_price:
                    exit_p = stop_price
                    exit_reason = "止损"
                    exit_signal = True
                elif not np.isnan(exit_upper.iloc[i]) and price > exit_upper.iloc[i]:
                    exit_p = price
                    exit_reason = "通道"
                    exit_signal = True

            if exit_signal:
                pnl = position * (exit_p - entry_price) * qty
                comm = (entry_price + exit_p) * qty * COMMISSION_PCT
                pnl -= comm
                equity += pnl
                trades.append({
                    "dir": position, "entry": entry_price, "exit": exit_p,
                    "pnl": pnl, "reason": exit_reason,
                    "bars_held": i - trades[-1]["entry_i"] if trades and "entry_i" in trades[-1] else 0,
                    "entry_i": i
                })
                # 修正: 记录正确的 bars_held
                trades[-1]["bars_held"] = i - trades[-1].get("_ei", i)
                position = 0

        # ── 空仓检查入场 ──
        if position == 0:
            # GARCH 波动率过滤: 波动率收缩期不入场
            if vol_filter is not None and not vol_filter.iloc[i]:
                equity_curve[i] = equity
                continue

            unit_size = (equity * RISK_PCT) / atr_vals[i]
            if unit_size <= 0 or equity <= 0:
                equity_curve[i] = equity
                continue

            if not np.isnan(entry_upper.iloc[i]) and price > entry_upper.iloc[i]:
                position = 1
                entry_price = price
                qty = unit_size
                stop_price = price - STOP_MULT * atr_vals[i]
                trades.append({"_ei": i, "entry_i": i})
                trades.pop()  # placeholder removed
                # 用简化方式记录
                _entry_i = i
            elif not np.isnan(entry_lower.iloc[i]) and price < entry_lower.iloc[i]:
                position = -1
                entry_price = price
                qty = unit_size
                stop_price = price + STOP_MULT * atr_vals[i]
                _entry_i = i

        equity_curve[i] = equity
        peak_equity = max(peak_equity, equity)

    # 末尾平仓
    if position != 0:
        exit_p = close[-1]
        pnl = position * (exit_p - entry_price) * qty
        comm = (entry_price + exit_p) * qty * COMMISSION_PCT
        pnl -= comm
        equity += pnl
        trades.append({"dir": position, "entry": entry_price, "exit": exit_p,
                        "pnl": pnl, "reason": "结束"})
        equity_curve[-1] = equity

    # ── 统计 ──
    pnl_list = [t["pnl"] for t in trades if "pnl" in t]
    n_trades = len(pnl_list)

    if n_trades == 0:
        return _empty_result()

    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p <= 0]
    win_rate = len(wins) / n_trades * 100

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_ret = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # 最大回撤
    eq = pd.Series(equity_curve).dropna()
    if len(eq) > 1:
        rm = eq.cummax()
        dd = ((eq - rm) / rm * 100).min()
    else:
        dd = 0

    # 年化收益
    total_bars = n - warmup
    years = total_bars / (bars_per_day * 365.25) if bars_per_day > 0 else 1
    if years > 0 and equity > 0:
        ann_ret = ((equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    else:
        ann_ret = 0

    # Calmar ratio
    calmar = ann_ret / abs(dd) if dd != 0 else 0

    return {
        "total_return_pct": round(total_ret, 2),
        "annual_return_pct": round(ann_ret, 2),
        "final_equity": round(equity, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 3),
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "max_drawdown_pct": round(dd, 2),
        "calmar": round(calmar, 3),
        "equity_curve": equity_curve,
        "years": round(years, 2),
    }


def _empty_result():
    return {
        "total_return_pct": 0, "annual_return_pct": 0, "final_equity": INITIAL_CAPITAL,
        "n_trades": 0, "win_rate": 0, "profit_factor": 0,
        "avg_win": 0, "avg_loss": 0, "max_drawdown_pct": 0,
        "calmar": 0, "equity_curve": np.array([]), "years": 0,
    }


# ──────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────

def fetch_or_load(symbol="BTC/USDT:USDT", base_tf="15m",
                   start_date="2022-01-01"):
    """
    从 OKX 拉取历史 K 线, 支持增量更新.
    - 首次运行: 全量拉取并缓存为 CSV
    - 后续运行: 读取已有 CSV, 仅拉取新增的K线, 追加到缓存
    - --refresh 模式: 缓存目录已被 main() 删除, 自动走全量拉取
    """
    cache_dir = str(OKX_CACHE)           # quant/data/cache/okx/ (各项目共用)
    os.makedirs(cache_dir, exist_ok=True)

    tag = symbol.replace("/", "").replace(":", "_").lower()
    cache_file = os.path.join(cache_dir, f"{tag}_{base_tf}_{start_date}.csv")

    ex = create_exchange()

    # ── 增量更新: 已有缓存则只拉新数据 ──
    if os.path.exists(cache_file):
        df_existing = pd.read_csv(cache_file, parse_dates=["timestamp"])
        df_existing.set_index("timestamp", inplace=True)
        df_existing = df_existing[~df_existing.index.duplicated(keep='first')]

        last_ts = df_existing.index.max()
        # 转换为毫秒时间戳. 从最后一根本身开始拉 (不再 +1):
        # 上次保存时最后一根可能还没收盘, 重新拉取后用新数据覆盖 (下面 keep='last')
        last_ts_ms = int(last_ts.timestamp() * 1000)

        print(f"缓存已有 {len(df_existing)} 根K线 ({df_existing.index[0]} ~ {last_ts})")
        print(f"  增量拉取 {last_ts} 之后的新数据...")

        try:
            df_new = fetch_ohlcv(ex, symbol, timeframe=base_tf,
                                 start_date=start_date, since_ms=last_ts_ms)
        except TypeError:
            # 兼容旧版 fetch_data.py (没有 since_ms 参数)
            print(f"  [注意] fetch_ohlcv 不支持 since_ms, 直接使用缓存数据")
            df_new = None

        if df_new is not None and not df_new.empty:
            # 合并去重
            df = pd.concat([df_existing, df_new])
            df = df[~df.index.duplicated(keep='last')]  # 新数据优先
            df.sort_index(inplace=True)
            new_count = len(df) - len(df_existing)
            print(f"  新增 {new_count} 根K线, 总计 {len(df)} 根")
            # 保存更新后的缓存
            df.to_csv(cache_file)
        else:
            df = df_existing
            print(f"  无新数据, 使用已有缓存 ({len(df)} 根K线)")

        print(f"原始数据: {len(df)} 行, {df.index[0]} ~ {df.index[-1]}")
        return df

    # ── 全量拉取: 没有缓存 ──
    print(f"正在从 OKX 全量拉取 {symbol} {base_tf} 数据 (从 {start_date})...")
    print("  首次运行可能需要几分钟, 请耐心等待...\n")

    df = fetch_ohlcv(ex, symbol, timeframe=base_tf, start_date=start_date)

    if df is None or df.empty:
        print(f"\n[!] {symbol} 无法获取数据 (该币种可能未在 OKX 上市永续合约)")
        return None

    # 保存缓存
    df.to_csv(cache_file)
    print(f"\n已缓存至: {cache_file}")
    print(f"原始数据: {len(df)} 行, {df.index[0]} ~ {df.index[-1]}")
    return df


PRICES = {}   # {币种: 收盘价}, run_single_coin 填入, 报告里算"等权买入持有"基准用


def run_single_coin(symbol, start_date, verbose=True, use_garch=False,
                    garch_mode="rolling", garch_refit_days=21):
    """
    对单个币种跑全部时间框架 × 系统的回测, 返回 results dict.
    key = (tf_name, sys_name), value = backtest result

    use_garch       : 是否启用 GARCH(1,1) 波动率过滤器
    garch_mode      : "rolling" 滚动重估, 无前视偏差 (默认) / "insample" 原版全样本拟合
    garch_refit_days: 滚动模式下每隔多少天重估一次参数
    """
    coin = symbol.split("/")[0]
    df_raw = fetch_or_load(symbol=symbol, base_tf="15m", start_date=start_date)

    if df_raw is None or df_raw.empty:
        return None
    PRICES[coin] = df_raw["close"]          # 组合分析的基准要用

    all_results = {}

    for tf_name, tf_rule, bars_day in TIMEFRAMES:
        if tf_rule is None:
            df_tf = df_raw.copy()
        else:
            df_tf = resample_ohlcv(df_raw, tf_rule)

        if verbose:
            print(f"  ▸ {tf_name}  ({len(df_tf)} 根K线)")

        # GARCH 过滤器: 每个时间框架独立计算
        garch_filter = None
        if use_garch:
            garch_lookback = max(ATR_PERIOD_DAYS * bars_day, 20)
            if garch_mode == "insample":
                garch_filter = compute_garch_filter(df_tf, lookback=garch_lookback,
                                                     verbose=verbose)
            else:
                garch_filter = compute_garch_filter_rolling(
                    df_tf, lookback=garch_lookback,
                    min_train=int(365.25 * bars_day),                  # 训练 1 年
                    refit_every=max(int(garch_refit_days * bars_day), 1),
                    verbose=verbose)

        for sys_name, entry_d, exit_d in SYSTEMS:
            entry_bars = entry_d * bars_day
            exit_bars = exit_d * bars_day
            min_bars_needed = max(entry_bars, exit_bars, ATR_PERIOD_DAYS * bars_day) + 10
            if len(df_tf) < min_bars_needed:
                if verbose:
                    print(f"    {sys_name}: 数据不足, 跳过")
                continue

            result = run_backtest(df_tf, entry_d, exit_d, bars_day,
                                  vol_filter=garch_filter)
            all_results[(tf_name, sys_name)] = result

            if verbose:
                print(f"    {sys_name}  收益:{result['total_return_pct']:>+8.1f}%"
                      f"  年化:{result['annual_return_pct']:>+6.1f}%"
                      f"  夏普:{result.get('sharpe', 0):>5.2f}"
                      f"  PF:{result['profit_factor']:>5.2f}"
                      f"  回撤:{result['max_drawdown_pct']:>7.1f}%"
                      f"  Calmar:{result['calmar']:>6.3f}")

    return all_results


def print_single_coin_table(all_results, coin_name):
    """打印单币种的汇总对比表."""
    print(f"\n{'='*90}")
    print(f"{'汇总对比表 — ' + coin_name:^90}")
    print(f"{'='*90}")

    header = (f"{'K线':>5} {'系统':>12} {'交易':>5} {'总收益%':>9} {'年化%':>7} {'夏普':>6}"
              f" {'胜率%':>6} {'PF':>6} {'平均赢':>9} {'平均亏':>9}"
              f" {'回撤%':>8} {'Calmar':>7}")
    print(header)
    print("─" * 90)

    for tf_name, _, _ in TIMEFRAMES:
        for sys_name, _, _ in SYSTEMS:
            key = (tf_name, sys_name)
            if key not in all_results:
                continue
            r = all_results[key]
            print(f"{tf_name:>5} {sys_name:>12}"
                  f" {r['n_trades']:>5}"
                  f" {r['total_return_pct']:>+8.1f}"
                  f" {r['annual_return_pct']:>+6.1f}"
                  f" {r.get('sharpe', 0):>6.2f}"
                  f" {r['win_rate']:>5.1f}"
                  f" {r['profit_factor']:>5.2f}"
                  f" {r['avg_win']:>8.1f}"
                  f" {r['avg_loss']:>8.1f}"
                  f" {r['max_drawdown_pct']:>7.1f}"
                  f" {r['calmar']:>6.3f}")
        print("─" * 90)


def print_multi_coin_summary(grand_results, tf_name, sys_name):
    """
    打印某个 (时间框架, 系统) 组合下所有币种的横向对比.
    grand_results: {coin: {(tf, sys): result}}
    """
    coins = sorted(grand_results.keys())
    rows = []
    for coin in coins:
        r = grand_results[coin].get((tf_name, sys_name))
        if r:
            rows.append((coin, r))
    if not rows:
        return

    # 按年化收益排序
    rows.sort(key=lambda x: x[1]["annual_return_pct"], reverse=True)

    print(f"\n{'='*80}")
    print(f"  {tf_name} × {sys_name}  — {len(rows)} 个币种")
    print(f"{'='*80}")
    print(f"{'币种':>6} {'交易':>5} {'总收益%':>9} {'年化%':>7} {'夏普':>6}"
          f" {'胜率%':>6} {'PF':>6} {'回撤%':>8} {'Calmar':>7}")
    print("─" * 80)

    ann_list, dd_list, pf_list, calmar_list = [], [], [], []

    for coin, r in rows:
        print(f"{coin:>6}"
              f" {r['n_trades']:>5}"
              f" {r['total_return_pct']:>+8.1f}"
              f" {r['annual_return_pct']:>+6.1f}"
              f" {r.get('sharpe', 0):>6.2f}"
              f" {r['win_rate']:>5.1f}"
              f" {r['profit_factor']:>5.2f}"
              f" {r['max_drawdown_pct']:>7.1f}"
              f" {r['calmar']:>6.3f}")
        ann_list.append(r["annual_return_pct"])
        dd_list.append(r["max_drawdown_pct"])
        pf_list.append(r["profit_factor"])
        calmar_list.append(r["calmar"])

    print("─" * 80)
    n_pos = sum(1 for a in ann_list if a > 0)
    print(f"  中位数       "
          f" {np.median(ann_list):>+6.1f}"
          f" {'':>6}"
          f" {np.median(pf_list):>5.2f}"
          f" {np.median(dd_list):>7.1f}"
          f" {np.median(calmar_list):>6.3f}")
    print(f"  均  值       "
          f" {np.mean(ann_list):>+6.1f}"
          f" {'':>6}"
          f" {np.mean(pf_list):>5.2f}"
          f" {np.mean(dd_list):>7.1f}"
          f" {np.mean(calmar_list):>6.3f}")
    print(f"  盈利币种: {n_pos}/{len(rows)} ({n_pos/len(rows)*100:.0f}%)")


def save_single_chart(all_results, coin_name, out_dir):
    """保存单币种的 4 子图图表到 quant/figures/turtle/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for font in ["DejaVu Sans", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]:
            try:
                plt.rcParams["font.sans-serif"] = [font]
                plt.rcParams["axes.unicode_minus"] = False
                fig_t, ax_t = plt.subplots(); ax_t.set_title("t"); plt.close(fig_t)
                break
            except Exception:
                continue

        # 图表统一放到 charts/ 子文件夹
        charts_dir = str(FIGURES_DIR)
        os.makedirs(charts_dir, exist_ok=True)

        n_tf = len(TIMEFRAMES)
        fig, axes = plt.subplots(2, 2, figsize=(18, 13))
        fig.suptitle(f"Turtle System — Multi-Timeframe ({coin_name}, 1% Risk/ATR)",
                     fontsize=14, fontweight="bold")

        tf_names = [t[0] for t in TIMEFRAMES]
        sys_names = [s[0] for s in SYSTEMS]
        colors_sys = ["#2196F3", "#FF5722"]

        # 1. 总收益
        ax = axes[0, 0]
        x = np.arange(n_tf)
        width = 0.35
        for j, (sn, sc) in enumerate(zip(sys_names, colors_sys)):
            vals = [all_results.get((tn, sn), {}).get("total_return_pct", 0)
                    for tn in tf_names]
            offset = (j - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=sn, color=sc, alpha=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                        f"{v:+.0f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(tf_names)
        ax.set_title("Total Return by Timeframe"); ax.set_ylabel("Return %")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linewidth=0.5)

        # 2. 最大回撤
        ax = axes[0, 1]
        for j, (sn, sc) in enumerate(zip(sys_names, colors_sys)):
            vals = [abs(all_results.get((tn, sn), {}).get("max_drawdown_pct", 0))
                    for tn in tf_names]
            ax.bar(x + (j - 0.5) * width, vals, width, label=sn, color=sc, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(tf_names)
        ax.set_title("Max Drawdown"); ax.set_ylabel("Drawdown %")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

        # 3. PF & Win Rate
        ax = axes[1, 0]
        ax2 = ax.twinx()
        bw = 0.18
        for j, (sn, sc) in enumerate(zip(sys_names, colors_sys)):
            pf_vals = [min(all_results.get((tn, sn), {}).get("profit_factor", 0), 5)
                       for tn in tf_names]
            wr_vals = [all_results.get((tn, sn), {}).get("win_rate", 0)
                       for tn in tf_names]
            offset = (j - 0.5) * bw * 2
            ax.bar(x + offset - bw/2, pf_vals, bw, label=f"PF {sn}", color=sc, alpha=0.7)
            ax2.plot(x + offset, wr_vals, "o--", color=sc, markersize=6, label=f"WR {sn}")
        ax.set_xticks(x); ax.set_xticklabels(tf_names)
        ax.set_title("PF (bars) & Win Rate (dots)")
        ax.set_ylabel("Profit Factor"); ax2.set_ylabel("Win Rate %")
        ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y"); ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

        # 4. 权益曲线
        ax = axes[1, 1]
        line_styles = ["-", "--", "-.", ":", "-"]
        cm = plt.cm.tab10
        idx = 0
        for (tf_name, _, bars_day), ls in zip(TIMEFRAMES, line_styles):
            key = (tf_name, sys_names[0])
            if key not in all_results or "equity_curve" not in all_results[key]:
                continue
            eq = pd.Series(all_results[key]["equity_curve"]).dropna()
            if len(eq) < 2:
                continue
            step = max(bars_day, 1)
            eq_d = eq.iloc[::step]
            ax.plot(np.arange(len(eq_d)), eq_d.values, label=tf_name,
                    color=cm(idx), linewidth=1.2, linestyle=ls)
            idx += 1
        ax.set_title(f"Equity Curves — {sys_names[0]}")
        ax.set_ylabel("Equity ($)"); ax.set_xlabel("Trading Days")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        chart_path = os.path.join(charts_dir,
                                  f"turtle_tf_{coin_name.lower()}.png")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  图表: {chart_path}")
        return chart_path

    except Exception as e:
        print(f"  [!] 图表出错: {e}")
        return None


def generate_markdown_report(grand_results, failed, start_date, out_dir, tag="crypto"):
    """
    生成 Markdown 格式的回测报告.
    grand_results: {coin: {(tf, sys): result}}
    tag: 组合图片文件名的后缀 (基线 / GARCH 报告各一份)
    """
    from collections import Counter
    from datetime import datetime

    lines = []
    L = lines.append

    L(f"# 海龟交易系统 — 多币种稳健性回测报告\n")
    L(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    L(f"> 回测区间: {start_date} ~ 今  ")
    L(f"> 仓位: 1% 资金/ATR | 止损: 2N | 手续费: 0.05% | "
      + ("旧版引擎 (平仓权益, 无滑点)  " if USE_LEGACY else
         f"滑点: {SLIPPAGE*1e4:.0f}‱/次 | 权益逐根盯市, 止损跳空按开盘价  "))
    L(f"> 币种数: {len(grand_results)} 个成功"
      + (f", {len(failed)} 个跳过 ({', '.join(failed)})" if failed else "") + "\n")

    L("---\n")

    # ── 每个币的详细表 ──
    L("## 一、各币种详细回测\n")

    for coin in sorted(grand_results.keys()):
        results = grand_results[coin]
        L(f"### {coin}\n")
        # 报告和图片不在同一个文件夹, 用相对路径链接 (如 ../../figures/turtle/xxx.png)
        img = os.path.relpath(FIGURES_DIR / f"turtle_tf_{coin.lower()}.png", out_dir).replace("\\", "/")
        L(f"![{coin} 图表]({img})\n")
        L("| K线 | 系统 | 交易 | 总收益% | 年化% | 夏普 | 胜率% | PF | 回撤% | Calmar |")
        L("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for tf_name, _, _ in TIMEFRAMES:
            for sys_name, _, _ in SYSTEMS:
                key = (tf_name, sys_name)
                if key not in results:
                    continue
                r = results[key]
                L(f"| {tf_name} | {sys_name} | {r['n_trades']} "
                  f"| {r['total_return_pct']:+.1f} | {r['annual_return_pct']:+.1f} "
                  f"| {r.get('sharpe', 0):.2f} "
                  f"| {r['win_rate']:.1f} | {r['profit_factor']:.2f} "
                  f"| {r['max_drawdown_pct']:.1f} | {r['calmar']:.3f} |")

        L("")

    # ── 跨币种横向对比 ──
    L("---\n")
    L("## 二、跨币种横向对比\n")

    key_combos = [
        ("4h",  "系统2 (60/20)"),
        ("4h",  "系统1 (20/10)"),
        ("1h",  "系统2 (60/20)"),
        ("1d",  "系统2 (60/20)"),
        ("15min", "系统2 (60/20)"),
        ("30min", "系统2 (60/20)"),
    ]
    for tf_name, sys_name in key_combos:
        rows = []
        for coin in sorted(grand_results.keys()):
            r = grand_results[coin].get((tf_name, sys_name))
            if r:
                rows.append((coin, r))
        if not rows:
            continue

        rows.sort(key=lambda x: x[1]["annual_return_pct"], reverse=True)

        ann_list = [r["annual_return_pct"] for _, r in rows]
        dd_list = [r["max_drawdown_pct"] for _, r in rows]
        pf_list = [r["profit_factor"] for _, r in rows]
        calmar_list = [r["calmar"] for _, r in rows]
        n_pos = sum(1 for a in ann_list if a > 0)

        L(f"### {tf_name} × {sys_name} ({len(rows)} 币种)\n")
        sh_list = [r.get("sharpe", 0) for _, r in rows]
        L("| 币种 | 交易 | 总收益% | 年化% | 夏普 | 胜率% | PF | 回撤% | Calmar |")
        L("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for coin, r in rows:
            L(f"| {coin} | {r['n_trades']} "
              f"| {r['total_return_pct']:+.1f} | {r['annual_return_pct']:+.1f} "
              f"| {r.get('sharpe', 0):.2f} "
              f"| {r['win_rate']:.1f} | {r['profit_factor']:.2f} "
              f"| {r['max_drawdown_pct']:.1f} | {r['calmar']:.3f} |")
        L(f"| **中位数** | | | {np.median(ann_list):+.1f} | {np.median(sh_list):.2f} "
          f"| | {np.median(pf_list):.2f} "
          f"| {np.median(dd_list):.1f} | {np.median(calmar_list):.3f} |")
        L(f"| **均值** | | | {np.mean(ann_list):+.1f} | {np.mean(sh_list):.2f} "
          f"| | {np.mean(pf_list):.2f} "
          f"| {np.mean(dd_list):.1f} | {np.mean(calmar_list):.3f} |")
        L(f"\n盈利币种: {n_pos}/{len(rows)} ({n_pos/len(rows)*100:.0f}%)\n")

    # ── 最佳组合汇总 ──
    L("---\n")
    L("## 三、每个币种的最佳组合\n")

    best_list = []
    for coin, results in grand_results.items():
        best_key = max(results.keys(),
                       key=lambda k: results[k].get("calmar", 0))
        r = results[best_key]
        best_list.append((coin, best_key[0], best_key[1],
                          r["annual_return_pct"], r["profit_factor"],
                          r["max_drawdown_pct"], r["calmar"]))
    best_list.sort(key=lambda x: x[6], reverse=True)

    L("| 币种 | 最佳周期 | 最佳系统 | 年化% | PF | 回撤% | Calmar |")
    L("|---:|---:|---:|---:|---:|---:|---:|")
    for coin, tf, sys, ann, pf, dd, cal in best_list:
        L(f"| {coin} | {tf} | {sys} "
          f"| {ann:+.1f} | {pf:.2f} | {dd:.1f} | {cal:.3f} |")

    L("")

    # ── 频率统计 ──
    tf_counter = Counter(x[1] for x in best_list)
    sys_counter = Counter(x[2] for x in best_list)
    L("### 最佳组合频率分布\n")
    L("**时间框架**: "
      + ", ".join(f"{k} = {v}次" for k, v in tf_counter.most_common()) + "  ")
    L("**系统参数**: "
      + ", ".join(f"{k} = {v}次" for k, v in sys_counter.most_common()) + "\n")

    # ── 组合层面 + 风险特征 ──
    if not USE_LEGACY:
        L("\n".join(portfolio_section(grand_results, out_dir, tag)))

    # ── 写文件 ──
    md_path = os.path.join(out_dir, "turtle_backtest_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存: {md_path}")
    return md_path


# ──────────────────────────────────────────────────
# 组合层面 + 风险特征 (调用 turtle_analytics.py)
# ──────────────────────────────────────────────────

def portfolio_section(grand_results, out_dir, tag, main=("4h", "系统2 (60/20)")):
    """各币种等权组合: 所有 (周期, 系统) 的汇总表 + 主组合的净值图、逐年收益、smile 回归."""
    import turtle_analytics as ta
    L = ["---\n", "## 四、组合层面 (各币种等权, 参考 TSMOM 论文的 diversified 组合)\n"]
    rows = []
    for tf_name, _, _ in TIMEFRAMES:
        for sys_name, _, _ in SYSTEMS:
            pf = ta.build_portfolio({c: grand_results[c].get((tf_name, sys_name))
                                     for c in grand_results}, 365.25)
            rows.append((f"{tf_name} {sys_name}", pf))
    L += ta.portfolio_summary_table(rows) + [""]

    pf = ta.build_portfolio({c: grand_results[c].get(main) for c in grand_results}, 365.25)
    if pf is not None:
        bench = ta.build_benchmark({c: PRICES.get(c) for c in grand_results}, pf["returns"].index)
        L.append(f"### 主组合: {main[0]} × {main[1]}\n")
        L += ta.analysis_section(pf, bench, 365.25, out_dir, str(FIGURES_DIR), tag,
                                 f"{main[0]} {main[1]}")
    return L


# ──────────────────────────────────────────────────
# GARCH 对比报告
# ──────────────────────────────────────────────────

def _print_garch_comparison(results_base, results_garch, failed,
                             start_date, out_dir, garch_mode="rolling",
                             garch_refit_days=21):
    """
    打印 + 保存 GARCH vs 基线 的对比报告.
    results_base / results_garch: {coin: {(tf, sys): result}}
    """
    from datetime import datetime

    # ── 只保留 4h 和 1d (实战可用框架) ──
    focus_tfs = ["4h", "1d"]

    lines = []
    L = lines.append

    L(f"# 海龟系统 — GARCH(1,1) 对比回测报告\n")
    L(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    L(f"> 回测区间: {start_date} ~ 今  ")
    L(f"> 仓位: 1% 资金/ATR | 止损: 2N | 手续费: 0.05%  ")
    L(f"> 对比: 基线 (无过滤) vs GARCH(1,1) 波动率过滤  ")
    if garch_mode == "insample":
        L(f"> GARCH 模式: 全样本拟合 (原版, 有前视偏差)\n")
    else:
        L(f"> GARCH 模式: 滚动重估 (无前视偏差), 每 {garch_refit_days:g} 天重估, "
          f"首年仅训练不过滤\n")
    L("---\n")

    all_coins = sorted(set(list(results_base.keys()) + list(results_garch.keys())))

    # ── 1. 逐币种对比表 ──
    L("## 一、逐币种对比 (4h & 1d)\n")

    improvement_data = []  # (coin, tf, sys, delta_calmar, delta_dd, delta_ann)

    for coin in all_coins:
        r_base = results_base.get(coin, {})
        r_garch = results_garch.get(coin, {})

        L(f"### {coin}\n")
        L("| 周期 | 系统 | 模式 | 交易 | 年化% | PF | 回撤% | Calmar | Δ年化 | ΔCalmar |")
        L("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for tf_name in focus_tfs:
            for sys_name, _, _ in SYSTEMS:
                key = (tf_name, sys_name)
                rb = r_base.get(key)
                rg = r_garch.get(key)

                if rb:
                    L(f"| {tf_name} | {sys_name} | 基线 "
                      f"| {rb['n_trades']} "
                      f"| {rb['annual_return_pct']:+.1f} "
                      f"| {rb['profit_factor']:.2f} "
                      f"| {rb['max_drawdown_pct']:.1f} "
                      f"| {rb['calmar']:.3f} | — | — |")
                if rg:
                    delta_ann = rg['annual_return_pct'] - (rb['annual_return_pct'] if rb else 0)
                    delta_cal = rg['calmar'] - (rb['calmar'] if rb else 0)
                    delta_dd = rg['max_drawdown_pct'] - (rb['max_drawdown_pct'] if rb else 0)
                    L(f"| {tf_name} | {sys_name} | GARCH "
                      f"| {rg['n_trades']} "
                      f"| {rg['annual_return_pct']:+.1f} "
                      f"| {rg['profit_factor']:.2f} "
                      f"| {rg['max_drawdown_pct']:.1f} "
                      f"| {rg['calmar']:.3f} "
                      f"| {delta_ann:+.1f} | {delta_cal:+.3f} |")

                    if rb:
                        improvement_data.append((
                            coin, tf_name, sys_name,
                            delta_cal, delta_dd, delta_ann,
                            rb['calmar'], rg['calmar'],
                            rb['max_drawdown_pct'], rg['max_drawdown_pct'],
                            rb['n_trades'], rg['n_trades'],
                        ))

        L("")

    # ── 2. 汇总统计 ──
    L("---\n")
    L("## 二、GARCH 影响汇总\n")

    if improvement_data:
        improved = [d for d in improvement_data if d[3] > 0]   # delta_calmar > 0
        worsened = [d for d in improvement_data if d[3] < 0]
        unchanged = [d for d in improvement_data if d[3] == 0]

        L(f"- 总对比组数: {len(improvement_data)}")
        L(f"- GARCH 改善: {len(improved)} ({len(improved)/len(improvement_data)*100:.0f}%)")
        L(f"- GARCH 恶化: {len(worsened)} ({len(worsened)/len(improvement_data)*100:.0f}%)")
        L(f"- 无变化: {len(unchanged)}")

        avg_delta_cal = np.mean([d[3] for d in improvement_data])
        avg_delta_ann = np.mean([d[5] for d in improvement_data])
        avg_delta_dd = np.mean([d[4] for d in improvement_data])
        avg_trade_reduction = np.mean([(d[11] - d[10]) / max(d[10], 1) * 100
                                        for d in improvement_data])

        L(f"- 平均 Calmar 变化: {avg_delta_cal:+.3f}")
        d_sh = [results_garch[c][(t, sy)].get("sharpe", 0) - results_base[c][(t, sy)].get("sharpe", 0)
                for c, t, sy, *_ in improvement_data]
        L(f"- 平均夏普变化: {np.mean(d_sh):+.3f}  (夏普改善 {sum(x > 0 for x in d_sh)}/{len(d_sh)} 组)")
        L(f"- 平均年化变化: {avg_delta_ann:+.1f}%")
        L(f"- 平均回撤变化: {avg_delta_dd:+.1f}%")
        L(f"- 平均交易次数变化: {avg_trade_reduction:+.1f}%\n")

        # Top 改善
        if improved:
            improved.sort(key=lambda x: x[3], reverse=True)
            L("### GARCH 改善最大的组合\n")
            L("| 币种 | 周期 | 系统 | 基线Calmar | GARCH Calmar | ΔCalmar | 基线回撤 | GARCH回撤 |")
            L("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for d in improved[:10]:
                L(f"| {d[0]} | {d[1]} | {d[2]} "
                  f"| {d[6]:.3f} | {d[7]:.3f} | {d[3]:+.3f} "
                  f"| {d[8]:.1f}% | {d[9]:.1f}% |")
            L("")

        # Top 恶化
        if worsened:
            worsened.sort(key=lambda x: x[3])
            L("### GARCH 恶化最大的组合\n")
            L("| 币种 | 周期 | 系统 | 基线Calmar | GARCH Calmar | ΔCalmar | 基线回撤 | GARCH回撤 |")
            L("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for d in worsened[:10]:
                L(f"| {d[0]} | {d[1]} | {d[2]} "
                  f"| {d[6]:.3f} | {d[7]:.3f} | {d[3]:+.3f} "
                  f"| {d[8]:.1f}% | {d[9]:.1f}% |")
            L("")

    if failed:
        L(f"\n跳过 (数据不足/出错): {', '.join(failed)}\n")

    # ── 同时也打印到终端 ──
    report_text = "\n".join(lines)
    print(f"\n\n{'#'*80}")
    print(report_text)

    # ── 保存 Markdown ──
    md_path = os.path.join(out_dir, "turtle_garch_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n对比报告已保存: {md_path}")

    # ── 同时生成基线和 GARCH 的完整报告 ──
    if results_base:
        md1 = os.path.join(out_dir, "turtle_backtest_report_baseline.md")
        generate_markdown_report(results_base, failed, start_date, out_dir, tag="crypto")
        os.replace(os.path.join(out_dir, "turtle_backtest_report.md"), md1)   # replace: 目标已存在也能覆盖 (Windows 上 rename 会报错)
        print(f"基线报告: {md1}")

    if results_garch:
        md2 = os.path.join(out_dir, "turtle_backtest_report_garch.md")
        generate_markdown_report(results_garch, failed, start_date, out_dir, tag="crypto_garch")
        os.replace(os.path.join(out_dir, "turtle_backtest_report.md"), md2)
        print(f"GARCH报告: {md2}")


# ──────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="海龟系统 — 多时间框架回测")
    parser.add_argument("--symbol", default="BTC/USDT:USDT",
                        help="交易对 (默认: BTC/USDT:USDT)")
    parser.add_argument("--start", default="2022-01-01",
                        help="回测起始日期 (默认: 2022-01-01)")
    parser.add_argument("--top", type=int, default=0,
                        help="批量测试市值前 N 的币种 (如 --top 20)")
    parser.add_argument("--coins", type=str, default="",
                        help="指定币种列表, 逗号分隔 (如 --coins BTC,ETH,SOL)")
    parser.add_argument("--refresh", action="store_true",
                        help="强制重新拉取数据 (忽略缓存)")
    parser.add_argument("--garch", action="store_true",
                        help="启用 GARCH(1,1) 波动率过滤器 (需要 arch 库)")
    parser.add_argument("--garch-mode", choices=["rolling", "insample"], default="rolling",
                        help="rolling=滚动重估, 无前视偏差 (默认) / insample=原版全样本拟合")
    parser.add_argument("--garch-refit-days", type=float, default=21,
                        help="滚动模式下每隔多少天重估一次 GARCH 参数 (默认 21)")
    parser.add_argument("--engine", choices=["new", "legacy"], default="new",
                        help="new=新引擎 (逐根盯市/夏普/跳空/滑点, 默认) / legacy=旧版引擎, 用于对比")
    parser.add_argument("--slippage", type=float, default=None,
                        help="每次成交的滑点比例 (默认 0.0003 = 3 个基点)")
    parser.add_argument("--garch-compare", action="store_true",
                        help="同时跑有/无 GARCH 并生成对比报告")
    args = parser.parse_args()

    global USE_LEGACY, SLIPPAGE
    USE_LEGACY = args.engine == "legacy"
    if args.slippage is not None:
        SLIPPAGE = args.slippage

    # 如果 --refresh, 删除本脚本用的 15 分钟K线缓存
    # (缓存目录是各项目共用的, 只删 *_15m_*.csv, 不动别的项目的文件)
    if args.refresh:
        removed = list(OKX_CACHE.glob("*_15m_*.csv")) if OKX_CACHE.exists() else []
        for f in removed:
            f.unlink()
        print(f"已清除 {len(removed)} 个 15 分钟K线缓存, 将重新拉取数据.\n")

    out_dir = str(REPORTS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    garch_label = " + GARCH过滤" if args.garch else ""

    # ══════════════════════════════════════════════════
    # GARCH 对比模式: 同时跑有/无 GARCH
    # ══════════════════════════════════════════════════
    if args.garch_compare:
        # 确定币种列表
        if args.coins:
            coin_list = [c.strip().upper() for c in args.coins.split(",")]
            symbols = [f"{c}/USDT:USDT" for c in coin_list]
        elif args.top > 0:
            print(f"正在获取市值前 {args.top} 的币种列表...\n")
            ex = create_exchange()
            symbols = fetch_top_symbols(exchange=ex, top_n=args.top)
            if not symbols:
                print("[!] 无法获取币种列表"); sys.exit(1)
        else:
            symbols = [args.symbol]

        coin_names = [s.split("/")[0] for s in symbols]
        print(f"\n{'='*80}")
        print(f"海龟交易系统 — GARCH 对比回测")
        print(f"币种: {', '.join(coin_names)}")
        print(f"仓位: 1% 资金/ATR  |  止损: 2N  |  手续费: 0.05%  |  起始: {args.start}")
        print(f"{'='*80}")

        results_base = {}    # 无 GARCH
        results_garch = {}   # 有 GARCH
        failed = []

        for idx, (sym, coin) in enumerate(zip(symbols, coin_names), 1):
            print(f"\n[{idx}/{len(symbols)}] ── {coin} ──")
            try:
                # 1. 不带 GARCH
                print(f"  --- 基线 (无GARCH) ---")
                r_base = run_single_coin(sym, args.start, verbose=True,
                                          use_garch=False)
                if r_base:
                    results_base[coin] = r_base

                # 2. 带 GARCH
                print(f"  --- GARCH 过滤 ---")
                r_garch = run_single_coin(sym, args.start, verbose=True,
                                            use_garch=True, garch_mode=args.garch_mode,
                                            garch_refit_days=args.garch_refit_days)
                if r_garch:
                    results_garch[coin] = r_garch

                if not r_base and not r_garch:
                    failed.append(coin)
            except Exception as e:
                failed.append(coin)
                print(f"  [!] {coin} 出错: {e}")

        # ── 生成对比报告 ──
        _print_garch_comparison(results_base, results_garch, failed,
                                args.start, out_dir, garch_mode=args.garch_mode,
                                garch_refit_days=args.garch_refit_days)
        return

    # ══════════════════════════════════════════════════
    # 单币模式
    # ══════════════════════════════════════════════════
    if args.top <= 0 and not args.coins:
        coin_name = args.symbol.split("/")[0]
        print(f"\n{'='*80}")
        print(f"海龟交易系统 — 多时间框架回测{garch_label}  [{coin_name}]")
        print(f"仓位: 1% 资金/ATR  |  止损: 2N  |  手续费: 0.05%")
        print(f"{'='*80}\n")

        results = run_single_coin(args.symbol, args.start, verbose=True,
                                  use_garch=args.garch, garch_mode=args.garch_mode,
                                  garch_refit_days=args.garch_refit_days)
        if results:
            print_single_coin_table(results, coin_name)
            save_single_chart(results, coin_name, out_dir)
        print("\n回测完成。")
        return

    # ══════════════════════════════════════════════════
    # 多币模式: --top N 或 --coins
    # ══════════════════════════════════════════════════
    if args.coins:
        coin_list = [c.strip().upper() for c in args.coins.split(",")]
        symbols = [f"{c}/USDT:USDT" for c in coin_list]
    else:
        print(f"正在获取市值前 {args.top} 的币种列表...\n")
        ex = create_exchange()
        symbols = fetch_top_symbols(exchange=ex, top_n=args.top)

        if not symbols:
            print("[!] 无法获取币种列表, 请检查网络/代理设置")
            sys.exit(1)

    coin_names = [s.split("/")[0] for s in symbols]
    print(f"将测试 {len(symbols)} 个币种: {', '.join(coin_names)}")
    print(f"\n{'='*80}")
    print(f"海龟交易系统 — 多币种 × 多时间框架{garch_label}  (15min K线重采样)")
    print(f"仓位: 1% 资金/ATR  |  止损: 2N  |  手续费: 0.05%  |  起始: {args.start}")
    print(f"{'='*80}")

    grand_results = {}   # {coin: {(tf, sys): result}}
    failed = []

    for idx, (sym, coin) in enumerate(zip(symbols, coin_names), 1):
        print(f"\n[{idx}/{len(symbols)}] ── {coin} ──")
        try:
            results = run_single_coin(sym, args.start, verbose=True,
                                      use_garch=args.garch, garch_mode=args.garch_mode,
                                      garch_refit_days=args.garch_refit_days)
            if results:
                grand_results[coin] = results
                # 每个币也打表
                print_single_coin_table(results, coin)
                save_single_chart(results, coin, out_dir)
            else:
                failed.append(coin)
                print(f"  [!] {coin} 数据获取失败, 跳过")
        except Exception as e:
            failed.append(coin)
            print(f"  [!] {coin} 出错: {e}")

    if not grand_results:
        print("\n[!] 没有成功回测的币种")
        sys.exit(1)

    # ══════════════════════════════════════════════════
    # 多币种汇总: 对每个 (时间框架 × 系统) 横向对比
    # ══════════════════════════════════════════════════
    print(f"\n\n{'#'*80}")
    print(f"{'多币种横向对比':^80}")
    print(f"{'#'*80}")

    # 只打印最有参考价值的几个组合
    key_combos = [
        ("4h",  "系统2 (60/20)"),
        ("4h",  "系统1 (20/10)"),
        ("1h",  "系统2 (60/20)"),
        ("1d",  "系统2 (60/20)"),
    ]
    for tf, sys in key_combos:
        print_multi_coin_summary(grand_results, tf, sys)

    # ── 终极汇总: 每个币的最佳时间框架 ──
    print(f"\n{'='*80}")
    print(f"  每个币种的最佳组合 (按 Calmar 排序)")
    print(f"{'='*80}")
    print(f"{'币种':>6} {'最佳周期':>8} {'最佳系统':>12} {'年化%':>7}"
          f" {'PF':>6} {'回撤%':>8} {'Calmar':>7}")
    print("─" * 80)

    best_list = []
    for coin, results in grand_results.items():
        best_key = max(results.keys(),
                       key=lambda k: results[k].get("calmar", 0))
        r = results[best_key]
        best_list.append((coin, best_key[0], best_key[1],
                          r["annual_return_pct"], r["profit_factor"],
                          r["max_drawdown_pct"], r["calmar"]))

    best_list.sort(key=lambda x: x[6], reverse=True)
    for coin, tf, sys, ann, pf, dd, cal in best_list:
        print(f"{coin:>6} {tf:>8} {sys:>12}"
              f" {ann:>+6.1f} {pf:>5.2f} {dd:>7.1f} {cal:>6.3f}")
    print("─" * 80)

    # 统计最佳组合中各时间框架出现的频率
    from collections import Counter
    tf_counter = Counter(x[1] for x in best_list)
    sys_counter = Counter(x[2] for x in best_list)
    print(f"\n  最佳时间框架分布: "
          + ", ".join(f"{k}={v}" for k, v in tf_counter.most_common()))
    print(f"  最佳系统分布:     "
          + ", ".join(f"{k}={v}" for k, v in sys_counter.most_common()))

    if failed:
        print(f"\n  跳过 (数据不足/出错): {', '.join(failed)}")

    # ── 生成 Markdown 报告 ──
    generate_markdown_report(grand_results, failed, args.start, out_dir)

    print(f"\n多币种回测全部完成。共 {len(grand_results)} 个币种, "
          f"跳过 {len(failed)} 个。")


if __name__ == "__main__":
    main()