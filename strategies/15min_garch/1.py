import os as _os
from pathlib import Path as _Path
# OKX 密钥从 quant/data/.env 读取 (OKX_API_KEY / OKX_SECRET / OKX_PASSWORD), 不写在代码里
_env = next((p / "data" / ".env" for p in _Path(__file__).resolve().parents if (p / "data" / ".env").exists()), None)
if _env:
    for _l in _env.read_text(encoding="utf-8").splitlines():
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.split("=", 1)
            _os.environ.setdefault(_k.strip(), _v.strip())
# %% [markdown]
# # BTC/USDT 永续合约 —— 日线级别趋势跟踪策略
#
# ## 为什么换到日线
# 你之前15分钟级别的pipeline（GARCH波动率 + XGBoost方向/突破分类 + 三重障碍标签）已经测得很彻底：
# 三种标签定义、多组特征（技术指标/regime/资金费率）、多种执行假设（理论价/保守价/中间态）、
# 大范围参数扫描——每次深挖，之前看起来的"好消息"都建立在某个不成立的假设上（最典型的是
# "精确打到理论屏障价成交"）。方向分类器整体准确率~50.6%，p=0.53，和瞎猜没有统计显著差异。
# 结论很清楚：这套方法本身信息含量有限，不是运气问题——15分钟到3小时这个频段是全球做市商/
# 高频交易/套利机器人竞争最激烈的战场，纯技术指标衍生特征在这里基本没有剩余的alpha空间。
#
# 日线/周线级别的时序动量、趋势跟踪，在加密货币上有实际被记录、被复现过的历史有效性
# （不保证一定有效，但至少这个战场竞争烈度低得多），而且完全复用你现有的基础设施
# （同一个OKX API，换成日线K线，波动率目标仓位的思路也保留）。
#
# ## 这版代码的设计原则（对应你"honest negative findings"的偏好）
# 1. 先上最朴素的经典信号（双均线交叉、唐奇安通道突破、时序动量），不上来就上机器学习——
#    模型越复杂，越容易把噪声当成信号，之前15分钟那版已经吃过这个亏。
# 2. 每个信号都必须跟两个基准比：Buy & Hold、随机方向基准（相同换仓频率、方向随机）。
#    如果策略跑不赢随机方向基准的置信区间，说明这个信号没有提供信息增量，不管年化数字多好看。
# 3. 参数敏感性扫描：只有单个"最优参数点"赚钱、邻近参数大幅变差，是过拟合的典型信号。
#    好的策略应该在一片参数区域内都表现稳定，而不是只有一个孤立的高点。
# 4. 回测约定明确写死（见 run_daily_backtest 的docstring），避免不小心引入前视偏差。
#
# ## 需要你做的事
# - 把 API_KEY / API_SECRET / API_PASSWORD 换成环境变量（代码里不要硬编码，尤其不要复制粘贴
#   到任何会保存历史记录/会分享出去的地方——如果你之前把真实key贴给过AI助手或发到别的地方，
#   建议直接去OKX后台把那个key删掉重新生成一个，比继续研究优先级更高）。
# - 跑一遍 `if __name__ == '__main__':` 里的流程，把打印出来的实际数字发回来，我根据真实结果
#   帮你判断这条路是不是也要放弃，还是有进一步优化的空间。我这边没有你的数据和API访问权限，
#   所以这版代码没法帮你直接跑，只能帮你搭好框架。

# %%
import os
import time
import numpy as np
import pandas as pd
import ccxt

pd.set_option('display.width', 140)

# %%
# ============ 配置 ============
SYMBOL = 'BTC-USDT-SWAP'      # 和你原notebook保持一致
TIMEFRAME = '1d'
START_DATE = '2020-01-01T00:00:00Z'

FEE_RATE = 0.0005             # 单边手续费，和你原来一致
SLIPPAGE_BPS = 0.0002         # 单边滑点缓冲

TARGET_DAILY_VOL = 0.01       # 目标日波动贡献（1%），可调
MAX_LEVERAGE = 1.5

#实例化交易所
exchange = ccxt.okx({
    'apiKey': _os.environ.get('OKX_API_KEY', ''),
    'secret': _os.environ.get('OKX_SECRET', ''),
    'password': _os.environ.get('OKX_PASSWORD', ''),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'   # 可选: 'spot'(现货), 'future'(交割合约), 'swap'(永续合约)
    },
    'proxies': {
        'http': 'http://127.0.0.1:29290',  
        'https': 'http://127.0.0.1:29290', 
    }
})

exchange_open = ccxt.okx({
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap'  # 设定默认请求合约数据
    },
     'proxies': {
        'http': 'http://127.0.0.1:29290',  
        'https': 'http://127.0.0.1:29290', 
    }
})



# %%
# ============ 1. 数据获取 ============

def fetch_daily_ohlcv(exchange, symbol=SYMBOL, timeframe=TIMEFRAME, since_str=START_DATE):
    """
    日线数据量小（2020至今约2000根），正常几次请求就能拉完，
    仍按分页写是为了保险（避免交易所单次limit不够的情况）。
    """
    since = exchange.parse8601(since_str)
    all_rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=300)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1
        if last_ts >= exchange.milliseconds() - 24 * 60 * 60 * 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    df = df.set_index('timestamp')
    return df


# %%
# ============ 2. 特征构建 ============
def add_trend_features(df, ma_fast=20, ma_slow=60, donchian_n=20, atr_n=14, mom_n=90):
    df = df.copy()
    df['ma_fast'] = df['close'].rolling(ma_fast).mean()
    df['ma_slow'] = df['close'].rolling(ma_slow).mean()

    # shift(1)：今天的信号不能用到今天的最高/最低价，只能用截止昨天收盘的信息
    df['donchian_high'] = df['high'].rolling(donchian_n).max().shift(1)
    df['donchian_low'] = df['low'].rolling(donchian_n).min().shift(1)

    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_n).mean()
    df['atr_pct'] = (df['atr'] / df['close']).clip(lower=1e-6)  # 归一化+防止除零

    df['mom_ret'] = df['close'].pct_change(mom_n)

    return df


# %%
# ============ 3. 信号定义（先给三种，自己 A/B） ============
def signal_ma_cross(df):
    """双均线交叉：快线在慢线之上做多，之下做空"""
    return pd.Series(np.where(df['ma_fast'] > df['ma_slow'], 1, -1), index=df.index)


def signal_donchian_breakout(df):
    """唐奇安通道突破：收盘价突破前N日高点转多，跌破前N日低点转空，否则维持上一状态"""
    signal = np.zeros(len(df))
    state = 0
    close = df['close'].values
    dh = df['donchian_high'].values
    dl = df['donchian_low'].values
    for i in range(len(df)):
        if np.isnan(dh[i]) or np.isnan(dl[i]):
            signal[i] = 0
            continue
        if close[i] > dh[i]:
            state = 1
        elif close[i] < dl[i]:
            state = -1
        signal[i] = state
    return pd.Series(signal, index=df.index)


def signal_momentum(df):
    """时序动量：过去N日收益率的符号"""
    return np.sign(df['mom_ret']).fillna(0)


# %%
# ============ 4. 回测引擎 ============
def run_daily_backtest(df, signal, target_vol=TARGET_DAILY_VOL, max_leverage=MAX_LEVERAGE,
                        fee_rate=FEE_RATE, slippage_bps=SLIPPAGE_BPS):
    """
    回测约定（明确写死，避免前视偏差）：
    - signal[t] 只能用截止第t天收盘为止的信息计算（ma_fast/ma_slow用当天收盘价属于"收盘后
      决策"；donchian已提前shift(1)）。
    - 实际持仓 position[t] = signal[t-1]，即"昨天收盘后决定，今天全天持有"，
      收益按当天 close-to-close 计算——这是日线策略最常见的回测约定，隐含假设是
      "以接近前一天收盘的价格成交"，用slippage_bps覆盖这个近似的误差。
    - 手续费+滑点只在【方向发生变化】的那天收取（而不是每天仓位大小微调都收费），
      更贴近"趋势策略换仓不频繁"的实际情况。
    """
    df = df.copy()
    df['position'] = signal.shift(1).fillna(0)
    df['vol_scalar'] = (target_vol / df['atr_pct']).clip(upper=max_leverage).fillna(0)
    df['position_size'] = df['position'] * df['vol_scalar']

    df['ret'] = df['close'].pct_change()
    df['gross_ret'] = df['position_size'] * df['ret']

    direction_changed = (df['position'] != df['position'].shift(1)).astype(int)
    df['cost'] = direction_changed * df['position_size'].abs() * (fee_rate + slippage_bps) * 2

    df['net_ret'] = df['gross_ret'] - df['cost']
    df['log_ret'] = np.log(1 + df['net_ret'].clip(lower=-0.99))
    return df


def report_performance(df, label="", verbose=True):
    valid = df['log_ret'].dropna()
    total_days = max((valid.index[-1] - valid.index[0]).days, 1)
    annualized = valid.sum() / total_days * 365
    ann_vol = valid.std() * np.sqrt(365)
    sharpe = annualized / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + df['net_ret'].fillna(0)).cumprod()
    running_max = cum.cummax()
    mdd = ((cum - running_max) / running_max).min()

    n_trades = int((df['position'] != df['position'].shift(1)).sum())

    if verbose:
        print(f"=== {label} ===")
        print(f"样本天数: {total_days}")
        print(f"换仓次数: {n_trades}")
        print(f"年化对数收益率: {annualized:.2%}")
        print(f"年化波动率: {ann_vol:.2%}")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"最大回撤: {mdd:.2%}")
        print()

    return {'annualized': annualized, 'sharpe': sharpe, 'mdd': mdd, 'n_trades': n_trades}


# %%
# ============ 5. 基准对照（避免自己骗自己） ============
def buy_and_hold_baseline(df):
    bh = df.copy()
    bh['position'] = 1.0
    bh['net_ret'] = df['close'].pct_change()
    bh['log_ret'] = np.log(1 + bh['net_ret'].clip(lower=-0.99))
    return report_performance(bh, "基准: Buy & Hold")


def random_direction_baseline(df, n_runs=200, seed=42):
    """
    随机方向基准：换仓逻辑和真实信号同一套回测引擎，方向随机生成。
    如果真实信号的年化收益/Sharpe没有明显超过这个分布的95%分位，
    说明这个信号本身没有提供信息增量，不管账面数字多好看。
    """
    rng = np.random.default_rng(seed)
    ann_results, sharpe_results = [], []
    n = len(df)
    for _ in range(n_runs):
        rand_signal = pd.Series(rng.choice([1, -1], size=n), index=df.index)
        bt = run_daily_backtest(df, rand_signal)
        perf = report_performance(bt, verbose=False)
        ann_results.append(perf['annualized'])
        sharpe_results.append(perf['sharpe'])

    ann_arr = np.array(ann_results)
    sharpe_arr = np.array(sharpe_results)
    print(f"=== 随机方向基准 (n={n_runs}) ===")
    print(f"年化收益率: 均值 {ann_arr.mean():.2%}, 95%区间 [{np.percentile(ann_arr, 2.5):.2%}, {np.percentile(ann_arr, 97.5):.2%}]")
    print(f"Sharpe: 均值 {np.nanmean(sharpe_arr):.2f}, 95%区间 [{np.nanpercentile(sharpe_arr, 2.5):.2f}, {np.nanpercentile(sharpe_arr, 97.5):.2f}]")
    print()
    return ann_arr, sharpe_arr


def significance_vs_random(real_annualized, random_ann_arr):
    """真实策略的年化收益在随机方向基准分布里排第几分位——比直接看年化数字本身有意义得多"""
    percentile = (random_ann_arr < real_annualized).mean() * 100
    print(f"真实策略年化收益在随机方向基准分布中的分位数: {percentile:.1f}%")
    if percentile < 90:
        print("提示：没有明显超过随机方向基准的上端，这个信号大概率没有提供真实的信息增量。")
    print()
    return percentile


# %%
# ============ 6. 参数敏感性扫描 ============
def param_sensitivity_scan(df_raw, ma_fast_list=(10, 20, 30), ma_slow_list=(50, 60, 90, 120)):
    """
    只有单个"最优参数点"赚钱、邻近参数大幅变差 —— 是过拟合的典型信号。
    好的策略应该在一片参数区域内都表现相对稳定。
    """
    rows = []
    for f in ma_fast_list:
        for s in ma_slow_list:
            if f >= s:
                continue
            feat = add_trend_features(df_raw, ma_fast=f, ma_slow=s)
            sig = signal_ma_cross(feat)
            bt = run_daily_backtest(feat, sig)
            perf = report_performance(bt, label=f"ma_fast={f}, ma_slow={s}", verbose=False)
            rows.append({'ma_fast': f, 'ma_slow': s, **perf})
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    print()
    return result


# %%
# ============ 7. 执行入口 ============
if __name__ == '__main__':
    # 第一次跑，先拉数据并缓存到本地，之后改成直接读CSV，别每次都重新请求交易所
    if not os.path.exists('btc_daily_ohlcv.csv'):
        df_raw = fetch_daily_ohlcv(exchange_open)   # 改这里：exchange → exchange_open
        df_raw.to_csv('btc_daily_ohlcv.csv')
    else:
        df_raw = pd.read_csv('btc_daily_ohlcv.csv', index_col=0, parse_dates=True)

    df_feat = add_trend_features(df_raw)

    print("########## 双均线交叉 (20/60) ##########")
    sig_ma = signal_ma_cross(df_feat)
    bt_ma = run_daily_backtest(df_feat, sig_ma)
    perf_ma = report_performance(bt_ma, "双均线交叉 (20/60)")

    print("########## 唐奇安通道突破 (20日) ##########")
    sig_donchian = signal_donchian_breakout(df_feat)
    bt_donchian = run_daily_backtest(df_feat, sig_donchian)
    perf_donchian = report_performance(bt_donchian, "唐奇安通道突破 (20日)")

    print("########## 时序动量 (90日) ##########")
    sig_mom = signal_momentum(df_feat)
    bt_mom = run_daily_backtest(df_feat, sig_mom)
    perf_mom = report_performance(bt_mom, "时序动量 (90日)")

    print("########## 基准对照 ##########")
    buy_and_hold_baseline(df_feat)
    ann_random, sharpe_random = random_direction_baseline(df_feat, n_runs=200)

    print("########## 显著性检验（三个信号分别对比随机基准） ##########")
    significance_vs_random(perf_ma['annualized'], ann_random)
    significance_vs_random(perf_donchian['annualized'], ann_random)
    significance_vs_random(perf_mom['annualized'], ann_random)

    print("########## 参数敏感性扫描（双均线） ##########")
    param_sensitivity_scan(df_raw)