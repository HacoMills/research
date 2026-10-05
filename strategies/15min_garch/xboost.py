import os as _os
from pathlib import Path as _Path
# OKX 密钥从 quant/data/.env 读取 (OKX_API_KEY / OKX_SECRET / OKX_PASSWORD), 不写在代码里
_env = next((p / "data" / ".env" for p in _Path(__file__).resolve().parents if (p / "data" / ".env").exists()), None)
if _env:
    for _l in _env.read_text(encoding="utf-8").splitlines():
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.split("=", 1)
            _os.environ.setdefault(_k.strip(), _v.strip())
# %%
#环境
import numpy as np
test_arr = np.array([1, 2, 3, 4])
print(f"✅ 基础运算测试: [1,2,3,4] 求和结果 = {test_arr.sum()} (期望值: 10)")

import ccxt
print(ccxt.exchanges)

import pandas as pd
import pandas_ta as ta

import time
import urllib.request
proxies = urllib.request.getproxies()
print("系统当前代理为:", proxies)

import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from arch import arch_model
from tqdm import tqdm

# %%
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
#ccxt的help指令
print(exchange.features)

# %%
#爬15min数据
def fetch_large_ohlcv_okx(exchange, symbol='BTC/USDT', timeframe='15m', target_limit=10000):
    all_ohlcv = []
    limit_per_request = 100  # OKX 的单次上限是 100
    tf_ms = 15 * 60 * 1000   # 15分钟的毫秒数
    
    # 这里直接使用传进来的 exchange 对象
    current_time = exchange.milliseconds()
    since = current_time - (target_limit * tf_ms)
    
    print(f"正在从 OKX 抓取 {symbol} 的 {timeframe} 数据，目标约 {target_limit} 条...")
    
    while len(all_ohlcv) < target_limit:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_request)
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            print(f"已成功获取累计条数: {len(all_ohlcv)}")
            
            time.sleep(exchange.rateLimit / 1000)
            
            if len(ohlcv) < limit_per_request:
                break
                
        except Exception as e:
            print(f"抓取数据时发生错误: {e}")
            time.sleep(1)
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    if len(df) > target_limit:
        df = df.tail(target_limit)
        
    print(f"\n数据抓取完毕！最终得到 {len(df)} 行有效数据。")
    return df

df_15m = fetch_large_ohlcv_okx(exchange=exchange_open, symbol='BTC/USDT', timeframe='15m', target_limit=10000)

# === 4. 打印预览 ===
print("\n数据预览：")
print(df_15m.head())
print(df_15m.tail())

# %%
#存储数据
df_15m.to_csv('okx_btc_usdt_15m.csv')

# %%
#ADF检验,ARCH效应检验

#df_15m = pd.read_csv('okx_btc_usdt_15m.csv', index_col='timestamp', parse_dates=True) 

# ==========================================
# 1. 数据准备：计算对数收益率
# ==========================================
# 金融学中通常使用“对数收益率”来进行时间序列分析
df_15m['log_return'] = np.log(df_15m['close'] / df_15m['close'].shift(1))

# 剔除第一行的 NaN 值
returns = df_15m['log_return'].dropna()

# ==========================================
# 2. 平稳性检验：ADF 检验
# ==========================================
print("=== 1. ADF 平稳性检验 (Augmented Dickey-Fuller Test) ===")
adf_result = adfuller(returns)
print(f"ADF 统计量 (T-statistic): {adf_result[0]:.4f}")
print(f"P-Value: {adf_result[1]:.4e}")
print("结论: ", end="")
if adf_result[1] < 0.05:
    print("P-Value < 0.05，拒绝原假设，对数收益率序列是【平稳的】。")
else:
    print("P-Value >= 0.05，不能拒绝原假设，序列是【不平稳的】。")
print("-" * 50)

# ==========================================
# 3. ARCH 效应检验：Engle's ARCH-LM 检验
# ==========================================
print("=== 2. ARCH 效应检验 (Engle's ARCH-LM Test) ===")
# 通常我们对收益率去均值后（即残差）进行 ARCH 检验
resid = returns - returns.mean()
# 检验滞后项，这里我们看过去 15 个周期（约 3.75 小时）
arch_test = het_arch(resid, nlags=15) 
print(f"LM 统计量 (LM Statistic): {arch_test[0]:.4f}")
print(f"P-Value: {arch_test[1]:.4e}")
print("结论: ", end="")
if arch_test[1] < 0.05:
    print("P-Value < 0.05，拒绝原假设，存在显著的【ARCH效应 (条件异方差/波动率聚集)】。")
    print("👉 非常适合使用 GARCH 模型进行下一步建模！")
else:
    print("P-Value >= 0.05，不能拒绝原假设，不存在明显的 ARCH 效应。")
print("-" * 50)

# ==========================================
# 4. 可视化观察：波动率聚集现象
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 图1：对数收益率（看是否平稳围绕 0 波动）
axes[0].plot(returns, color='blue', linewidth=0.5)
axes[0].set_title('BTC/USDT 15分钟对数收益率 (Log Returns)')
axes[0].grid(True, alpha=0.3)

# 图2：对数收益率的平方（直观感受波动率聚集）
axes[1].plot(returns**2, color='red', linewidth=0.5)
axes[1].set_title('对数收益率的平方 (Squared Log Returns - 观察波动率聚集)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# 1. 准备数据：计算对数收益率，并放大 100 倍变成百分比 (%)
# 💡 注意：放大 100 倍是行业惯例，因为收益率数值太小（如 0.0001），GARCH 底层优化器很难收敛
df_15m['log_return'] = np.log(df_15m['close'] / df_15m['close'].shift(1))
returns = df_15m['log_return'].dropna() * 100  

def calculate_rolling_garch(returns, window_size=2000, refit_every=96):
    print(f"启动 GARCH 移动窗口计算 (总数据量: {len(returns)})...")
    out_of_sample_vol = pd.Series(index=returns.index, dtype=float)
    out_of_sample_mu = pd.Series(index=returns.index, dtype=float)
    
    n_samples = len(returns)
    
    for i in tqdm(range(window_size, n_samples, refit_every)):
        # 严格只用过去的数据
        train_window = returns.iloc[i - window_size : i]
        am = arch_model(train_window, mean='Constant', vol='Garch', p=1, q=1, dist='Normal')
        
        try:
            res = am.fit(disp='off', show_warning=False)
        except:
            continue
            
        steps_to_predict = min(refit_every, n_samples - i)
        forecasts = res.forecast(horizon=steps_to_predict, align='origin')
        
        # 提取真正的样本外预测值
        pred_vol = np.sqrt(forecasts.variance.iloc[-1].values)
        pred_mu = forecasts.mean.iloc[-1].values
        
        out_of_sample_vol.iloc[i : i + steps_to_predict] = pred_vol
        out_of_sample_mu.iloc[i : i + steps_to_predict] = pred_mu

    return out_of_sample_vol.dropna(), out_of_sample_mu.dropna()

# 1. 运行核心函数 (大概需要等待 1-2 分钟，看进度条)
oos_vol, oos_mu = calculate_rolling_garch(returns, window_size=2000, refit_every=96)

# ==========================================
# 2 & 4 合并: 计算严格样本外的 99% VaR (做多和做空)
# ==========================================
# 设定做多的 99% 风险底线 (防暴跌)
alpha_long = 0.01
z_score_long = stats.norm.ppf(alpha_long) 
oos_var_long = oos_mu + oos_vol * z_score_long  # 修复: 这里改用 oos_mu 和 oos_vol

# 设定做空的 99% 风险底线 (防暴涨)
alpha_short = 0.99
z_score_short = stats.norm.ppf(alpha_short) 
oos_var_short = oos_mu + oos_vol * z_score_short # 修复: 这里改用 oos_mu 和 oos_vol

# ==========================================
# 3. 把安全的数据合并到主 DataFrame 中
# ==========================================
df_15m['cond_vol_oos'] = oos_vol
df_15m['var_long_oos'] = oos_var_long
df_15m['var_short_oos'] = oos_var_short  # 修复: 把做空VaR也存进去，供后续画图用

# 砍掉前面没有波动率预测结果的 2000 行（即前20天）
df_ready = df_15m.dropna().copy()
print(f"\n✅ 移动窗口处理完毕！最终可用于特征工程的数据行数: {len(df_ready)}")


# ==========================================
# 5. 可视化：让“动态风控”变得直观
# ==========================================
# 为了看清楚细节，我们只截取最后 500 根 K 线（约 5 天的数据）来画图
plot_returns = df_ready['log_return'].tail(500)
plot_var_long = df_ready['var_long_oos'].tail(500)
plot_var_short = df_ready['var_short_oos'].tail(500) # 修复: 定义 plot_var_short

plt.figure(figsize=(14, 7))

# 画出实际收益率（蓝色柱子/线）
plt.plot(plot_returns.index, plot_returns, color='blue', alpha=0.6, label='BTC/USDT 实际收益率 (%)')

# 画出动态 VaR 预警线（红色实线 / 绿色实线）
plt.plot(plot_var_long.index, plot_var_long, color='red', linewidth=2, label='99% 动态 VaR (做多防线)')
plt.plot(plot_var_short.index, plot_var_short, color='green', linewidth=2, label='99% 动态 VaR (做空防线)')

# 填充 VaR 以下的极端危险区域
plt.fill_between(plot_var_long.index, plot_var_long, plot_var_long.min() - 1, color='red', alpha=0.1)
plt.fill_between(plot_var_short.index, plot_var_short, plot_var_short.max() + 1, color='green', alpha=0.1)

# 找出“突破做多防线”的黑天鹅时刻（实际亏损超过了 99% VaR 的点）
exceptions = plot_returns[plot_returns < plot_var_long]
if len(exceptions) > 0:
    plt.scatter(exceptions.index, exceptions, color='black', s=50, zorder=5, label='破防点 (VaR Exceptions)')

plt.title('BTC/USDT 15分钟级别 动态 VaR (GARCH 模型预测)', fontsize=16)
plt.ylabel('收益率 (%)', fontsize=12)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# --- 计算最后一根 K 线的实时风险 ---
# ==========================================
last_vol = df_ready['cond_vol_oos'].iloc[-1]   # 修复: 从 df_ready 中取最后一个值
last_var = df_ready['var_long_oos'].iloc[-1]   # 修复: 从 df_ready 中取最后一个值

print(f"\n💡 【实时风控警报】根据最后一根 K 线预测:")
print(f"当前 15 分钟的预期波动率 (Sigma) 为: {last_vol:.2f}%")
print(f"99% 置信度下的 动态 VaR 为: {last_var:.2f}%")
print(f"解释: 在接下来的 15 分钟内，你有 99% 的概率，资产亏损不会超过 {abs(last_var):.2f}%！")

# %%
df_ready = df_15m.dropna().copy() # 基于上一步清洗后的数据

# 将 GARCH 输出的百分比 (如 2.5%) 转换为小数 (0.025)，方便计算
df_ready['garch_vol_oos'] = df_ready['cond_vol_oos'] / 100
df_ready['garch_var_oos'] = df_ready['var_long_oos'] / 100

# ==========================================
# 2. 特征工程 (保持不变，但使用 OOS 样本外数据)
# ==========================================
def create_features_oos(data):
    df = data.copy()
    
    df['return_t-1'] = df['close'].pct_change(1)
    df['volume_change'] = df['volume'].pct_change(1)
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    
    # 24H Z-Score (使用无未来函数的 vol)
    window_24h = 96
    rolling_mean = df['garch_vol_oos'].rolling(window=window_24h).mean()
    rolling_std = df['garch_vol_oos'].rolling(window=window_24h).std()
    df['garch_vol_zscore'] = (df['garch_vol_oos'] - rolling_mean) / rolling_std
    
    df['delta_garch_vol'] = df['garch_vol_oos'].diff(1)
    df['return_1H'] = df['close'].pct_change(4)
    df['rsi_1D'] = ta.rsi(df['close'], length=96)
    
    return df.dropna()

df_features = create_features_oos(df_ready)

# ==========================================
# 3. 核心升级：动态三重屏障标签 (Triple-Barrier Method)
# ==========================================
def apply_triple_barrier(df, horizon=12, var_multiplier=1.0):
    """
    horizon: 时间屏障，未来 12 根 K 线 (3小时)
    var_multiplier: 动态屏障宽度乘数。用 GARCH_VaR 乘以这个系数作为止盈止损线。
    """
    print(f"\n正在计算三重屏障标签 (时间屏障={horizon}根K线)...")
    
    # 预分配 Label 列，默认 1 (震荡/观望)
    labels = np.ones(len(df))
    
    close_prices = df['close'].values
    # VaR 是负数，取绝对值作为边界宽度
    garch_var = df['garch_var_oos'].abs().values * var_multiplier 
    
    # 遍历每一根 K 线，模拟真实的未来轨迹 (Path Dependency)
    for i in range(len(df) - horizon):
        p0 = close_prices[i]
        
        # 动态计算当前时刻的上下屏障价格
        upper_barrier = p0 * (1 + garch_var[i])
        lower_barrier = p0 * (1 - garch_var[i])
        
        # 提取未来 horizon 根 K 线的价格轨迹
        path = close_prices[i+1 : i+1+horizon]
        
        # 寻找轨迹中第一次触碰上/下屏障的索引
        hit_upper = np.where(path >= upper_barrier)[0]
        hit_lower = np.where(path <= lower_barrier)[0]
        
        # 获取触碰的时间点（如果没有触碰，设为无穷大）
        idx_upper = hit_upper[0] if len(hit_upper) > 0 else np.inf
        idx_lower = hit_lower[0] if len(hit_lower) > 0 else np.inf
        
        # 逻辑判断：谁先被触碰？
        if idx_upper < idx_lower:
            labels[i] = 2  # 先碰到上屏障 -> 真正有效的上涨 (做多)
        elif idx_lower < idx_upper:
            labels[i] = 0  # 先碰到下屏障 -> 真正有效的下跌 (做空)
        else:
            labels[i] = 1  # 超时(时间屏障)或同时触碰(极罕见) -> 震荡不接 (观望)
            
    df['label'] = labels
    
    # 砍掉最后 horizon 行，因为它们没有足够的未来数据来判断屏障
    return df.iloc[:-horizon].copy()

# 应用三重屏障 (由于 GARCH VaR 是99%极限值，可能偏大，可以乘以 0.8 作为合理的止盈止损线)
df_labeled = apply_triple_barrier(df_features, horizon=12, var_multiplier=0.8)

print("\n三重屏障标签分布情况 (0:跌穿底线, 1:震荡超时, 2:涨破天际):")
print(df_labeled['label'].value_counts())

# ==========================================
# 4. XGBoost 模型训练与样本权重平衡 (Class Imbalance)
# ==========================================
feature_cols = [
    'return_t-1', 'volume_change', 'rsi_14', 
    'garch_vol_zscore', 'delta_garch_vol', 'garch_var_oos', 
    'return_1H', 'rsi_1D'
]

X = df_labeled[feature_cols]
y = df_labeled['label']

# 严格按时间切分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 💡 核心优化：计算样本权重。
# 如果 '1'(震荡) 太多，'balanced' 模式会自动给 '0' 和 '2' 赋予极高的学习权重
# 强制 XGBoost 认真对待罕见但关键的突破信号！
train_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 初始化 XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    max_depth=4, 
    learning_rate=0.03, # 稍微调低学习率，让加权学习更稳定
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("\n开始训练 XGBoost 模型 (已启用样本权重平衡)...")
xgb_model.fit(
    X_train, y_train,
    sample_weight=train_weights,          # 💡 传入计算好的权重
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

# 预测测试集
y_pred = xgb_model.predict(X_test)

print("\n=== 进阶版 XGBoost 预测结果报告 ===")
print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Flat (1)', 'Up (2)']))

# ==========================================
# 5. 特征重要性可视化
# ==========================================
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=10, height=0.5, color='darkred')
plt.title('Triple-Barrier XGBoost Feature Importance (Gain)', fontsize=12)
plt.tight_layout()
plt.show()

# %%
# 提取测试集的特征
X_test_alpha = X_test.copy()

# ⚠️ 核心技巧：获取属于 0(跌), 1(平), 2(涨) 的概率矩阵
probs = xgb_model.predict_proba(X_test)

# 将概率拆解并存入 DataFrame 中
X_test_alpha['prob_down'] = probs[:, 0]  # 做空的概率
X_test_alpha['prob_flat'] = probs[:, 1]  # 震荡的概率
X_test_alpha['prob_up'] = probs[:, 2]    # 做多的概率

print("概率矩阵预览：")
print(X_test_alpha[['prob_down', 'prob_flat', 'prob_up']].head())

threshold_up = 0.40
threshold_down = 0.45 # 你模型的做空召回率高，做空阈值可以设高一点提高胜率

# 初始 Alpha 信号为 0 (空仓观望)
X_test_alpha['alpha_signal'] = 0

# 当做多概率大于阈值，发出 +1 信号
X_test_alpha.loc[X_test_alpha['prob_up'] > threshold_up, 'alpha_signal'] = 1

# 当做空概率大于阈值，发出 -1 信号
X_test_alpha.loc[X_test_alpha['prob_down'] > threshold_down, 'alpha_signal'] = -1

print("\n根据置信度过滤后的交易信号分布：")
print(X_test_alpha['alpha_signal'].value_counts())

# 假设你的初始资金为 10,000 U
# 我们希望每笔交易的潜在亏损(Risk)控制在总资金的 1% (即 100 U)
target_risk_per_trade = 0.01

# GARCH VaR 代表了极端的预期跌幅 (比如 0.02 就是预期可能跌 2%)
# 仓位大小 = 目标风险 / 预期风险
X_test_alpha['position_size'] = target_risk_per_trade / X_test_alpha['garch_var_oos'].abs()

# 设置最大杠杆限制（防止波动率极小时算出 100 倍杠杆）
max_leverage = 3
X_test_alpha['position_size'] = X_test_alpha['position_size'].clip(upper=max_leverage)

# 最终 Alpha 头寸 = 信号方向 * 仓位大小
X_test_alpha['target_position'] = X_test_alpha['alpha_signal'] * X_test_alpha['position_size']

print("\n最终的 Alpha 目标仓位 (部分展示)：")
print(X_test_alpha[['alpha_signal', 'garch_var_oos', 'position_size', 'target_position']].head(10))

xgb_model.save_model('xgb_alpha_model.json')
print("模型已保存为 xgb_alpha_model.json")
