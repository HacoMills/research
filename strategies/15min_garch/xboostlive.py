import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import xgboost as xgb
from arch import arch_model
import time
import datetime
import logging
import scipy.stats as stats
import csv
import os

# ==========================================
# 0. 配置与初始化 (加入了日志保存功能)
# ==========================================
# 让 logging 同时输出到屏幕和本地的 bot_run.log 文件中
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_run.log", encoding='utf-8'),  # 记录到本地文件
        logging.StreamHandler()                                # 同时打印到屏幕
    ]
)

SYMBOL = 'BTC/USDT:USDT'  # OKX Swap 标准写法
TIMEFRAME = '15m'
LEVERAGE_MAX = 3
TARGET_RISK_PER_TRADE = 0.01  # 每笔交易最大亏损总资金的 1%
THRESHOLD_UP = 0.40
THRESHOLD_DOWN = 0.45
TRADE_RECORD_FILE = 'trade_records.csv' # 交易记录保存的表格名字

# 初始化交易所 (填入你的实盘API)
exchange = ccxt.okx({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET',
    'password': 'YOUR_PASSWORD',
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
    'proxies': {'http': 'http://127.0.0.1:29290', 'https': 'http://127.0.0.1:29290'}
})

# ==========================================
# 1. 模型热更新加载模块
# ==========================================
def load_my_model():
    """读取本地 JSON 模型的函数"""
    try:
        model = xgb.XGBClassifier()
        model.load_model('xgb_alpha_model.json')
        logging.info("成功加载最新的 XGBoost 策略模型。")
        return model
    except Exception as e:
        logging.error(f"加载模型失败，请检查 json 文件是否存在: {e}")
        return None

# 初始化启动时加载模型
model = load_my_model()

# ==========================================
# 2. 本地记录模块 (新增)
# ==========================================
def record_trade_to_csv(action, delta_contracts, current_contracts, price, total_equity, alpha_signal, garch_var):
    """将每次的交易行为写入 CSV 表格，方便 Excel 打开复盘"""
    try:
        # 检查文件是否存在，如果不存在则需要先写入表头
        file_exists = os.path.isfile(TRADE_RECORD_FILE)
        
        with open(TRADE_RECORD_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                # 写入表头
                writer.writerow(['时间', '交易方向', '交易张数', '交易后总持仓', '成交参考价', '策略信号', '预期风险(VaR)', '账户总权益(U)'])
            
            # 写入本次交易数据
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                action,
                abs(delta_contracts),
                current_contracts + delta_contracts, # 交易后的理论持仓
                price,
                alpha_signal,
                round(garch_var, 4),
                round(total_equity, 2)
            ])
        logging.info(f"💾 交易已成功记录到本地文件: {TRADE_RECORD_FILE}")
    except Exception as e:
        logging.error(f"⚠️ 记录交易到 CSV 时发生错误: {e}")

# ==========================================
# 3. 核心数据与特征计算函数
# ==========================================
def fetch_recent_data(limit=250):
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def get_latest_garch(returns, window_size=96):
    train_window = returns.tail(window_size) * 100 
    am = arch_model(train_window, mean='Constant', vol='Garch', p=1, q=1, dist='Normal')
    res = am.fit(disp='off', show_warning=False)
    
    forecasts = res.forecast(horizon=1)
    pred_vol = np.sqrt(forecasts.variance.iloc[-1].values[0]) / 100  
    pred_mu = forecasts.mean.iloc[-1].values[0] / 100
    
    z_score = stats.norm.ppf(0.01)
    var_99 = pred_mu + pred_vol * z_score
    return pred_vol, var_99

def prepare_latest_features(df):
    df['return_t-1'] = df['close'].pct_change(1)
    df['volume_change'] = df['volume'].pct_change(1)
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['return_1H'] = df['close'].pct_change(4)
    df['rsi_1D'] = ta.rsi(df['close'], length=96)
    
    returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    latest_vol, latest_var = get_latest_garch(returns)
    
    vol_history = returns.tail(96).rolling(10).std()
    current_zscore = (latest_vol - vol_history.mean()) / vol_history.std()
    
    latest_features = {
        'return_t-1': df['return_t-1'].iloc[-1],
        'volume_change': df['volume_change'].iloc[-1],
        'rsi_14': df['rsi_14'].iloc[-1],
        'garch_vol_zscore': current_zscore,
        'delta_garch_vol': latest_vol - vol_history.iloc[-2], 
        'garch_var_oos': latest_var,
        'return_1H': df['return_1H'].iloc[-1],
        'rsi_1D': df['rsi_1D'].iloc[-1]
    }
    return pd.DataFrame([latest_features]), latest_var

# ==========================================
# 4. 交易执行模块
# ==========================================
def execute_trade(alpha_signal, garch_var):
    try:
        balance = exchange.fetch_balance()
        total_equity = balance['USDT']['total']
        
        position_size_ratio = TARGET_RISK_PER_TRADE / abs(garch_var)
        position_size_ratio = min(position_size_ratio, LEVERAGE_MAX) 
        
        target_notional = total_equity * position_size_ratio * alpha_signal
        
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        
        market_info = exchange.market(SYMBOL)
        contract_size = market_info['contractSize'] 
        
        target_contracts = round((target_notional / current_price) / contract_size)
        
        positions = exchange.fetch_positions([SYMBOL])
        current_contracts = 0
        if positions:
            for p in positions:
                if p['symbol'] == SYMBOL:
                    current_contracts = p['contracts'] if p['side'] == 'long' else -p['contracts']
                    
        delta_contracts = target_contracts - current_contracts
        
        logging.info(f"总权益: {total_equity:.2f} U | 目标张数: {target_contracts} | 当前张数: {current_contracts}")
        
        # 3. 发送订单并记录到 CSV
        if delta_contracts > 0:
            logging.info(f"==> 执行买入(开多/平空): {delta_contracts} 张")
            # 真实下单代码 (请在确认无误后解开注释)
            # exchange.create_market_order(SYMBOL, 'buy', abs(delta_contracts))
            
            # 调用记录函数
            record_trade_to_csv('BUY', delta_contracts, current_contracts, current_price, total_equity, alpha_signal, garch_var)
            
        elif delta_contracts < 0:
            logging.info(f"==> 执行卖出(开空/平多): {abs(delta_contracts)} 张")
            # 真实下单代码 (请在确认无误后解开注释)
            # exchange.create_market_order(SYMBOL, 'sell', abs(delta_contracts))
            
            # 调用记录函数
            record_trade_to_csv('SELL', delta_contracts, current_contracts, current_price, total_equity, alpha_signal, garch_var)
            
        else:
            logging.info("无需调仓。")
            
    except Exception as e:
        logging.error(f"执行交易时出错: {e}")

# ==========================================
# 5. 实盘主循环
# ==========================================
def run_bot():
    global model
    logging.info("🚀 实盘机器人已启动，等待 15 分钟 K 线收盘...")
    
    while True:
        try:
            now = datetime.datetime.now()
            
            # 热更新：每天凌晨 00:00:00 自动重新加载模型 (搭配你偶尔跑一次 xboost.py)
            if now.hour == 0 and now.minute == 0 and now.second == 0:
                logging.info("🔄 触发每日模型热更新...")
                new_model = load_my_model()
                if new_model is not None:
                    model = new_model
                time.sleep(1)
                
            # 每 15 分钟的第 2 秒触发 (如 10:00:02, 10:15:02)
            if now.minute % 15 == 0 and now.second == 2:
                logging.info("=== ⏳ 开始新的 15 分钟决策周期 ===")
                
                df = fetch_recent_data()
                features_df, garch_var = prepare_latest_features(df)
                
                probs = model.predict_proba(features_df)
                prob_down, prob_flat, prob_up = probs[0][0], probs[0][1], probs[0][2]
                logging.info(f"📊 预测概率 -> 跌: {prob_down:.2%}, 震荡: {prob_flat:.2%}, 涨: {prob_up:.2%}")
                
                alpha_signal = 0
                if prob_up > THRESHOLD_UP:
                    alpha_signal = 1
                elif prob_down > THRESHOLD_DOWN:
                    alpha_signal = -1
                    
                execute_trade(alpha_signal, garch_var)
                
                # 休息 60 秒，防止在这一分钟内重复触发
                time.sleep(60)
            else:
                time.sleep(1)
                
        except Exception as e:
            logging.error(f"🚨 主循环发生严重错误: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_bot()