"""
特征工程：技术指标 + GARCH 衍生特征 + 趋势/环境（regime）特征。

所有特征都只使用截止当前K线为止的信息（pct_change / rolling / ewm 都是
向后看的滑动窗口），不存在未来函数。
"""
import numpy as np
import pandas as pd
import pandas_ta as ta

FEATURE_COLUMNS = [
    "return_t-1",
    "volume_change",
    "rsi_14",
    "garch_vol_zscore",
    "delta_garch_vol",
    "garch_var_oos",
    "return_1H",
    "rsi_1D",
    "trend_alignment",
    "ema_60_slope",
    "adx_14",
    "vol_percentile_30d",
    "price_position_7d",
    "return_3d",
    "return_7d",
]


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：已经跑过 garch_vol.run_garch_pipeline 的 DataFrame（含 garch_vol_oos / garch_var_oos）。
    输出：附加了 FEATURE_COLUMNS 里全部特征列的 DataFrame。
    """
    df = df.copy()

    # --- 基础价格/成交量特征 ---
    df["return_t-1"] = df["close"].pct_change(1)
    df["volume_change"] = df["volume"].pct_change(1)
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["return_1H"] = df["close"].pct_change(4)
    df["rsi_1D"] = ta.rsi(df["close"], length=96)

    # --- GARCH 衍生特征 ---
    df["delta_garch_vol"] = df["garch_vol_oos"].diff(1)
    window_24h = 96
    rolling_mean = df["garch_vol_oos"].rolling(window=window_24h).mean()
    rolling_std = df["garch_vol_oos"].rolling(window=window_24h).std()
    df["garch_vol_zscore"] = (df["garch_vol_oos"] - rolling_mean) / rolling_std
    df["vol_percentile_30d"] = df["garch_vol_oos"].rolling(96 * 30).rank(pct=True)

    # --- 趋势 / 市场环境（regime）特征 ---
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_60"] = df["close"].ewm(span=60).mean()
    df["ema_120"] = df["close"].ewm(span=120).mean()
    df["trend_alignment"] = (df["ema_20"] - df["ema_120"]) / df["ema_120"]
    df["ema_60_slope"] = df["ema_60"].pct_change(20)

    adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx_14"] = adx_result["ADX_14"] if adx_result is not None else np.nan

    rolling_high = df["close"].rolling(96 * 7).max()
    rolling_low = df["close"].rolling(96 * 7).min()
    df["price_position_7d"] = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-9)
    df["return_3d"] = df["close"].pct_change(96 * 3)
    df["return_7d"] = df["close"].pct_change(96 * 7)

    return df
