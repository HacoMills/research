"""
滚动窗口 GARCH(1,1) 样本外波动率与 VaR 估计。

关键设计：每一个时刻的波动率预测，只用它之前的数据拟合出来的模型预测得到，
不存在用未来数据"偷看"的问题（no look-ahead）。这是整个项目里最容易埋雷、
也是最值得展示的一块——很多新手版本的 GARCH 特征其实是用全样本拟合后
再切训练/测试集，本质上已经泄露了未来信息。
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model
from tqdm import tqdm

from . import config


def calculate_rolling_garch(
    returns: pd.Series,
    window_size: int = config.GARCH_WINDOW_SIZE,
    refit_every: int = config.GARCH_REFIT_EVERY,
) -> tuple[pd.Series, pd.Series]:
    """
    对收益率序列做滚动窗口 GARCH(1,1) 拟合，每 refit_every 根K线重新拟合一次，
    每次拟合只使用截止当前时刻之前 window_size 根K线的历史数据。

    Args:
        returns: 对数收益率序列，建议已经放大 100 倍（百分比形式），
                 否则数值太小 GARCH 优化器容易不收敛。
        window_size: 每次拟合使用的历史样本数。
        refit_every: 每隔多少根K线重新拟合一次模型。

    Returns:
        (样本外条件波动率, 样本外条件均值) 两个 pandas Series，已 dropna。
    """
    out_of_sample_vol = pd.Series(index=returns.index, dtype=float)
    out_of_sample_mu = pd.Series(index=returns.index, dtype=float)

    n_samples = len(returns)

    for i in tqdm(range(window_size, n_samples, refit_every), desc="Rolling GARCH"):
        train_window = returns.iloc[i - window_size : i]
        am = arch_model(train_window, mean="Constant", vol="Garch", p=1, q=1, dist="Normal")

        try:
            res = am.fit(disp="off", show_warning=False)
        except Exception:  # noqa: BLE001 - 个别窗口拟合失败就跳过，不影响整体流程
            continue

        steps_to_predict = min(refit_every, n_samples - i)
        forecasts = res.forecast(horizon=steps_to_predict, align="origin")

        pred_vol = np.sqrt(forecasts.variance.iloc[-1].values)
        pred_mu = forecasts.mean.iloc[-1].values

        out_of_sample_vol.iloc[i : i + steps_to_predict] = pred_vol
        out_of_sample_mu.iloc[i : i + steps_to_predict] = pred_mu

    return out_of_sample_vol.dropna(), out_of_sample_mu.dropna()


def run_garch_pipeline(
    df: pd.DataFrame,
    window_size: int = config.GARCH_WINDOW_SIZE,
    refit_every: int = config.GARCH_REFIT_EVERY,
    var_confidence: float = 0.01,
) -> pd.DataFrame:
    """
    端到端流水线：给一个带 close 列的 OHLCV DataFrame，输出附加了
    cond_vol_oos / var_oos / garch_vol_oos / garch_var_oos 几列的 DataFrame。

    garch_vol_oos / garch_var_oos 是已经从"放大100倍的百分比"换算回小数
    的版本，可以直接乘价格用作屏障宽度。
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    returns = df["log_return"].dropna() * 100  # 放大100倍便于优化器收敛

    out_of_sample_vol, out_of_sample_mu = calculate_rolling_garch(returns, window_size, refit_every)

    z_score = stats.norm.ppf(var_confidence)
    var_oos = out_of_sample_mu + out_of_sample_vol * z_score

    df["cond_vol_oos"] = out_of_sample_vol
    df["var_oos"] = var_oos
    df["garch_vol_oos"] = df["cond_vol_oos"] / 100
    df["garch_var_oos"] = df["var_oos"] / 100

    return df.dropna(subset=["garch_var_oos"]).copy()
