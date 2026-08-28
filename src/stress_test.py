"""
蒙特卡洛压力测试：在长周期数据里随机抽取大量固定长度的片段独立回测，
用分布（均值/中位数/95% VaR/极端回撤）而不是单一一条历史回测曲线来
评估策略的稳健性。单条回测曲线很容易只是"运气好走了一段行情"，
分布视角能看出策略在不同市场环境下的表现有多离散。
"""
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from .strategy import VolatilityBreakoutStrategy


def run_stress_test(
    df_prepared: pd.DataFrame,
    strategy: VolatilityBreakoutStrategy,
    n_samples: int = 20,
    segment_days: int = 90,
    bars_per_day: int = 96,
) -> pd.DataFrame:
    """
    对长周期数据进行随机采样压力测试。

    Args:
        df_prepared: 包含全部特征、标签、模型概率的最终数据集。
        strategy: 已实例化的 VolatilityBreakoutStrategy。
        n_samples: 随机抽取的片段数量。
        segment_days: 每个片段的时间跨度（天数）。

    Returns:
        每个采样片段一行的结果表：Sample, Start, End, Trades, Win Rate,
        Net Profit, Max Drawdown。
    """
    segment_bars = segment_days * bars_per_day
    max_start_idx = len(df_prepared) - segment_bars

    if max_start_idx <= 0:
        raise ValueError(f"数据总长度({len(df_prepared)})不足以切分 {segment_days} 天的片段！")

    test_results = []

    for i in tqdm(range(n_samples), desc="蒙特卡洛压力测试"):
        start_idx = random.randint(0, max_start_idx)
        end_idx = start_idx + segment_bars
        segment_df = df_prepared.iloc[start_idx:end_idx].copy()

        start_date = segment_df.index[0].strftime("%Y-%m-%d")
        end_date = segment_df.index[-1].strftime("%Y-%m-%d")

        trades = strategy.run_backtest(segment_df)

        if len(trades) > 0:
            trades["cum_net_value"] = (1 + trades["net_pnl"]).cumprod()
            running_max = trades["cum_net_value"].cummax()
            mdd = ((trades["cum_net_value"] - running_max) / running_max).min()
            win_rate = (trades["net_pnl"] > 0).mean()
            net_profit = trades["cum_net_value"].iloc[-1] - 1
            n_trades = len(trades)
        else:
            mdd = 0
            win_rate = 0
            net_profit = 0
            n_trades = 0

        test_results.append(
            {
                "Sample": f"Sample_{i + 1}",
                "Start": start_date,
                "End": end_date,
                "Trades": n_trades,
                "Win Rate": win_rate,
                "Net Profit": net_profit,
                "Max Drawdown": mdd,
            }
        )

    return pd.DataFrame(test_results)


def summarize_stress_test(results_df: pd.DataFrame) -> dict:
    """把 run_stress_test 的输出汇总成关键风险/收益指标。"""
    valid = results_df[results_df["Trades"] > 0]
    if len(valid) == 0:
        return {"valid_samples": 0}

    return {
        "valid_samples": len(valid),
        "total_samples": len(results_df),
        "avg_trades_per_period": valid["Trades"].mean(),
        "avg_win_rate": valid["Win Rate"].mean(),
        "mean_net_profit": valid["Net Profit"].mean(),
        "median_net_profit": valid["Net Profit"].median(),
        "positive_period_ratio": (valid["Net Profit"] > 0).mean(),
        "var_95": np.percentile(valid["Net Profit"], 5),
        "avg_max_drawdown": valid["Max Drawdown"].mean(),
        "worst_1pct_drawdown": np.percentile(valid["Max Drawdown"], 1),
    }
