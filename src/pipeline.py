"""
端到端流水线：从原始 OHLCV 数据，一路跑到"每根K线的突破概率"这份最终数据集。

跑完这个函数得到的 DataFrame，可以直接喂给 strategy.VolatilityBreakoutStrategy
做回测，或者喂给 stress_test.run_stress_test 做蒙特卡洛压力测试。

注意：完整跑一遍（尤其是 walk_forward_xgboost 那一步）在几年的 15 分钟线
数据上可能需要几十分钟到几小时，取决于机器性能，这也是为什么原 notebook
里每一步都把中间结果存成 csv——建议你在自己的机器上跑的时候也这样做。
"""
import pandas as pd

from . import config
from .features import FEATURE_COLUMNS, create_all_features
from .garch_vol import run_garch_pipeline
from .labeling import apply_breakout_labels
from .model import walk_forward_xgboost


def build_final_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df_raw: 原始 OHLCV DataFrame，index 为时间戳，至少包含
                 open/high/low/close/volume 列。

    Returns:
        附加了全部特征、is_breakout 标签、prob_breakout_v5 样本外概率的
        最终数据集，可直接用于回测。
    """
    df_raw = df_raw.sort_index().dropna(subset=["close"])

    print("第 1 步：滚动 GARCH 计算样本外波动率（较耗时）...")
    df_garch = run_garch_pipeline(
        df_raw, window_size=config.GARCH_WINDOW_SIZE, refit_every=config.GARCH_REFIT_EVERY
    )

    print("第 2 步：特征工程...")
    df_features = create_all_features(df_garch)

    print("第 3 步：三重屏障打标签...")
    df_labeled = apply_breakout_labels(
        df_features, horizon=config.BARRIER_HORIZON, var_multiplier=config.BARRIER_VAR_MULTIPLIER
    )

    df_labeled = df_labeled.replace([float("inf"), float("-inf")], pd.NA)
    df_clean = df_labeled.dropna(subset=FEATURE_COLUMNS + ["is_breakout"]).copy()
    print(f"清洗后可用于建模的数据量: {len(df_clean)} 行")

    print("第 4 步：Walk-forward 训练 XGBoost，输出样本外突破概率（最耗时的一步）...")
    breakout_probs = walk_forward_xgboost(
        df_clean,
        FEATURE_COLUMNS,
        train_window=config.WF_TRAIN_WINDOW,
        test_window=config.WF_TEST_WINDOW,
        step=config.WF_STEP,
    )

    df_final = df_clean.loc[breakout_probs.index].copy()
    df_final["prob_breakout_v5"] = breakout_probs["prob_breakout_v5"]

    return df_final
