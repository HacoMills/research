"""
Walk-forward 训练 XGBoost 分类器，输出样本外突破概率。

不用简单的一次性 train/test split，而是滚动切多个不重叠的窗口，
每个窗口只用它之前的数据训练、在它自己的窗口上预测，拼接起来的
就是一条完全样本外（out-of-sample）的概率序列——避免"未来偷看过去"
这种最常见、也最容易让回测好看但实盘失效的错误。
"""
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

from . import config
from .features import FEATURE_COLUMNS


def walk_forward_xgboost(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    label_col: str = "is_breakout",
    train_window: int = config.WF_TRAIN_WINDOW,
    test_window: int = config.WF_TEST_WINDOW,
    step: int = config.WF_STEP,
) -> pd.DataFrame:
    """
    Returns:
        一个 DataFrame，index 是原始时间戳，列 prob_breakout_v5 是该时刻的
        样本外突破概率预测值。只覆盖被至少一个测试窗口覆盖到的时间段。
    """
    n = len(df)
    all_probs = []

    for start in tqdm(range(0, n - train_window - test_window, step), desc="Walk-forward XGBoost"):
        train_end = start + train_window
        test_end = train_end + test_window

        train_slice = df.iloc[start:train_end]
        test_slice = df.iloc[train_end:test_end]

        X_tr, y_tr = train_slice[feature_cols], train_slice[label_col]
        X_te = test_slice[feature_cols]

        weights = compute_sample_weight(class_weight="balanced", y=y_tr)

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=4,
            learning_rate=0.03,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, sample_weight=weights, verbose=False)

        prob_breakout = model.predict_proba(X_te)[:, 1]
        all_probs.append(pd.DataFrame({"prob_breakout_v5": prob_breakout}, index=test_slice.index))

    return pd.concat(all_probs)
