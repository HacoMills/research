"""
三重屏障标签法（Triple-Barrier Labeling），改自 Marcos López de Prado 的
《Advances in Financial Machine Learning》。

屏障宽度不是固定百分比，而是用当前时刻的 GARCH VaR 动态计算——波动大的
时候屏障更宽，波动小的时候屏障更窄，这样标签的"真涨/真跌"定义会随行情
的波动水平自适应，而不是在剧烈波动期把噪音当成信号。
"""
import numpy as np
import pandas as pd


def apply_triple_barrier(
    df: pd.DataFrame,
    horizon: int = 12,
    var_multiplier: float = 0.8,
) -> pd.DataFrame:
    """
    三分类版本：0=先触碰下屏障(跌), 1=超时未触碰(震荡), 2=先触碰上屏障(涨)。

    Args:
        df: 需包含 close 列和 garch_var_oos 列。
        horizon: 时间屏障，未来多少根K线内必须分出胜负。
        var_multiplier: 屏障宽度 = |garch_var_oos| * var_multiplier。

    Returns:
        附加了 label_tri 列的 DataFrame，末尾 horizon 行因缺乏未来数据被截掉。
    """
    df = df.copy()
    labels = np.ones(len(df))
    close_prices = df["close"].values
    garch_var = df["garch_var_oos"].abs().values * var_multiplier

    for i in range(len(df) - horizon):
        p0 = close_prices[i]
        upper_barrier = p0 * (1 + garch_var[i])
        lower_barrier = p0 * (1 - garch_var[i])
        path = close_prices[i + 1 : i + 1 + horizon]

        hit_upper = np.where(path >= upper_barrier)[0]
        hit_lower = np.where(path <= lower_barrier)[0]
        idx_upper = hit_upper[0] if len(hit_upper) > 0 else np.inf
        idx_lower = hit_lower[0] if len(hit_lower) > 0 else np.inf

        if idx_upper < idx_lower:
            labels[i] = 2
        elif idx_lower < idx_upper:
            labels[i] = 0
        else:
            labels[i] = 1

    df["label_tri"] = labels
    return df.iloc[:-horizon].copy()


def apply_breakout_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    var_multiplier: float = 0.8,
) -> pd.DataFrame:
    """
    在三分类基础上再做一次二分类：is_breakout = 1 表示先触碰了任一屏障（不管方向），
    is_breakout = 0 表示在时间屏障内一直没有突破（震荡）。

    这是最终"波动率突破策略"用的标签——不预测方向，只预测"会不会有大波动"，
    方向交给回测阶段"先碰哪边就跟哪边"的规则来决定。
    """
    df = apply_triple_barrier(df, horizon=horizon, var_multiplier=var_multiplier)
    df["is_breakout"] = (df["label_tri"] != 1).astype(int)
    return df
