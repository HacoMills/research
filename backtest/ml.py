"""
机器学习信号: 用模型自动生成信号 / alpha
=========================================
两种用法, 共用同一个滚动训练 (walk-forward) 引擎:

  时序   xgb:5 / mlp:5 / ridge:5 / lgbm:5     接到信号层, 和 donchian / tsmom 一样用字符串切换
         每个品种用自己的历史训练, 预测未来 h 天 (波动率标准化后的) 收益;
         预测 > 阈值开多, < -阈值开空, 预测回到 0 出场.
         写法 "xgb:预测天数,阈值,训练天数,重训间隔天数", 例如 xgb:5,0.1,756,63
  截面   ml_alpha(P, model)                    接到 cross_section.py
         把所有股票的特征叠在一起训练, 预测 "下一期收益的截面排名", 预测值就是 alpha.

防前视 (这是机器学习回测最容易出错的地方):
  1. 特征只用 t 日收盘及以前的数据
  2. 滚动训练: 在 T 日重新训练时, 只用 T 日以前的样本; 预测 [T, T + 重训间隔) 这一段
  3. 隔离期 (embargo): 标签是未来 h 天收益, t 日样本的标签要到 t+h 日才知道,
     所以训练集只取 t < T - h 的样本, 否则训练集会 "看到" 测试期的价格
  4. 超参数是事先固定的, 没有在测试期上调参 (调参也要放进滚动训练里, 否则又是过拟合)
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .signals import SIGNALS, Signal, SignalArrays, _bars

FEATURE_WARMUP_DAYS = 250        # 最长的特征窗口 (天)


# ──────────────────────────────────────────────────
# 模型 (超参数事先固定, 刻意保守: 金融数据信噪比很低, 复杂模型容易拟合噪音)
# ──────────────────────────────────────────────────

MODELS = ("ridge", "xgb", "lgbm", "mlp")


def make_model(name: str, seed: int = 0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "ridge":
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=50, n_jobs=4, random_state=seed,
                            verbosity=0)
    if name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, min_child_samples=100,
                             random_state=seed, verbose=-1)
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                                            early_stopping=True, max_iter=60,
                                                            random_state=seed))
    raise ValueError(f"未知模型 {name!r}, 可选: {', '.join(MODELS)}")


def feature_importance(model, names: Sequence[str]) -> pd.Series:
    """树模型: 分裂增益占比; 线性模型: 标准化后的系数. 神经网络没有直接的重要性."""
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=names).sort_values(ascending=False)
    if hasattr(est, "coef_"):
        return pd.Series(est.coef_, index=names).sort_values(key=abs, ascending=False)
    return pd.Series(dtype=float)


# ──────────────────────────────────────────────────
# 滚动训练
# ──────────────────────────────────────────────────

def walk_forward(X: np.ndarray, y: np.ndarray, t: np.ndarray, model: str, start: int, train: int,
                 step: int, embargo: int, max_rows: Optional[int] = None,
                 seed: int = 0) -> Tuple[np.ndarray, object]:
    """
    X, y, t: 每行一个样本, t 是整数时间 (第几天/第几根K线). 截面数据同一个 t 有多行.
    从 T = start 开始, 每 step 个时间单位重新训练一次:
        训练集 = T - train <= t < T - embargo 且标签已知
        测试集 = T <= t < T + step
    返回 (样本外预测, 最后一次训练的模型). start 之前的预测为 NaN.
    """
    pred = np.full(len(y), np.nan)
    known = ~np.isnan(y)
    rng = np.random.default_rng(seed)
    last = None
    for T in range(int(start), int(t.max()) + 1, int(step)):
        tr = np.flatnonzero((t >= T - train) & (t < T - embargo) & known)
        te = (t >= T) & (t < T + step)
        if len(tr) < 100 or not te.any():
            continue
        if max_rows and len(tr) > max_rows:          # 截面样本太多时随机抽样, 控制训练时间
            tr = rng.choice(tr, max_rows, replace=False)
        m = make_model(model, seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        last = m
    return pred, last


# ──────────────────────────────────────────────────
# 时序: 单品种特征 + 信号
# ──────────────────────────────────────────────────

def _has_raw(df: pd.DataFrame) -> bool:
    return "scale" in df.columns and "raw_close" in df.columns


def _ret(df: pd.DataFrame, n: int) -> pd.Series:
    """过去 n 根K线的真实涨跌幅. 与 TSMOM 信号同样的换算: 差值复权的期货不能直接用 c / c.shift(n)."""
    c = df["close"]
    if _has_raw(df):
        return (c - c.shift(n)) * df["scale"] / df["raw_close"].shift(n)
    return c / c.shift(n) - 1


def _true_change(df: pd.DataFrame, diff: pd.Series) -> pd.Series:
    """复权价的点数差 → 相对当前真实价格的比例."""
    return diff * df["scale"] / df["raw_close"] if _has_raw(df) else diff / df["close"]


def ts_features(df: pd.DataFrame, bars_per_day: float) -> pd.DataFrame:
    """单品种特征, 全部做了波动率标准化, 不同品种、不同时期可比."""
    c, h, l = df["close"], df["high"], df["low"]
    r1 = _ret(df, 1)
    vol = r1.rolling(_bars(20, bars_per_day)).std()
    f = {}
    for d in (5, 20, 60, 120, 250):
        n = _bars(d, bars_per_day)
        f[f"mom{d}"] = _ret(df, n) / (vol * math.sqrt(n))             # 动量的 t 值
    f["vol_ratio"] = vol / r1.rolling(_bars(250, bars_per_day)).std()  # 现在的波动 / 过去一年的波动
    for d in (20, 60, 250):
        n = _bars(d, bars_per_day)
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        f[f"pos{d}"] = (c - lo) / (hi - lo) - 0.5                       # 在 N 天通道里的位置
    for d in (20, 60, 200):
        n = _bars(d, bars_per_day)
        f[f"ma{d}"] = _true_change(df, c - c.rolling(n).mean()) / vol    # 离均线几个波动率
    f["ret1"] = r1 / vol
    out = pd.DataFrame(f, index=df.index).replace([np.inf, -np.inf], np.nan)
    return out


def ts_label(df: pd.DataFrame, bars_per_day: float, h: int) -> pd.Series:
    """未来 h 根K线的收益 / 波动率, 截断在 ±5 (只用于训练, 不进入交易决策)."""
    vol = _ret(df, 1).rolling(_bars(20, bars_per_day)).std()
    fwd = _ret(df, h).shift(-h)                                        # t → t+h 的收益
    return (fwd / (vol * math.sqrt(h))).clip(-5, 5)


class MLSignal(Signal):
    """
    机器学习时序信号. 参数单位是天:
      horizon_days  预测未来几天的收益
      threshold     预测值 (单位: 波动率) 超过多少才入场, 0 = 只看方向
      train_days    每次训练用过去多少天
      retrain_days  每隔多少天重新训练
    """
    name = "ml_base"
    model = "ridge"
    default_stop = None

    def __init__(self, horizon_days: float = 5, threshold: float = 0.0, train_days: float = 756,
                 retrain_days: float = 63, stop: Optional[float] = None):
        super().__init__(stop)
        self.horizon_days, self.threshold = horizon_days, threshold
        self.train_days, self.retrain_days = train_days, retrain_days

    def params(self):
        return (self.horizon_days, self.threshold)

    def label(self):
        s = f"{self.name}:{self.horizon_days:g},{self.threshold:g}"
        if (self.train_days, self.retrain_days) != (756, 63):
            s += f",{self.train_days:g},{self.retrain_days:g}"
        return s + (f",stop={self.stop_mult:g}" if self.stop_mult else "")

    def predict(self, df: pd.DataFrame, bars_per_day: float) -> pd.Series:
        h = _bars(self.horizon_days, bars_per_day)
        X = ts_features(df, bars_per_day)
        y = ts_label(df, bars_per_day, h)
        warm = _bars(FEATURE_WARMUP_DAYS, bars_per_day)
        train = _bars(self.train_days, bars_per_day)
        pred, _ = walk_forward(X.fillna(0).values, y.values, np.arange(len(df)), self.model,
                               start=warm + train, train=train,
                               step=_bars(self.retrain_days, bars_per_day), embargo=h)
        return pd.Series(pred, index=df.index)

    def compute(self, df, bars_per_day):
        p = self.predict(df, bars_per_day).values
        valid = np.flatnonzero(~np.isnan(p))
        warmup = int(valid[0]) if len(valid) else len(p)
        th = self.threshold
        with np.errstate(invalid="ignore"):
            return SignalArrays(p > th, p < -th, p <= 0, p >= 0, warmup)


class RidgeSignal(MLSignal):
    name, model = "ridge", "ridge"


class XGBSignal(MLSignal):
    name, model = "xgb", "xgb"


class LGBMSignal(MLSignal):
    name, model = "lgbm", "lgbm"


class MLPSignal(MLSignal):
    name, model = "mlp", "mlp"


for _cls in (RidgeSignal, XGBSignal, LGBMSignal, MLPSignal):
    SIGNALS[_cls.name] = _cls


# ──────────────────────────────────────────────────
# 截面: 面板特征 + alpha
# ──────────────────────────────────────────────────

def cs_features(P: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """截面特征, 每天在所有股票之间排名后减 0.5 (−0.5 ~ 0.5), 让不同时期可比."""
    o, h, l, c, v = P["open"], P["high"], P["low"], P["close"], P["volume"]
    ret = c.pct_change(fill_method=None)
    f = {}
    for n in (1, 5, 20, 60, 120):
        f[f"r{n}"] = c / c.shift(n) - 1                       # 短期反转 / 中期动量
    f["r250_20"] = c.shift(20) / c.shift(250) - 1             # 经典动量: 过去一年, 跳过最近一个月
    f["vol20"] = ret.rolling(20).std()
    f["vol60"] = ret.rolling(60).std()
    f["intraday"] = -(c - o) / (h - l).where(h > l)            # 日内反转 (前面那个 alpha 的核心)
    f["overnight"] = o / c.shift(1) - 1                        # 隔夜跳空
    f["rel_vol"] = v / v.rolling(20).mean()                    # 相对放量
    f["range"] = (h - l) / c                                   # 日内振幅
    hi, lo = h.rolling(20).max(), l.rolling(20).min()
    f["pos20"] = (c - lo) / (hi - lo)
    f["ma20"] = c / c.rolling(20).mean() - 1
    f["ma60"] = c / c.rolling(60).mean() - 1
    return {k: x.replace([np.inf, -np.inf], np.nan).rank(axis=1, pct=True) - 0.5 for k, x in f.items()}


def ml_alpha(P: Dict[str, pd.DataFrame], model: str = "xgb", horizon: int = 1, train_days: int = 756,
             retrain_days: int = 63, max_rows: Optional[int] = 200_000,
             features: Optional[Dict[str, pd.DataFrame]] = None, seed: int = 0):
    """
    标签 = 未来 horizon 天收益的截面排名 (−0.5 ~ 0.5): 只学 "谁比谁好", 不学大盘涨跌.
    返回 (alpha 宽表, 最后一次训练的特征重要性). 只有样本外预测, 训练期为 NaN.
    """
    F = features or cs_features(P)
    names = list(F)
    close = P["close"]
    fwd = close.shift(-horizon) / close - 1
    y = fwd.rank(axis=1, pct=True) - 0.5

    n_days, n_stocks = close.shape
    X = np.column_stack([F[k].reindex_like(close).values.ravel() for k in names])
    X = np.nan_to_num(X, nan=0.0)                              # 缺失 = 截面中位数
    yv = y.values.ravel()
    t = np.repeat(np.arange(n_days), n_stocks)
    live = close.notna().values.ravel()

    pred = np.full(len(yv), np.nan)
    p, last = walk_forward(X[live], yv[live], t[live], model, start=FEATURE_WARMUP_DAYS + train_days,
                           train=train_days, step=retrain_days, embargo=horizon, max_rows=max_rows,
                           seed=seed)
    pred[live] = p
    alpha = pd.DataFrame(pred.reshape(n_days, n_stocks), index=close.index, columns=close.columns)
    imp = feature_importance(last, names) if last is not None else pd.Series(dtype=float)
    return alpha, imp
