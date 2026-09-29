"""
第 4 层: 跨资产 / 跨周期组合
============================
每个 "品种 × 周期" 单独回测 (执行层已按 1% 风险/ATR 做了风险均衡), 再把日收益组合起来.

组合方式 (weighting):
  class       先在每个资产类别内等权, 再把各类别等权 (默认). 否则商品十几个品种会压倒只有几个品种的类别
  instrument  所有 "品种 × 周期" 直接等权
多个周期时, 同一品种的不同周期视为同一类别里的不同成员 (跨周期分散).
每天只用当时已经开始交易的品种; 某品种开始交易后遇到本市场休市, 当天收益记 0.

波动率缩放 (参考 TSMOM 论文, 只用过去数据估计, 无前视):
  instrument_vol  每个品种先缩放到相同的年化波动 (论文 40%), 0 = 不缩放
  target_vol      组合整体缩放到目标年化波动 (如 10%), 0 = 不缩放
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import markets
from .engine import MarketSpec, run_backtest
from .engine import performance
from .filters import Filter, FilterContext, NoFilter
from .signals import Blend


@dataclass
class Instrument:
    cls: str                  # 资产类别
    market: str
    symbol: str
    tf: str
    bars_per_day: float
    df: pd.DataFrame
    spec: MarketSpec

    @property
    def key(self):
        return f"{self.symbol}@{self.tf}"


@dataclass
class PortfolioResult:
    signal: str
    filter: str
    results: Dict[str, Dict]                  # key → 单品种回测结果
    returns: pd.Series                        # 组合日收益 (未缩放)
    by_class: pd.DataFrame                    # 各类别日收益
    by_inst: pd.DataFrame                     # 各品种日收益 (列名 "类别:品种@周期")
    by_tf: Dict[str, pd.Series]               # 各周期单独组合
    dpy: float                                # 每年收益观测数
    scaled: Optional[pd.Series] = None        # 缩放到目标波动后的组合
    leverage: Optional[pd.Series] = None

    @property
    def main(self) -> pd.Series:
        """报告和稳定性检验使用的组合收益: 有目标波动时用缩放后的."""
        return self.scaled if self.scaled is not None else self.returns


# ──────────────────────────────────────────────────
# 组合工具
# ──────────────────────────────────────────────────

def span_fill(R: pd.DataFrame) -> pd.DataFrame:
    """每个品种在"开始交易 ~ 最后一天"之间没有数据的日子 (对方市场休市) 记为 0 收益."""
    R = R.copy()
    for c in R.columns:
        s = R[c]
        a, b = s.first_valid_index(), s.last_valid_index()
        if a is not None:
            R.loc[a:b, c] = s.loc[a:b].fillna(0.0)
    return R


def combine(returns: Dict[str, Dict[str, pd.Series]], weighting: str = "class"):
    """{类别: {成员: 日收益}} → (组合日收益, 类别 DataFrame, 成员 DataFrame)."""
    series = [r for d in returns.values() for r in d.values() if r is not None and len(r)]
    if not series:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()
    idx = pd.DatetimeIndex(sorted(set().union(*[set(r.index) for r in series])))
    by_class, all_inst = {}, {}
    for cls, d in returns.items():
        d = {k: v for k, v in d.items() if v is not None and len(v)}
        if not d:
            continue
        R = span_fill(pd.DataFrame(d).reindex(idx))
        by_class[cls] = R.mean(axis=1, skipna=True)
        all_inst.update({f"{cls}:{k}": R[k] for k in R.columns})
    C, I = pd.DataFrame(by_class), pd.DataFrame(all_inst)
    port = (C if weighting == "class" else I).mean(axis=1, skipna=True).dropna()
    return port, C.loc[port.index], I.loc[port.index]


def vol_scale(r: pd.Series, target: float, dpy: float, com: int = 60, max_lev: float = 10.0):
    """
    事前波动率缩放: 杠杆_t = 目标波动 / σ_{t-1}, σ 用 EWMA (质心 60 天) 估计, shift(1) 无前视.
    只用收益不为 0 的日子估计 σ (空仓日收益为 0, 计入会低估 σ, 再入场时杠杆暴涨).
    """
    active = r.where(r != 0)
    sigma = active.ewm(com=com, min_periods=com, ignore_na=True).std().ffill() * math.sqrt(dpy)
    lev = (target / sigma).clip(upper=max_lev).shift(1)
    out = (r * lev).dropna()
    return out, lev.reindex(out.index)


def obs_per_year(r: pd.Series) -> float:
    if len(r) < 2:
        return 252.0
    years = (r.index[-1] - r.index[0]).days / 365.25
    return len(r) / years if years > 0 else 252.0


def within_cross_corr(I: pd.DataFrame):
    """类内 / 跨类 品种两两相关性平均 (TSMOM 论文 Table 4)."""
    corr = I.corr()
    cls = [c.split(":")[0] for c in corr.columns]
    within, cross = [], []
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            v = corr.iat[i, j]
            if not np.isnan(v):
                (within if cls[i] == cls[j] else cross).append(v)
    return (np.mean(within) if within else np.nan), (np.mean(cross) if cross else np.nan)


def price_returns(px) -> pd.Series:
    """
    买入持有的日收益. 带 raw_close/scale 的复权数据: 复权价变动点数 × scale / 前一天真实价格
    (差值复权的期货价格可能接近 0 甚至为负, 不能直接 pct_change).
    """
    if isinstance(px, pd.DataFrame):
        d = px.dropna(subset=["close"]).resample("1D").last().dropna(subset=["close"])
        if "raw_close" in d.columns and "scale" in d.columns:
            r = d["close"].diff() * d["scale"] / d["raw_close"].shift(1)
            return r.replace([np.inf, -np.inf], np.nan).dropna()
        px = d["close"]
    daily = px[px > 0].dropna().resample("1D").last().dropna()
    return daily.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


# ──────────────────────────────────────────────────
# Runner: 数据只加载一次, 可以反复换信号 / 过滤器 / 成本重跑
# ──────────────────────────────────────────────────

class Runner:
    def __init__(self, cfg: dict, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.pcfg = cfg.get("portfolio", {})
        self.specs = markets.base_specs(cfg.get("costs"))
        self.multipliers = {k.upper(): v for k, v in cfg.get("multipliers", {}).items()}
        self.instruments: List[Instrument] = []
        self.prices: Dict[str, Dict[str, pd.DataFrame]] = {}     # 类别 → 品种 → 基础 K线 (基准用)
        self.failed: List[str] = []
        self._filter_cache: Dict[tuple, Optional[pd.Series]] = {}

    # 数据层
    def load(self):
        data_cfg = self.cfg.get("data", {})
        tfs = data_cfg.get("timeframes", ["1d"])
        for cls, items in self.cfg["universe"].items():
            if self.verbose:
                print(f"\n[{cls}]")
            for item in items:
                market, sym = markets.parse_instrument(item)
                try:
                    df = markets.load(market, sym, data_cfg)
                except Exception as e:
                    print(f"  [!] {sym} 取数出错: {type(e).__name__}: {e}")
                    df = None
                if df is None or df.empty:
                    self.failed.append(sym)
                    continue
                self.prices.setdefault(cls, {})[sym] = df
                got = []
                for tf in tfs:
                    conv = markets.to_timeframe(df, market, tf)
                    if conv is None:
                        continue
                    d, bpd = conv
                    spec = markets.spec_for(market, sym, d, self.specs, bpd, self.multipliers)
                    self.instruments.append(Instrument(cls, market, sym, tf, bpd, d, spec))
                    got.append(f"{tf}:{len(d)}")
                if self.verbose:
                    print(f"  {sym:<8} {df.index[0].date()} ~ {df.index[-1].date()}  ({', '.join(got) or '无可用周期'})")
        if not self.instruments:
            raise RuntimeError("没有可用的品种数据")
        return self

    @property
    def timeframes(self):
        return list(dict.fromkeys(i.tf for i in self.instruments))

    def _allow(self, inst: Instrument, filt: Filter):
        if filt is None or isinstance(filt, NoFilter):
            return None
        k = (filt.label(), inst.market, inst.symbol, inst.tf)
        if k not in self._filter_cache:
            ctx = FilterContext(inst.market, inst.symbol, inst.spec.days_per_year)
            self._filter_cache[k] = filt.allow(inst.df, inst.bars_per_day, ctx)
        return self._filter_cache[k]

    # 信号 + 过滤 + 执行 → 组合
    def run(self, signal, filt: Filter = None, slip_mult: float = 1.0, verbose: bool = None,
            start: str = None) -> PortfolioResult:
        from dataclasses import replace
        verbose = self.verbose if verbose is None else verbose
        results, rets, tf_rets = {}, {}, {}
        for inst in self.instruments:
            spec = inst.spec if slip_mult == 1.0 else replace(
                inst.spec, slippage=inst.spec.slippage * slip_mult,
                slippage_ticks=inst.spec.slippage_ticks * slip_mult)
            allow = self._allow(inst, filt)
            if isinstance(signal, Blend):
                r = self._run_blend(inst, spec, signal, allow)
            else:
                r = run_backtest(inst.df, inst.bars_per_day, spec, signal, allow)
            if r["n_trades"] == 0:
                continue
            key = inst.key if len(self.timeframes) > 1 else inst.symbol
            r["_inst"] = inst
            results[key] = r
            dr = r["daily_returns"]
            if start:
                dr = dr[dr.index >= pd.Timestamp(start)]
            ivol = float(self.pcfg.get("instrument_vol", 0) or 0)
            if ivol > 0 and len(dr) > 60:
                dr = vol_scale(dr, ivol, obs_per_year(dr))[0]
            rets.setdefault(inst.cls, {})[key] = dr
            tf_rets.setdefault(inst.tf, {}).setdefault(inst.cls, {})[key] = dr
            if verbose:
                print(f"  {key:<14} 交易:{r['n_trades']:>4}  年化:{r['annual_return_pct']:>+6.1f}%"
                      f"  夏普:{r['sharpe']:>5.2f}  回撤:{r['max_drawdown_pct']:>6.1f}%"
                      + (f"  过滤挡掉:{r['blocked']}" if r.get("blocked") else ""))
        weighting = self.pcfg.get("weighting", "class")
        port, C, I = combine(rets, weighting)
        by_tf = {tf: combine(d, weighting)[0] for tf, d in tf_rets.items()} if len(tf_rets) > 1 else {}
        res = PortfolioResult(signal.label(), (filt or NoFilter()).label(), results, port, C, I,
                              by_tf, obs_per_year(port))
        tv = float(self.pcfg.get("target_vol", 0) or 0)
        if tv > 0 and len(port) > 80:
            res.scaled, res.leverage = vol_scale(port, tv, res.dpy)
        return res

    @staticmethod
    def _run_blend(inst: Instrument, spec: MarketSpec, blend: Blend, allow) -> Dict:
        """每个子信号独立回测 (各自一份本金), 日收益等权平均; 汇总成和单信号相同格式的结果."""
        subs = [run_backtest(inst.df, inst.bars_per_day, spec, p, allow) for p in blend.parts]
        rets = [s["daily_returns"] for s in subs if len(s["daily_returns"])]
        out = {"n_trades": sum(s["n_trades"] for s in subs), "trades": [t for s in subs for t in s["trades"]],
               "blocked": sum(s.get("blocked", 0) for s in subs), "sub_results": subs}
        if len(rets) < len(blend.parts):
            out["n_trades"] = 0                               # 有子信号数据不够, 这个品种不参与
            return out
        start = max(r.index[0] for r in rets)                 # 最慢的子信号开始后才算
        R = pd.concat(rets, axis=1).loc[start:].fillna(0.0)
        dr = R.mean(axis=1)
        eq = (1 + dr).cumprod() * spec.initial_capital
        years = len(dr) / (spec.days_per_year * min(inst.bars_per_day, 1))
        ann = (eq.iloc[-1] / spec.initial_capital) ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else 0
        perf = performance(eq, spec.days_per_year * min(inst.bars_per_day, 1))
        pnl = [t["pnl"] for t in out["trades"]]
        gl = abs(sum(p for p in pnl if p <= 0))
        dd = perf["max_drawdown_pct"]
        out.update({
            "equity": eq, "daily_returns": dr,
            "annual_return_pct": round(ann * 100, 2), "sharpe": perf["sharpe"],
            "ann_vol_pct": perf["ann_vol_pct"], "sortino": perf["sortino"], "max_drawdown_pct": dd,
            "calmar": round(ann * 100 / abs(dd), 3) if dd else 0,
            "profit_factor": round(sum(p for p in pnl if p > 0) / gl, 3) if gl > 0 else float("inf"),
            "exposure_pct": round(float(np.mean([s["exposure_pct"] for s in subs])), 1),
        })
        return out

    def benchmark(self, index: pd.DatetimeIndex) -> pd.Series:
        """同样的品种和类别权重, 买入持有."""
        b = {cls: {s: price_returns(df) for s, df in d.items()} for cls, d in self.prices.items()}
        bench, _, _ = combine(b, self.pcfg.get("weighting", "class"))
        return bench.reindex(index).fillna(0.0)
