"""
外汇利差 (carry) 策略
=====================
规则在看结果之前固定, 之后不调参:
  信号   外币 3 个月利率 − 美元 3 个月利率, 用两个月前的月度数据 (月均利率下个月才公布, 再留一个月余量);
         > 0 做多外币, < 0 做空外币
  收益   汇率变动 + 每天计提的实际利差 (真正持有外币时赚的是当时的利差)
  仓位   每个货币对按 1 / 过去 60 天波动 (滞后一天) 加权, 风险均等; 组合缩放到 10% 波动
  成本   每次换方向按单边 1bp
  区间   2002–2014 样本内, 2015 起样本外 (与 README 里汇率趋势的分段一致)

数据: 汇率 = 东方财富日线 (data/cache/fx), 利率 = OECD 月度 3 个月银行间利率 (IR3TIB).
运行 (在 quant 根目录):
    python strategies/turtle/fx_carry.py
"""

import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backtest.report import md_table                              # noqa: E402
from data.paths import FX_CACHE, INDEX_CACHE, REPORTS_DIR         # noqa: E402

# 货币对 → (外币利率代码, 做多外币时货币对的方向: +1 = XXXUSD 做多, −1 = USDXXX 做空)
PAIRS = {"EURUSD": ("EA20", 1), "GBPUSD": ("GBR", 1), "AUDUSD": ("AUS", 1),
         "USDJPY": ("JPN", -1), "USDCNH": ("CHN", -1)}
START, IS_END = "2002-01-01", "2014-12-31"
COST = 0.0001
DAYS = 260                                                        # 外汇一年约 260 个交易日


def load_rates(refresh: bool = False) -> pd.DataFrame:
    path = INDEX_CACHE / "oecd_3m_rates.csv"
    if not path.exists() or refresh:
        import requests
        url = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
               "USA+EA20+JPN+AUS+GBR+CHN.M.IR3TIB.PA.....?startPeriod=1998-12&format=csvfilewithlabels")
        d = pd.read_csv(io.StringIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=90).text))
        d.pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE").to_csv(path)
    p = pd.read_csv(path, index_col=0)
    p.index = pd.PeriodIndex(p.index, freq="M")
    return p.ffill() / 100                                          # 百分比 → 小数; 个别月份缺失沿用上月


def to_daily(monthly: pd.DataFrame, index: pd.DatetimeIndex, lag_months: int) -> pd.DataFrame:
    """月度数据 → 日度: 第 m 月的值从第 m + lag_months 月的第一天开始生效."""
    m = monthly.copy()
    m.index = (m.index + lag_months).to_timestamp()
    return m.reindex(m.index.union(index)).ffill().reindex(index)


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {}
    eq = (1 + r).cumprod()
    return {"夏普": r.mean() / r.std() * math.sqrt(DAYS), "年化%": r.mean() * DAYS * 100,
            "波动%": r.std() * math.sqrt(DAYS) * 100, "最大回撤%": (eq / eq.cummax() - 1).min() * 100}


def main():
    rates = load_rates()
    close = pd.DataFrame({p: pd.read_csv(FX_CACHE / f"fx_{p}_daily.csv", parse_dates=["timestamp"],
                                         index_col="timestamp")["close"] for p in PAIRS})
    close = close.loc[close.index >= "2001-06-01"]
    idx = close.index

    spot, carry, signal = {}, {}, {}
    actual = to_daily(rates, idx, 0)                               # 当月实际利率: 用于计提利息
    lagged = to_daily(rates, idx, 2)                               # 两个月前的利率: 用于决定方向
    for p, (ccy, side) in PAIRS.items():
        spot[p] = side * close[p].pct_change(fill_method=None)       # 做多外币的汇率收益
        carry[p] = (actual[ccy] - actual["USA"]) / DAYS               # 做多外币每天赚的利差
        signal[p] = np.sign(lagged[ccy] - lagged["USA"])
    spot, carry, signal = pd.DataFrame(spot), pd.DataFrame(carry), pd.DataFrame(signal)
    total = (spot + carry).where(spot.notna())                       # 做多外币的总收益

    pos = signal.shift(1)                                          # 昨天收盘定的方向
    vol = total.ewm(com=60, min_periods=40).std().shift(1)
    risk_w = (1 / vol).div((1 / vol).sum(axis=1), axis=0)          # 只在有数据的货币对之间分配
    w = (pos * risk_w).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1)
    port = (w * total.fillna(0)).sum(axis=1) - turnover * COST
    port_spot = (w * spot.fillna(0)).sum(axis=1)                   # 只看汇率部分 (不含利息)
    lev = (0.10 / (port.ewm(com=60, min_periods=40).std() * math.sqrt(DAYS))).clip(upper=5).shift(1)
    port10 = port * lev

    # 对照: 不看利差, 一直做多外币 (买入持有), 含利息 vs 不含利息
    bh_total = (risk_w * total.fillna(0)).sum(axis=1)
    bh_spot = (risk_w * spot.fillna(0)).sum(axis=1)

    periods = [("样本内 2002–2014", START, IS_END), ("样本外 2015–2026", "2015-01-01", None),
               ("全部", START, None)]
    rows = []
    for name, s in (("利差策略 (10% 波动)", port10), ("利差策略 汇率部分", port_spot),
                    ("一直做多外币 含利息", bh_total), ("一直做多外币 不含利息", bh_spot)):
        for pn, a, b in periods:
            rows.append({"": name, "区间": pn, **stats(s.loc[a:b])})
    tab = pd.DataFrame(rows)

    pair_rows = []
    for p in PAIRS:
        r = (w[p] * total[p].fillna(0))
        r = r[w[p] != 0]
        for pn, a, b in periods[:2]:
            s = stats(r.loc[a:b])
            if s:
                pair_rows.append({"货币对": p, "区间": pn, "夏普": s["夏普"],
                                  "做多外币天数占比%": (signal[p].loc[a:b] > 0).mean() * 100})
    pair_tab = pd.DataFrame(pair_rows)

    crises = [("2008 金融危机", "2008-08-01", "2008-12-31"), ("2015 人民币 8·11 汇改", "2015-08-10", "2015-08-31"),
              ("2020 疫情冲击", "2020-02-20", "2020-03-31"), ("2024-08 日元套息平仓", "2024-07-31", "2024-08-07")]
    crisis_tab = pd.DataFrame([{"时期": n, "区间": f"{a} ~ {b}",
                                "利差策略%": ((1 + port10.loc[a:b]).prod() - 1) * 100} for n, a, b in crises])

    pd.set_option("display.width", 200, "display.max_columns", 20)
    for t in (tab, pair_tab, crisis_tab):
        print(t.round(2).to_string(index=False)); print()

    L = ["# 外汇利差 (carry) 策略", "", *[l.strip() for l in __doc__.strip().splitlines()[2:13]], "",
         "## 组合", "", *md_table(tab), "", "## 各货币对", "", *md_table(pair_tab), "",
         "## 压力时期", "", *md_table(crisis_tab)]
    out = REPORTS_DIR / "turtle" / "fx_carry_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    port10.rename("fx_carry").to_csv(REPORTS_DIR / "turtle" / "fx_carry_returns.csv")
    print(f"报告: {out}")


if __name__ == "__main__":
    main()
