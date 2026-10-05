# data — 数据与取数代码

所有项目共用这里的数据。**项目文件夹里不放原始数据。**

## 文件夹结构

```
data/
├── paths.py          所有路径的统一定义 (改文件夹结构只改这里)
├── fetch_data.py     币圈取数: OKX K线、资金费率、持仓量
├── market_data.py    A股 / 期货 / 外汇 / VIX 取数: CSMAR、AKShare、FRED、本地 CSV
├── .env              OKX API Key (不要上传或分享)
│
├── raw/              ★ 手动下载的原始数据 —— 重要, 不要删
│   ├── securities/   CSMAR 导出的 Excel (日度 TRD_Dalyr*, 月度 TRD_Mnth, 三因子 STK_MKT_THRFACMONTH)
│   ├── vix/          (可选) CBOE 官网下载的 VIX_History.csv, 网络拿不到 VIX 时用
│   └── btc_usdt_15m_2022_to_now.csv 等
│
└── cache/            脚本自动下载 / 生成的缓存 —— 可以随时删, 删了会重新下载
    ├── okx/          币圈 K线 (如 BTC_USDT_USDT_4h.csv, btcusdt_usdt_15m_2022-01-01.csv)
    ├── cn_futures/   国内期货 (contracts/ 下是单月合约)
    ├── cn_stock/     A股 (AKShare)
    ├── csmar/        CSMAR 日度 Excel 合并后的 TRD_Dalyr_all.pkl
    ├── fx/           外汇日线
    └── index/        波动率指数: 美国 VIX、中国 QVIX (每天更新一次)
```

## 规则

| 数据是怎么来的 | 放哪 |
|---|---|
| 自己从 CSMAR / Wind / 网站手动下载的 | `raw/` |
| 脚本自动下载的、或由原始数据加工出来的缓存 | `cache/` (代码会自动放) |
| 某个项目算出来的中间结果 (如 ff3 的 `*.pkl`) | 留在项目自己的文件夹 |

## 在代码里怎么用

```python
from data.paths import raw, OKX_CACHE, figures, reports

raw("securities")        # quant/data/raw/securities
OKX_CACHE                # quant/data/cache/okx
figures("turtle")        # quant/figures/turtle   (自动创建)
reports("turtle")        # quant/reports/turtle   (自动创建)
```

`raw(...)` 兼容旧位置：文件还没挪进 `raw/` 时，会自动去 `data/` 下找。
