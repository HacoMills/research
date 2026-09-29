# 海龟 / 趋势策略 (Turtle)

回测用共用的**分层框架** (`quant/backtest/`，说明见 `backtest/README.md`)：信号、过滤器、品种池、周期都能切换，不用改代码。
支持加密货币 (OKX 永续)、A股 (CSMAR / AKShare)、国内期货 (新浪单月合约自拼复权连续)、外汇。

---

## 一、先看这里

| 文件 | 干什么 |
|---|---|
| `configs/*.toml` | **回测配置**：一份配置 = 一次实验，用 `python -m backtest` 运行 |
| `turtle_signal.py` | **实盘信号**：现在该开仓 / 平仓吗，记录持仓 |
| `turtle_dashboard_garch.py` | **网页看盘**：信号 + GARCH 波动率曲线，可推送 Telegram |

其余文件：

| 文件 | 用途 |
|---|---|
| `turtle_screener.py` | 选币：批量扫描哪些币适合跑海龟 |
| `alpha_test.py` | 检验策略是否有 alpha：CAPM 回归、逐年分解、蒙特卡洛置换检验 |
| `test_okx.py` | 测试 OKX API 能否连上 |
| `config.json` | OKX API Key、代理、Telegram 配置 (**不要上传或分享**) |
| `archive/` | 旧脚本 (`turtle_timeframe_backtest.py`、`turtle_multi_asset.py`、`turtle_cross_asset.py` 等)，在 archive 里仍可运行 |

准备 (只需一次)：

```
pip install pandas numpy arch ccxt matplotlib streamlit plotly requests
pip install akshare python-calamine        # A股 / 期货 / 外汇 / QVIX
```

需要 Python 3.11 及以上 (读配置文件用到 `tomllib`)。

---

## 二、分层结构

```
quant/
├── data/                  第 1 层 数据: 取数、缓存、复权 (market_data.py, fetch_data.py)
├── backtest/
│   ├── markets.py         第 1 层接口: "市场:代码" → K线; 各市场交易规则 (手续费、滑点、乘数、涨跌停)
│   ├── signals.py         第 2 层 信号: 什么时候开多 / 开空 / 平仓
│   ├── filters.py         第 3 层 过滤: 现在允不允许开新仓
│   ├── engine.py          执行: 1%/ATR 仓位、止损、成本、逐根盯市
│   ├── portfolio.py       第 4 层 组合: 跨资产、跨周期, 波动率缩放
│   ├── robustness.py      第 5 层 稳定性检验
│   ├── report.py          第 6 层 报告 (Markdown + 图 + 日收益 CSV)
│   └── analytics.py       报告用的统计、回归、画图
│   └── __main__.py        统一入口: python -m backtest <配置>
└── strategies/turtle/
    └── configs/           cross_asset / futures / stock / crypto
```

每层只做一件事，都只用当时已知的数据 (已做截断测试)。
唐奇安 60/20、不过滤时，新框架与旧引擎的结果逐位一致。

---

## 三、怎么跑

在 VS Code 终端进入 **quant 根目录** (不是 turtle 文件夹)：

```
cd E:\haco_mills\08-Vscode\Project\quant

python -m backtest strategies/turtle/configs/cross_asset.toml   # 跨资产 (含稳定性检验)
python -m backtest strategies/turtle/configs/futures.toml       # 国内期货
python -m backtest strategies/turtle/configs/stock.toml         # A股
python -m backtest strategies/turtle/configs/crypto.toml        # 加密货币, 1h + 4h + 1d 三个周期
```

命令行可以临时覆盖配置 (优先于配置文件)：

| 参数 | 例子 | 作用 |
|---|---|---|
| `--signal` | `--signal tsmom:252` | 换信号 |
| `--signal-module` | `--signal-module strategies/_template/signals.py` | 临时加载自己写的信号 |
| `--filter` | `--filter vix:low,80` | 换过滤器 |
| `--timeframes` | `--timeframes 4h,1d` | 换周期 |
| `--symbols` | `--symbols future:RB,future:CU` | 临时品种池 |
| `--start` | `--start 2019-01-01` | 起始日期 |
| `--target-vol` | `--target-vol 0` | 组合目标波动，0 = 不缩放 |
| `--weighting` | `--weighting instrument` | 组合方式 |
| `--robustness` / `--no-robustness` | | 打开 / 关掉稳定性检验 |
| `--name` | `--name test1` | 报告文件名 |
| `--refresh` | | 重新下载数据 |

临时换了信号或过滤器时，报告会另存一份 (如 `cross_asset__tsmom-252_report.md`)，不覆盖原来的。

---

## 四、信号 (第 2 层)

参数单位都是**天**，会按周期自动换算成 K线根数 (4h 周期的 60 天 = 360 根)。

| 写法 | 规则 | 止损 |
|---|---|---|
| `donchian:60,20` | 收盘价突破过去 60 天最高 / 最低价入场，反向突破 20 天通道出场 (海龟系统 2) | 2N |
| `donchian:20,10` | 海龟系统 1 | 2N |
| `tsmom:252` | 时间序列动量 (TSMOM 论文)：过去 252 天涨就做多、跌就做空，一直在场 | 无 |
| `ma:50,200` | 50 天均线在 200 天均线上方做多，下方做空 | 无 |

**多周期混合**：用 `+` 连接多个信号，如
`donchian:20,10 + donchian:60,20 + donchian:120,40 + tsmom:63 + tsmom:126 + tsmom:252`。
每个品种上各子信号独立跑 (各一份本金)，日收益等权平均。不用猜哪个资产适合哪个周期，也不增加挑参数的自由度。

附加 `stop=N` 改止损：`donchian:60,20,stop=3` 放宽到 3N，`tsmom:252,stop=2` 给动量加 2N 止损，`stop=0` 不止损。

**新增信号**：不用改框架。在自己的策略文件夹写 `signals.py` (继承 `Signal`，实现 `compute()` 返回开多、开空、平多、平空四个布尔数组)，
配置里写 `signal_module = "strategies/xxx/signals.py"`。模板和详细步骤见 `strategies/_template/` 和 `backtest/README.md`。

---

## 五、过滤器 (第 3 层)

过滤器只决定"能不能开新仓"，已有的仓位照常平仓。

| 写法 | 规则 |
|---|---|
| `none` | 不过滤 |
| `garch` | GARCH(1,1) 条件波动率高于过去 20 天均值才开仓 (滚动估计，第一年只训练不过滤) |
| `vix:low,80` | VIX 低于过去一年的 80% 分位才开仓 (避开恐慌期) |
| `vix:high,50` | VIX 高于过去一年的中位数才开仓 (只在波动大时做趋势) |
| `vix:low,80,source=us` | 指定指数：`us` 美国 VIX / `qvix50` / `qvix300` |
| `vix:low,80,window=504` | 分位数用过去 504 天 |
| `vix:low,level=20` | 用固定点位：VIX 低于 20 才开仓 |

VIX 数据源 (默认 `source=auto`，按市场自动选)：

| 市场 | 用哪个 | 时间对齐 |
|---|---|---|
| 国内期货、A股 | 中国 QVIX (50ETF 期权波动率指数，2015 年起，AKShare) | 当天收盘可得，当天用 |
| 外汇、加密 | 美国 VIX (FRED → CBOE → 本地 `data/raw/vix/VIX_History.csv`) | 美股收盘在北京时间凌晨，**只用前一天**的值 |

VIX 数据开始之前的日子不过滤。
⚠️ 趋势策略往往在危机中赚得最多 (smile)，"VIX 高时不开仓"可能适得其反，用稳定性检验里的过滤器对比来判断。

---

## 六、配置文件

以 `configs/cross_asset.toml` 为例：

```toml
name  = "cross_asset"                 # 报告文件名
title = "跨资产组合回测"

[universe]                            # "类别" = ["市场:代码", ...]  市场: future / fx / stock / crypto
"股指" = ["future:IF", "future:IH", "future:IC", "future:IM"]
"汇率" = ["fx:USDCNH", "fx:EURUSD"]

[data]
start          = "2015-01-01"
timeframes     = ["1d"]               # 加密: 15min/30min/1h/4h/1d/1w; 其他市场: 1d/1w
futures_adjust = "diff"               # diff 差值复权 / ratio 比例复权

[strategy]
signal = "donchian:60,20"
filter = "none"

[portfolio]
weighting      = "class"              # class 两层等权 / instrument 全部等权
target_vol     = 0.10                 # 组合目标波动, 0 = 不缩放
instrument_vol = 0                    # 论文做法: 每个品种先缩放到 0.40

[costs.future]                        # 覆盖默认交易规则
initial_capital = 10_000_000

[robustness]
enabled   = true
oos_start = "2022-01-01"
signals   = ["donchian:60,20", "tsmom:252", "ma:50,200"]
filters   = ["none", "garch", "vix:low,80", "vix:high,50"]

[robustness.grid]
donchian = [[20, 40, 60, 100, 150], [10, 20, 30, 50]]
```

TOML 里中文的键名要加引号 (`"股指" = ...`)。

### 默认交易规则 (`[costs.<市场>]` 可覆盖任意一项)

| 市场 | 手续费 (单边) | 其他 | 滑点 | 做空 | 每品种本金 |
|---|---|---|---|---|---|
| crypto | 5‱ | | 3‱ | 可以 | 10,000 |
| future | 1‱ (国债 3 元/手) | 整手，合约乘数按品种 | 1 跳 (最小变动价位) | 可以 | 10,000,000 |
| stock | 2.5‱ | 卖出印花税 5‱，100 股一手，现金账户，涨跌停 | 10‱ | 不可以 | 1,000,000 |
| fx | 0.2‱ | 不含利差 | 1‱ | 可以 | 1,000,000 |

内置表里没有的期货品种，在配置里加 `[multipliers]`，如 `XX = 10`。

期货滑点按**最小变动价位**算 (表在 `backtest/markets.py` 的 `FUTURE_TICKS`)：每次成交吃 1 跳，
`[costs.future] slippage_ticks = 2` 改成 2 跳。国债 1 跳只有约 0.2–0.4 个基点，按统一 3 个基点算会把国债成本高估近十倍。
表里没有最小变动价位的品种退回按 3 个基点计算，报告开头会列出来。

---

## 七、组合 (第 4 层)

**跨资产**：默认 `weighting = "class"`，两层等权：
1. 同一类别内的品种等权
2. 各类别再等权

这样商品十几个品种不会压倒其他类别。5 类都有时每个品种的权重：

| 类别 | 品种数 | 类别权重 | 每个品种 |
|---|---|---|---|
| 股指 | 4 | 20% | 5% |
| 国债 | 4 | 20% | 5% |
| 商品 | 13 | 20% | 约 1.5% |
| 汇率 | 5 | 20% | 4% |
| 加密 | 2 | 20% | 10% |

`instrument`：所有品种直接等权，商品合计会占近一半。

**跨周期**：`timeframes` 写多个周期时，同一品种的每个周期算一个成员 (如 `BTC@1h`、`BTC@4h`)，
报告里另有"各周期单独组合"和周期之间的相关性。

补充：
- 等权的是每个品种的**策略收益**。品种内部已按 1%/ATR 定仓位，风险大致相同；
  加 `instrument_vol = 0.40` 后每个品种先缩放到相同波动，等权就严格等于等风险 (论文做法)
- 每天只平均当时已有数据的品种 (IM 2022 年、TL 2023 年才上市)
- 某个市场休市时，已经开始交易的品种当天记 0 收益

**目标波动率** (`target_vol`)：品种多、相关性低时组合波动很低，期货和永续只交保证金，实际会加杠杆调到目标波动 (CTA 常用 10–15%)。
- 杠杆 = 目标波动 ÷ 事前估计的波动率，上限 10 倍
- 波动率用 EWMA (质心 60 天，与论文相同)，只用前一天及以前的数据
- 只用持仓日估计 (空仓日收益为 0，计入会低估波动率，再入场时杠杆暴涨)
- 论文里的 40% 是**单品种**的目标，不要把整个组合调到 40%

---

## 八、稳定性检验 (第 5 层)

`[robustness] enabled = true` 或 `--robustness` 打开，报告第七节：

| 检验 | 回答什么 |
|---|---|
| 换信号 | 同一批品种上唐奇安 / TSMOM / 均线谁更好，它们之间相关性多高 (低相关的可以组合在一起) |
| 换过滤器 | GARCH、VIX 到底有没有用；挡掉了多少入场 |
| 参数网格 | 夏普热力图：一片高原 = 参数不敏感；一根尖刺 = 过拟合 |
| 样本内外 | 样本内挑出的最好参数，到样本外还剩多少夏普 |
| 分年度 | 每年收益和夏普，赚钱年份占比 |
| 自助法 | 按 20 天分块重抽日收益 1000 次，夏普的 95% 置信区间 |
| 成本敏感性 | 滑点 ×0 / ×1 / ×2 / ×4 |

每种检验都要把整个品种池重跑一遍：跨资产 (日线) 一两分钟；加密 15 分钟周期会慢很多。

---

## 九、报告 (第 6 层)

| 内容 | 位置 |
|---|---|
| 报告 | `quant/reports/turtle/<name>_report.md` (VS Code 里 `Ctrl+Shift+V` 预览；文件夹名取自配置所在的策略文件夹) |
| 组合日收益 | `quant/reports/turtle/<name>_returns.csv` (策略、缩放后、买入持有基准) |
| 图片 | `quant/figures/turtle/<name>_portfolio.png` 净值与回撤、`_smile.png`、`_signals.png`、`_filters.png`、`_grid.png`、`_bootstrap.png` |

报告内容：
1. 各品种表现
2. 各周期单独组合 (多周期时)
3. 各资产类别、类别间相关性、类内 / 跨类相关性 (多类别时)
4. 目标波动率：原始 vs 缩放后
5. 压力时期 (2015 股灾、2020 疫情、2022 LUNA / FTX 等)
6. 组合 vs 等权买入持有、逐年收益、CAPM α、smile、波动率状态回归
7. 稳定性检验
8. 和沪深300 组合：单独持有沪深300 vs 按比例加入策略 / 叠加策略，比较夏普、回撤、股票下跌月份的表现 (配置 `[overlay]`)

---

## 十、实盘与看盘

```
python turtle_signal.py                             # 扫描当前信号
python turtle_signal.py --watch 300                 # 每 5 分钟自动刷新
python turtle_signal.py --status                    # 查看持仓
python turtle_signal.py --record BTC long 63200     # 记录开仓
python turtle_signal.py --close  BTC 65000          # 记录平仓
streamlit run turtle_dashboard_garch.py             # 网页看盘
```

---

## 十一、数据说明

- **期货**：新浪单月合约按前一天收盘的持仓量换月，自拼复权连续；新浪只保留 2018 年以后的到期合约。
  第一次每个品种要下载一百多个合约，二十多个品种可能要半小时，之后走缓存
- **A股**：默认读 `data/raw/securities/` 下的 CSMAR 日度文件。现有文件只有收盘价，通道和 ATR 按收盘价算；
  重新导出时勾选 `Opnprc / Hiprc / Loprc` 会自动用上真实开高低
- **外汇**：东方财富日线，只算价格变化，不含利差
- **加密**：OKX 15 分钟数据合成各周期，缓存与旧脚本共用
- 网络断开时自动改用已有缓存

---

## 十二、回测做了哪些处理 (面试可能会问)

**已处理**
- **逐根盯市**：回撤、夏普包含持仓浮亏 (BTC 4h 60/20：只在平仓时更新的旧算法回撤 11.7%，盯市后 24.0%)
- **止损跳空**：开盘价已越过止损位时按开盘价成交；每次成交按不利方向加滑点
- **无前视偏差**：信号只用当根及以前的数据；GARCH 滚动估计；美国 VIX 滞后一天；目标波动率用前一天的估计；都做过截断测试
- **期货换月**：按已知的持仓量决定次日换月；信号用复权价，手数、手续费、盈亏按当时真实合约价格
- **A股复权**：默认后复权 (以后的分红不会改写历史)；手数和费用按不复权价格
- **交易规则**：A股不能做空、整手、涨跌停、印花税；期货合约乘数、整张
- **参数稳健性**：参数网格 + 样本内外检验

**还没处理**
- 币圈永续合约的资金费率
- 期货换月时平旧开新的额外成本
- 期货保证金占用 (各品种独立本金，没有模拟共用账户)
- 海龟原版的加仓 (每 0.5N 加一个单位，最多 4 个)
