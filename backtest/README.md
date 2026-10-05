# backtest — 分层回测框架

所有策略共用。你只需要写**信号**和**配置文件**，其余 (数据、执行、组合、稳定性检验、报告) 都由框架负责。

```
第 1 层 数据        quant/data/ + markets.py      "市场:代码" → K线; 各市场交易规则
第 2 层 信号        signals.py (+ 你自己的)         什么时候开多 / 开空 / 平仓
第 3 层 过滤        filters.py                     现在允不允许开新仓 (garch / vix)
执行               engine.py                      1%/ATR 仓位、止损、手续费、滑点、逐根盯市
第 4 层 组合        portfolio.py                   跨资产、跨周期; 波动率缩放
第 5 层 稳定性      robustness.py                  换信号 / 换过滤器 / 信号×过滤器组合普查 / 参数网格 / 样本内外 / 自助法 / 成本
第 6 层 报告        report.py + analytics.py       Markdown + 图 + 日收益 CSV
入口               __main__.py                    python -m backtest <配置>
```

---

## 怎么运行

在 **quant 根目录** 打开终端：

```
cd E:\haco_mills\08-Vscode\Project\quant
python -m backtest strategies/turtle/configs/cross_asset.toml
```

(`No module named backtest` = 终端不在 quant 根目录。也可以在任何位置运行 `python <quant路径>\backtest\__main__.py <配置>`。)

命令行参数优先于配置文件：

| 参数 | 例子 |
|---|---|
| `--signal` | `--signal tsmom:252` |
| `--signal-module` | `--signal-module strategies/_template/signals.py` (临时加载自己的信号) |
| `--filter` | `--filter vix:low,80` |
| `--timeframes` | `--timeframes 4h,1d` |
| `--symbols` | `--symbols future:RB,future:CU` |
| `--start` | `--start 2019-01-01` |
| `--target-vol` | `--target-vol 0` |
| `--weighting` | `--weighting instrument` |
| `--robustness` / `--no-robustness` | 打开 / 关掉稳定性检验 |
| `--name` | 报告文件名 |
| `--refresh` | 重新下载数据 |

报告位置：`quant/reports/<策略文件夹名>/<name>_report.md`，图片在 `quant/figures/<策略文件夹名>/`。

---

## 写一个新策略 (3 步)

### 1. 复制模板

把 `strategies/_template/` 整个复制一份，改成英文名，比如 `strategies/bollinger/`：

```
strategies/bollinger/
├── signals.py           你的信号
└── configs/
    └── example.toml     你的实验
```

### 2. 写信号 (`signals.py`)

```python
import numpy as np
from backtest.signals import Signal, SignalArrays, _bars

class Bollinger(Signal):
    name = "boll"             # 配置里用的名字
    default_stop = 2.0        # 默认 2N 止损; None = 不止损

    def __init__(self, window_days=20, width=2.0, stop=None):
        super().__init__(stop)
        self.window_days, self.width = window_days, width

    def label(self):
        return f"boll:{self.window_days:g},{self.width:g}"

    def compute(self, df, bars_per_day):
        n = _bars(self.window_days, bars_per_day)       # 天数 → K线根数
        c = df["close"]
        mid, sd = c.rolling(n).mean(), c.rolling(n).std()
        upper, lower = (mid + self.width * sd).values, (mid - self.width * sd).values
        c, mid = c.values, mid.values
        with np.errstate(invalid="ignore"):
            return SignalArrays(
                long_entry=c > upper,      # 空仓时: 开多
                short_entry=c < lower,     # 空仓时: 开空
                long_exit=c < mid,         # 持多时: 平仓
                short_exit=c > mid,        # 持空时: 平仓
                warmup=n,                  # 前 n 根不产生信号
            )
```

规则：
- `compute()` 返回四个布尔数组，长度和 K线一样
- **只能用当根及以前的数据**。`rolling`、`shift(1)` 都可以；`shift(-1)` 是偷看未来，绝对不行
- 参数用"天"做单位，`_bars()` 换算成根数，这样同一个信号可以直接跑 1h / 4h / 1d
- `df` 的列：`open high low close`，复权数据还有 `raw_close`、`scale` (真实价格换算，一般用不到)

### 3. 写配置 (`configs/example.toml`)

```toml
name  = "example"
title = "布林带突破"

[universe]
"商品" = ["future:RB", "future:CU", "future:AU"]
"汇率" = ["fx:EURUSD"]

[data]
start      = "2019-01-01"
timeframes = ["1d"]

[strategy]
signal_module = "strategies/bollinger/signals.py"     # 你的信号在哪
signal = "boll:20,2"
filter = "none"

[portfolio]
weighting  = "class"
target_vol = 0.10

[robustness]
enabled = true
signals = ["boll:20,2", "donchian:60,20", "tsmom:252"]  # 和框架自带的信号放在一起比
filters = ["none", "vix:low,80"]

[robustness.grid]
boll = [[10, 20, 40, 60], [1.5, 2, 2.5]]              # 参数网格
```

运行：

```
python -m backtest strategies/bollinger/configs/example.toml
```

完整的配置项说明 (品种写法、各市场默认成本、组合方式、稳定性检验) 见 `strategies/turtle/README.md` 第六节。

---

## 自带的信号和过滤器

| 信号 | 规则 | 止损 |
|---|---|---|
| `donchian:60,20` | 突破 60 天高 / 低点入场，反向突破 20 天通道出场 | 2N |
| `tsmom:252` | 过去 252 天涨就做多、跌就做空，一直在场 | 无 |
| `ma:50,200` | 均线交叉 | 无 |
| `boll:20,2` | 突破 20 天均线 ± 2 倍标准差入场，回到均线出场 | 2N |
| `keltner:20,2` | 突破 20 天 EMA ± 2 倍 ATR 入场，回到 EMA 出场 | 2N |
| `supertrend:10,3` | ATR(10 天) × 3 的跟踪线，价格穿线就反手 | 无 |
| `ma3:10,30,100` | 三条均线快 > 中 > 慢做多、反之做空，快线穿回中线出场 | 无 |
| `regress:60,2` | 过去 60 天对数价格回归斜率的 t 值 > 2 做多、< -2 做空，回到 0 出场 | 无 |
| `riskmom:252,0.5` | 过去 252 天收益 ÷ 同期波动 > 0.5 做多、< -0.5 做空，回到 0 出场 | 无 |
| `knrp:9,2` | 网上流传的"随机指标 × RSI ÷ 动量"组合，2 根均线上升做多、下降做空（翻转很频繁） | 无 |
| `emavwap:200,0.5,60,40,3` | 突破 200 天 EMA ± 0.5%，且价格在 20 天滚动 VWAP 同侧、MFI > 60 / < 40、大周期 EMA 同向、连续 3 根同向；回到 EMA 出场。需要成交量，外汇和只有收盘价的 A 股数据不产生信号 | 2N |

任何信号都可以加 `stop=N` 改止损倍数 (`stop=0` 不止损)。用 `+` 连接多个信号 = 多周期混合 (各自独立跑，日收益平均)，自己写的信号也可以参与混合。

| 过滤器 | 规则 |
|---|---|
| `none` | 不过滤 |
| `garch` | GARCH 波动率高于 20 天均值才开仓 (滚动估计) |
| `vix:low,80` / `vix:high,50` | VIX 低于 80% 分位 / 高于中位数才开仓；期货、A股用中国 QVIX，外汇、加密用美国 VIX (滞后一天) |

自定义过滤器也一样：在 `signal_module` 指向的文件里继承 `backtest.filters.Filter`，设置 `name`，实现 `allow()`。

---

## 信号普查：一次测完所有信号、时间窗口和过滤器

```
python -m backtest strategies/turtle/configs/signal_scan.toml     # 跨资产日线, 约 34 个信号 × 4 个过滤器, 二十分钟左右
python -m backtest strategies/turtle/configs/crypto_scan.toml     # 加密 4h, 半小时以上
```

配置里 `[robustness] combos = true` 时，每个信号 × 每组时间窗口 × 每个过滤器各跑一遍：

```toml
[robustness]
combos  = true
periods = ["2018-01-01", "2020-01-01", "2022-01-01", "2024-01-01"]   # 分时期看夏普 (不写就按年)
signals = ["donchian:60,20", "tsmom:252"]                             # 7.1 换信号对比用的代表参数
filters = ["none", "garch", "vix:low,80", "vix:high,50"]

[robustness.windows]              # 普查时额外展开的时间窗口 (单位: 天)
donchian = ["20,10", "60,20", "120,40"]
tsmom    = ["63", "126", "252"]
```

`[robustness.windows]` 必须放在 `[robustness]` 其他配置项的后面 (TOML 规则：子表之后的键都属于子表)。

报告第七节给出：

- 每个信号的整体表现，以及在股指 / 国债 / 商品 / 汇率各类别上的夏普
- 信号 × 过滤器的样本内夏普矩阵和样本外夏普矩阵
- 按样本内夏普挑出的前 5 个组合，以及它们到样本外还剩多少
- **按信号类型汇总**：每类信号的全部窗口 × 全部过滤器的中位数表现 (比单个最优参数更可信)
- **分时期夏普**：每个组合在各时期的夏普、最差时期、盈利时期占比
- 全部组合的明细另存为 `reports/<策略>/<name>_combos.csv`，可以用 Excel 排序筛选

试的组合越多，样本内最好的那个越可能只是运气。判断标准是：前几名到样本外，是否仍明显高于全部组合的中位数。
