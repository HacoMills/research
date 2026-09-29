# quant — 量化研究

```
quant/
├── data/         数据 (raw 手动下载 / cache 自动下载) 和取数代码, 见 data/README.md
├── backtest/     分层回测框架 (数据接口→信号→过滤→执行→组合→稳定性→报告), 各项目共用
├── strategies/   代码, 每个项目一个文件夹, 各自有 README
├── figures/      图片, 按项目分文件夹
├── reports/      报告, 按项目分文件夹
└── notes/        学习笔记、临时实验
```

## 项目

| 项目 | 内容 | 状态 |
|---|---|---|
| `strategies/turtle/` | 趋势策略：回测配置 (唐奇安 / TSMOM / 均线, GARCH / VIX 过滤)，实盘信号、看盘 | 进行中，见 README |
| `strategies/_template/` | 新策略模板：复制一份改名即可 | |
| `strategies/15min_garch/` | BTC 15 分钟 GARCH + XGBoost 方向预测 | 已结束 (方向 alpha 被证伪) |
| `strategies/ff3/` | A股 Fama-French 三因子 | |
| `strategies/6hmom/` | 6 小时动量 | |

## 回测

所有策略共用 `backtest/` 框架，在 quant 根目录运行：

```
python -m backtest strategies/turtle/configs/cross_asset.toml
python -m backtest strategies/<策略>/configs/<实验>.toml
```

写新策略见 `backtest/README.md`。

## 规则

1. **代码**放 `strategies/<项目>/`，回测配置放 `strategies/<项目>/configs/`
2. **原始数据**放 `data/raw/`，**自动缓存**放 `data/cache/` (代码会自动放)
3. **图片**放 `figures/<项目>/`，**报告**放 `reports/<项目>/`
4. 每个项目一个 `README.md`：干什么、怎么运行、结果在哪
5. 旧版本代码不删，放进项目的 `archive/`
6. 所有路径从 `data/paths.py` 取，不在脚本里写死
