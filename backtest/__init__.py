"""
quant/backtest — 分层回测框架 (各策略项目共用)

  第 1 层 数据      quant/data/ (取数、缓存、复权) + backtest/markets.py (各市场交易规则)
  第 2 层 信号      backtest/signals.py    唐奇安 / TSMOM / 均线 ..., 用字符串切换: "tsmom:252"
  第 3 层 过滤      backtest/filters.py    不过滤 / GARCH / VIX: "vix:low,80"
  执行             backtest/engine.py     仓位、止损、成本、盯市
  第 4 层 组合      backtest/portfolio.py  跨资产、跨周期组合, 波动率缩放
  第 5 层 稳定性    backtest/robustness.py 信号/过滤器对比、参数网格、样本内外、自助法、成本
  第 6 层 报告      backtest/report.py + analytics.py

入口 (在 quant 根目录): python -m backtest strategies/<策略>/configs/<实验>.toml
自己写的信号: 放在 strategies/<策略>/signals.py, 配置里写 signal_module; 模板见 strategies/_template/
"""
