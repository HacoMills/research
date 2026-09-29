#!/usr/bin/env python3
"""
统一回测入口
============
一份配置文件 = 一次实验 (品种池、周期、信号、过滤器、组合方式、稳定性检验), 放在 configs/ 下.

用法:
  python run.py configs/cross_asset.toml                      # 按配置跑
  python run.py configs/cross_asset.toml --signal tsmom:252   # 临时换信号
  python run.py configs/futures.toml --filter vix:low,80      # 临时加 VIX 过滤
  python run.py configs/crypto.toml --timeframes 4h,1d        # 临时换周期
  python run.py configs/stock.toml --symbols stock:600519,stock:000001
  python run.py configs/cross_asset.toml --robustness         # 打开稳定性检验
  python run.py configs/cross_asset.toml --no-robustness      # 关掉稳定性检验 (快)

信号写法:   donchian:60,20 | donchian:20,10,stop=3 | tsmom:252 | ma:50,200
过滤器写法: none | garch | vix:low,80 | vix:high,50 | vix:low,80,source=us
命令行参数优先于配置文件. 报告: quant/reports/turtle/<name>_report.md
"""

import argparse
import sys
import time
import tomllib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_here = Path(__file__).resolve().parent
_root = next(p for p in _here.parents if (p / "data" / "paths.py").exists())   # quant 根目录
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from backtest import report, robustness                      # noqa: E402
from backtest.filters import parse_filter                    # noqa: E402
from backtest.portfolio import Runner                        # noqa: E402
from backtest.signals import parse_signal                    # noqa: E402
from data.paths import figures, reports                      # noqa: E402


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists() and (_here / path).exists():
        p = _here / path
    p = p.resolve()
    with open(p, "rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = p.relative_to(_root).as_posix() if p.is_relative_to(_root) else str(p)
    cfg.setdefault("name", p.stem)
    return cfg


def apply_overrides(cfg: dict, a) -> dict:
    st, data, pf, rb = (cfg.setdefault(k, {}) for k in ("strategy", "data", "portfolio", "robustness"))
    if a.signal:
        st["signal"] = a.signal
    if a.filter:
        st["filter"] = a.filter
    if a.start:
        data["start"] = a.start
    if a.timeframes:
        data["timeframes"] = [t.strip() for t in a.timeframes.split(",")]
    if a.refresh:
        data["refresh"] = True
    if a.target_vol is not None:
        pf["target_vol"] = a.target_vol
    if a.weighting:
        pf["weighting"] = a.weighting
    if a.symbols:
        cfg["universe"] = {"自选": [s.strip() for s in a.symbols.split(",")]}
    if a.robustness:
        rb["enabled"] = True
    if a.no_robustness:
        rb["enabled"] = False
    if a.name:
        cfg["name"] = a.name
    elif a.signal or a.filter:                     # 临时换了信号/过滤器, 报告另存一份, 不覆盖原来的
        tag = "_".join(x.replace(":", "-").replace(",", "-").replace("=", "")
                       for x in (a.signal, a.filter) if x)
        cfg["name"] = f"{cfg['name']}__{tag}"
    return cfg


def main():
    ap = argparse.ArgumentParser(description="统一回测入口 (配置文件 + 命令行覆盖)")
    ap.add_argument("config", help="配置文件, 如 configs/cross_asset.toml")
    ap.add_argument("--signal", help="信号, 如 donchian:60,20 / tsmom:252 / ma:50,200")
    ap.add_argument("--filter", help="过滤器, 如 none / garch / vix:low,80")
    ap.add_argument("--start", help="起始日期")
    ap.add_argument("--timeframes", help="周期, 如 4h,1d")
    ap.add_argument("--symbols", help="临时品种池, 如 future:RB,future:CU")
    ap.add_argument("--target-vol", type=float, help="组合目标波动, 0 = 不缩放")
    ap.add_argument("--weighting", choices=["class", "instrument"])
    ap.add_argument("--robustness", action="store_true", help="打开稳定性检验")
    ap.add_argument("--no-robustness", action="store_true", help="关掉稳定性检验")
    ap.add_argument("--name", help="报告文件名 (默认用配置里的 name)")
    ap.add_argument("--refresh", action="store_true", help="重新下载数据")
    a = ap.parse_args()

    cfg = apply_overrides(load_config(a.config), a)
    signal = parse_signal(cfg["strategy"].get("signal", "donchian:60,20"))
    filt = parse_filter(cfg["strategy"].get("filter", "none"))
    project = cfg.get("project", "turtle")
    t0 = time.time()

    print(f"\n{'=' * 80}\n{cfg.get('title', cfg['name'])}\n信号: {signal.label()} | 过滤: {filt.label()}"
          f"\n{'=' * 80}")

    print("\n── 第 1 层: 数据 ──")
    runner = Runner(cfg).load()

    print(f"\n── 第 2~4 层: 信号 → 过滤 → 执行 → 组合 ──")
    pf = runner.run(signal, filt)
    if len(pf.returns) < 2:
        print("\n[!] 没有成交, 无法组合")
        sys.exit(1)
    s = robustness.summary(pf.returns, pf.dpy)
    print(f"\n组合: 年化 {s['年化收益%']:+.1f}%  波动 {s['年化波动%']:.1f}%  夏普 {s['夏普']:.2f}"
          f"  最大回撤 {s['最大回撤%']:.1f}%")
    if pf.scaled is not None:
        s2 = robustness.summary(pf.scaled, pf.dpy)
        print(f"缩放到 {cfg['portfolio']['target_vol']:.0%} 波动: 年化 {s2['年化收益%']:+.1f}%"
              f"  夏普 {s2['夏普']:.2f}  最大回撤 {s2['最大回撤%']:.1f}%  平均杠杆 {pf.leverage.mean():.1f}")

    rob = None
    if cfg.get("robustness", {}).get("enabled"):
        print("\n── 第 5 层: 稳定性检验 ──")
        rob = robustness.run_all(runner, pf, signal, filt, cfg["robustness"])

    print("\n── 第 6 层: 报告 ──")
    bench = runner.benchmark(pf.main.index)
    path = report.write(cfg, runner, pf, bench, str(reports(project)), str(figures(project)), rob)
    print(f"报告已保存: {path}\n用时 {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
