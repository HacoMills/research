"""
项目路径统一定义
================
所有脚本的数据、缓存、图片、报告位置都从这里取.
以后调整文件夹结构, 只需要改这一个文件.

quant/
├── data/
│   ├── raw/        手动下载的原始数据 (CSMAR Excel、BTC csv 等), 重要, 不要删
│   └── cache/      脚本自动下载/生成的缓存, 可以随时删, 删了会重新下载
│       ├── okx/          币圈 K 线、资金费率、持仓量
│       ├── cn_futures/   国内期货 (新浪主力连续、单月合约)
│       ├── cn_stock/     A股 (AKShare)
│       ├── fx/           汇率 (AKShare, 东方财富)
│       ├── index/        VIX (FRED/CBOE)、中国 QVIX (AKShare)
│       └── csmar/        CSMAR 日度 Excel 合并后的缓存
├── strategies/<项目>/   代码
├── figures/<项目>/      图片
└── reports/<项目>/      报告
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent          # quant/data
PROJECT_ROOT = DATA_DIR.parent                      # quant

RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"

OKX_CACHE = CACHE_DIR / "okx"
CN_FUTURES_CACHE = CACHE_DIR / "cn_futures"
CN_STOCK_CACHE = CACHE_DIR / "cn_stock"
CSMAR_CACHE = CACHE_DIR / "csmar"
FX_CACHE = CACHE_DIR / "fx"
INDEX_CACHE = CACHE_DIR / "index"        # VIX、QVIX 等指数

FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


def raw(*parts) -> Path:
    """
    原始数据路径, 如 raw("securities") → quant/data/raw/securities.
    兼容旧位置: 如果文件还没挪进 raw/, 但在 data/ 下能找到, 就用旧位置.
    """
    new = RAW_DIR.joinpath(*parts)
    old = DATA_DIR.joinpath(*parts)
    if not new.exists() and old.exists():
        return old
    return new


def figures(project: str) -> Path:
    """某个项目的图片文件夹, 不存在就创建."""
    p = FIGURES_DIR / project
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports(project: str) -> Path:
    """某个项目的报告文件夹, 不存在就创建."""
    p = REPORTS_DIR / project
    p.mkdir(parents=True, exist_ok=True)
    return p
