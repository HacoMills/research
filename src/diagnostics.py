"""
建模前的统计检验：序列平稳性（ADF）和 ARCH 效应（Engle's ARCH-LM）。

这一步是决定"该不该用 GARCH"的前提检验，而不是可有可无的装饰：
只有在收益率序列平稳、且存在显著 ARCH 效应（波动率聚集）的前提下，
用 GARCH 建模波动率才有统计学上的正当性。
"""
import pandas as pd
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller


def run_stationarity_and_arch_tests(returns: pd.Series, arch_lags: int = 15) -> dict:
    """
    对对数收益率序列做 ADF 平稳性检验和 ARCH-LM 效应检验。

    Returns:
        dict，包含两个检验的统计量、p-value 和文字结论，方便直接打印或写进报告。
    """
    adf_result = adfuller(returns)
    is_stationary = adf_result[1] < 0.05

    resid = returns - returns.mean()
    arch_result = het_arch(resid, nlags=arch_lags)
    has_arch_effect = arch_result[1] < 0.05

    return {
        "adf_statistic": adf_result[0],
        "adf_pvalue": adf_result[1],
        "is_stationary": is_stationary,
        "arch_lm_statistic": arch_result[0],
        "arch_lm_pvalue": arch_result[1],
        "has_arch_effect": has_arch_effect,
        "summary": (
            f"ADF: 序列{'平稳' if is_stationary else '不平稳'} (p={adf_result[1]:.4e})；"
            f"ARCH-LM: {'存在' if has_arch_effect else '不存在'}显著波动率聚集效应 "
            f"(p={arch_result[1]:.4e})"
            + ("，适合用 GARCH 建模。" if (is_stationary and has_arch_effect) else "。")
        ),
    }
