"""
报童问题任务定义
提供最优解计算、利润计算等工具函数
"""
import numpy as np
from scipy.stats import norm
from config import NEWSVENDOR_PARAMS, OPTIMAL_Q


def compute_optimal_q(cost=None, price=None, salvage=None, mu=None, sigma=None):
    """
    计算报童问题的最优订货量

    Q* = mu + Phi^{-1}((p-c)/(p-v)) * sigma

    Returns:
        float: 最优订货量
    """
    c = cost or NEWSVENDOR_PARAMS["unit_cost"]
    p = price or NEWSVENDOR_PARAMS["unit_price"]
    v = salvage or NEWSVENDOR_PARAMS["salvage_value"]
    m = mu or NEWSVENDOR_PARAMS["demand_mean"]
    s = sigma or NEWSVENDOR_PARAMS["demand_std"]

    critical_fractile = (p - c) / (p - v)
    z = norm.ppf(critical_fractile)
    q_star = m + z * s
    return q_star


def expected_profit(Q, cost=None, price=None, salvage=None, mu=None, sigma=None):
    """
    计算期望利润

    E[pi(Q)] = p*E[min(Q,D)] + v*E[(Q-D)+] - c*Q

    Returns:
        float: 期望利润
    """
    c = cost or NEWSVENDOR_PARAMS["unit_cost"]
    p = price or NEWSVENDOR_PARAMS["unit_price"]
    v = salvage or NEWSVENDOR_PARAMS["salvage_value"]
    m = mu or NEWSVENDOR_PARAMS["demand_mean"]
    s = sigma or NEWSVENDOR_PARAMS["demand_std"]

    z = (Q - m) / s
    # 期望销售量
    expected_sales = m * norm.cdf(z) - s * norm.pdf(z) + Q * (1 - norm.cdf(z))
    # 期望残值
    expected_salvage = (Q - m) * norm.cdf(z) + s * norm.pdf(z)
    if expected_salvage < 0:
        expected_salvage = max(0, Q - expected_sales)

    profit = p * expected_sales + v * expected_salvage - c * Q
    return profit


def compute_bias(q_llm, q_optimal=None):
    """
    计算LLM决策偏差

    Args:
        q_llm: LLM的订货量
        q_optimal: 最优订货量（默认使用OPTIMAL_Q）

    Returns:
        dict: {"absolute": float, "relative": float, "direction": str}
    """
    q_star = q_optimal if q_optimal is not None else OPTIMAL_Q
    absolute_bias = q_llm - q_star
    relative_bias = (absolute_bias / q_star) * 100 if q_star != 0 else 0
    direction = "high" if absolute_bias > 0 else ("low" if absolute_bias < 0 else "optimal")

    return {
        "absolute": absolute_bias,
        "relative": relative_bias,
        "direction": direction,
        "q_llm": q_llm,
        "q_optimal": q_star,
    }


def compute_anchoring_effect(q_anchored, q_baseline):
    """
    计算锚定效应量

    Anchoring Effect = (Q_anchored - Q_baseline) / Q_baseline * 100%

    Returns:
        float: 锚定效应百分比
    """
    if q_baseline == 0:
        return 0.0
    return ((q_anchored - q_baseline) / q_baseline) * 100


def compute_calibration_error(coverage_empirical, coverage_nominal=0.90):
    """
    计算校准误差

    CE = |Coverage_empirical - Coverage_nominal|

    Returns:
        float: 校准误差
    """
    return abs(coverage_empirical - coverage_nominal)


def compute_ece(confidence_bins, accuracy_bins):
    """
    计算期望校准误差 (Expected Calibration Error)

    ECE = sum(n_j/N * |acc_j - conf_j|)

    Args:
        confidence_bins: list of (conf_mean, count) for each bin
        accuracy_bins: list of accuracy for each bin

    Returns:
        float: ECE
    """
    total_n = sum(c for _, c in confidence_bins)
    if total_n == 0:
        return 0.0

    ece = 0.0
    for (conf, n), acc in zip(confidence_bins, accuracy_bins):
        if n > 0:
            ece += (n / total_n) * abs(acc - conf)
    return ece


def compute_debiasing_effect(bias_before, bias_after):
    """
    计算纠偏效果

    Debiasing Effect = (|bias_before| - |bias_after|) / |bias_before| * 100%

    Returns:
        float: 纠偏效果百分比
    """
    if abs(bias_before) < 1e-10:
        return 0.0
    return ((abs(bias_before) - abs(bias_after)) / abs(bias_before)) * 100


def compute_instruction_sensitivity(q_values):
    """
    计算指令敏感性指数

    ISI = sigma_Q / mean_Q * 100%

    Args:
        q_values: 不同表述下的订货量列表

    Returns:
        float: ISI百分比
    """
    if not q_values or len(q_values) < 2:
        return 0.0
    arr = np.array(q_values)
    if np.mean(arr) == 0:
        return 0.0
    return (np.std(arr) / np.mean(arr)) * 100


# ============================================================
# 批量计算工具
# ============================================================
def compute_summary_stats(results: list, q_optimal=None):
    """
    计算实验结果汇总统计

    Args:
        results: 解析后的结果列表，每个元素包含order_quantity等字段
        q_optimal: 最优解

    Returns:
        dict: 汇总统计
    """
    q_star = q_optimal if q_optimal is not None else OPTIMAL_Q
    q_values = [r.get("order_quantity") for r in results if r.get("order_quantity") is not None]

    if not q_values:
        return {"n": 0, "error": "No valid results"}

    arr = np.array(q_values)
    biases = [compute_bias(q, q_star) for q in q_values]

    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "mean_absolute_bias": float(np.mean([b["absolute"] for b in biases])),
        "mean_relative_bias": float(np.mean([b["relative"] for b in biases])),
        "rmse": float(np.sqrt(np.mean((arr - q_star) ** 2))),
        "q_optimal": q_star,
        "parse_success_rate": sum(1 for r in results if r.get("parse_success")) / len(results) if results else 0,
    }