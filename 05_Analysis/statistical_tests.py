"""
统计分析脚本
ICMSE 2026: 算法有限理性 — 大语言模型Agent运营决策偏差识别与纠偏机制研究

所有统计检验，α=0.05
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from config import OPTIMAL_Q, OUTPUT_DIRS


def load_all_results(raw_dir=None):
    """加载所有实验结果"""
    if raw_dir is None:
        raw_dir = os.path.join(os.path.dirname(__file__), "04_Data", "raw")

    results = {}
    if not os.path.exists(raw_dir):
        print(f"Directory not found: {raw_dir}")
        return results

    for fname in os.listdir(raw_dir):
        if fname.endswith(".jsonl"):
            exp_name = fname.replace(".jsonl", "")
            data = []
            with open(os.path.join(raw_dir, fname), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            if data:
                results[exp_name] = data
    return results


def get_valid_q(results, condition=None, model=None):
    """提取有效的订货量数据"""
    q_values = []
    for r in results:
        if condition and r.get("condition") != condition:
            continue
        if model and r.get("model") != model:
            continue
        q = r.get("order_quantity")
        if q is not None and 0 <= q <= 500:
            q_values.append(q)
    return np.array(q_values)


def get_condition_results(results, condition):
    """获取特定条件的结果"""
    return [r for r in results if r.get("condition") == condition]


# ============================================================
# E1: 基准测试 - 单样本t检验
# ============================================================
def test_baseline_vs_optimal(exp1_results):
    """
    H0: mu_LLM = Q*
    H1: mu_LLM != Q*
    """
    print("\n" + "=" * 60)
    print("E1: 单样本t检验 — LLM vs 最优解 (Q*=" + str(OPTIMAL_Q) + ")")
    print("=" * 60)

    results = {}
    for model_name in set(r.get("model") for r in exp1_results if r.get("model")):
        condition = "E1_baseline"
        q_values = get_valid_q(exp1_results, condition=condition, model=model_name)

        if len(q_values) < 2:
            print(f"\n{model_name}: 数据不足 (n={len(q_values)})")
            continue

        t_stat, p_value = stats.ttest_1samp(q_values, OPTIMAL_Q)
        mean_bias = np.mean(q_values - OPTIMAL_Q)
        cohens_d = (np.mean(q_values) - OPTIMAL_Q) / np.std(q_values, ddof=1)

        print(f"\n{model_name} (n={len(q_values)}):")
        print(f"  均值: {np.mean(q_values):.1f} (最优: {OPTIMAL_Q})")
        print(f"  偏差: {mean_bias:+.1f} ({mean_bias/OPTIMAL_Q*100:+.1f}%)")
        print(f"  t = {t_stat:.3f}, p = {p_value:.4f}, d = {cohens_d:.3f}")
        print(f"  显著: {'是' if p_value < 0.05 else '否'}")

        results[model_name] = {
            "n": len(q_values),
            "mean": float(np.mean(q_values)),
            "bias": float(mean_bias),
            "bias_pct": float(mean_bias / OPTIMAL_Q * 100),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": bool(p_value < 0.05),
        }

    return results


# ============================================================
# E2: 锚定效应 - ANOVA
# ============================================================
def test_anchoring_effect(exp2_results, exp1_results=None):
    """
    单因素ANOVA: 不同锚定值之间的差异
    事后检验: Tukey HSD
    """
    print("\n" + "=" * 60)
    print("E2: 锚定效应 — 单因素ANOVA")
    print("=" * 60)

    # 如果有E1数据，合并无锚定条件
    all_results = list(exp2_results)
    if exp1_results:
        for r in exp1_results:
            r_copy = dict(r)
            r_copy["condition"] = "E2_no_anchor"
            all_results.append(r_copy)

    anchor_conditions = sorted(set(
        r["condition"] for r in all_results
        if "E2" in r.get("condition", "")
    ))

    results = {}
    for model_name in set(r.get("model") for r in all_results if r.get("model")):
        groups = {}
        for cond in anchor_conditions:
            q = get_valid_q(all_results, condition=cond, model=model_name)
            if len(q) > 0:
                groups[cond] = q

        if len(groups) < 2:
            print(f"\n{model_name}: 条件不足")
            continue

        # ANOVA
        group_data = list(groups.values())
        f_stat, p_value = stats.f_oneway(*group_data)

        print(f"\n{model_name}:")
        for cond, q in groups.items():
            print(f"  {cond}: n={len(q)}, mean={np.mean(q):.1f}, std={np.std(q):.1f}")

        print(f"  F = {f_stat:.3f}, p = {p_value:.4f}")
        print(f"  显著: {'是' if p_value < 0.05 else '否'}")

        # 效应量 eta-squared
        grand_mean = np.mean(np.concatenate(group_data))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
        ss_total = sum((x - grand_mean) ** 2 for g in group_data for x in g)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        print(f"  η² = {eta_sq:.4f}")

        # 计算锚定效应量
        no_anchor_key = [c for c in groups if "no_anchor" in c]
        if no_anchor_key:
            baseline_q = np.mean(groups[no_anchor_key[0]])
            for cond, q in groups.items():
                if "no_anchor" not in cond:
                    effect = (np.mean(q) - baseline_q) / baseline_q * 100
                    print(f"  锚定效应 ({cond}): {effect:+.1f}%")

        results[model_name] = {
            "f_stat": float(f_stat),
            "p_value": float(p_value),
            "eta_sq": float(eta_sq),
            "significant": bool(p_value < 0.05),
            "group_means": {c: float(np.mean(g)) for c, g in groups.items()},
        }

    return results


# ============================================================
# E4: 指令敏感性 - Kruskal-Wallis
# ============================================================
def test_instruction_sensitivity(exp4_results):
    """
    Kruskal-Wallis检验: 5种表述的分布差异
    """
    print("\n" + "=" * 60)
    print("E4: 指令敏感性 — Kruskal-Wallis检验")
    print("=" * 60)

    variants = sorted(set(
        r["condition"] for r in exp4_results if "E4" in r.get("condition", "")
    ))

    results = {}
    for model_name in set(r.get("model") for r in exp4_results if r.get("model")):
        groups = {}
        for var in variants:
            q = get_valid_q(exp4_results, condition=var, model=model_name)
            if len(q) > 0:
                groups[var] = q

        if len(groups) < 2:
            continue

        group_data = list(groups.values())
        h_stat, p_value = stats.kruskal(*group_data)

        print(f"\n{model_name}:")
        for var, q in groups.items():
            print(f"  {var}: n={len(q)}, mean={np.mean(q):.1f}, std={np.std(q):.1f}")

        print(f"  H = {h_stat:.3f}, p = {p_value:.4f}")
        print(f"  显著: {'是' if p_value < 0.05 else '否'}")

        # 指令敏感性指数
        all_means = [np.mean(g) for g in group_data]
        isi = (np.std(all_means) / np.mean(all_means)) * 100 if np.mean(all_means) != 0 else 0
        print(f"  ISI = {isi:.1f}%")

        results[model_name] = {
            "h_stat": float(h_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "isi": float(isi),
        }

    return results


# ============================================================
# E5: 过度自信 - 覆盖率检验
# ============================================================
def _optimal_q_for_sigma(sigma):
    """各难度条件理性最优解：Q*(σ) = round(mu + Φ^{-1}((p-c)/(p-v))·σ)
    简单σ=10→107，中等σ=30→122，困难σ=60→144"""
    from scipy.stats import norm as _norm
    from config import NEWSVENDOR_PARAMS as _p
    _cf = (_p["unit_price"] - _p["unit_cost"]) / \
          (_p["unit_price"] - _p["salvage_value"])
    return round(_p["demand_mean"] + _norm.ppf(_cf) * sigma)


def test_overconfidence(exp5_results):
    """
    覆盖率检验: 理论覆盖率90% vs 实际覆盖率
    以各难度理性最优解为参照（简单107、中等122、困难144），
    而非统一使用σ=30条件下的Q*=122。
    """
    print("\n" + "=" * 60)
    print("E5: 过度自信 — 覆盖率检验（以各难度理性最优为参照）")
    print("=" * 60)

    sigma_map = {"easy": 10, "medium": 30, "hard": 60}
    results = {}
    for model_name in set(r.get("model") for r in exp5_results if r.get("model")):
        for diff in ["easy", "medium", "hard"]:
            condition = f"E5_{diff}"
            cond_results = get_condition_results(exp5_results, condition)
            cond_results = [r for r in cond_results if r.get("model") == model_name]

            if not cond_results:
                continue

            q_star = _optimal_q_for_sigma(sigma_map[diff])

            # 检查Q*是否在置信区间内
            covered = 0
            total = 0
            for r in cond_results:
                q_low = r.get("Q_low")
                q_high = r.get("Q_high")
                if q_low is not None and q_high is not None:
                    total += 1
                    if q_low <= q_star <= q_high:
                        covered += 1

            if total > 0:
                coverage = covered / total
                ce = abs(coverage - 0.90)
                # 二项检验
                p_value = stats.binomtest(covered, total, p=0.90).pvalue

                key = f"{model_name}_{diff}"
                print(f"\n{key} (n={total}, Q*={q_star}):")
                print(f"  覆盖率: {coverage:.1%} (目标: 90%)")
                print(f"  校准误差: {ce:.3f}")
                print(f"  二项检验 p = {p_value:.4f}")
                print(f"  过度自信: {'是' if coverage < 0.85 else '否'}")

                results[key] = {
                    "q_star": int(q_star),
                    "coverage": float(coverage),
                    "calibration_error": float(ce),
                    "p_value": float(p_value),
                    "overconfident": bool(coverage < 0.85),
                }

    return results


# ============================================================
# E6/E7/E8: 纠偏效果 - 配对t检验
# ============================================================
def test_debiasing_effect(exp_results, baseline_results, exp_name):
    """
    配对t检验: 纠偏前 vs 纠偏后

    H0: 纠偏无效 (|bias_after| = |bias_before|)
    H1: 纠偏有效 (|bias_after| < |bias_before|)
    """
    print(f"\n{'=' * 60}")
    print(f"{exp_name}: 纠偏效果 — 配对t检验")
    print(f"{'=' * 60}")

    results = {}
    for model_name in set(r.get("model") for r in baseline_results if r.get("model")):
        baseline_q = get_valid_q(baseline_results, model=model_name)
        bias_before = np.abs(baseline_q - OPTIMAL_Q)

        corrected_q = get_valid_q(exp_results, model=model_name)
        if len(corrected_q) == 0:
            continue
        bias_after = np.abs(corrected_q - OPTIMAL_Q)

        # 对齐长度
        min_len = min(len(bias_before), len(bias_after))
        if min_len < 2:
            continue

        bias_before = bias_before[:min_len]
        bias_after = bias_after[:min_len]

        # 配对t检验
        t_stat, p_value = stats.ttest_rel(bias_before, bias_after)

        # Cohen's d
        diff = bias_before - bias_after
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

        # 改善率
        improvement = (np.mean(bias_before) - np.mean(bias_after)) / np.mean(bias_before) * 100 if np.mean(bias_before) > 0 else 0

        print(f"\n{model_name}:")
        print(f"  纠偏前偏差: {np.mean(bias_before):.1f}")
        print(f"  纠偏后偏差: {np.mean(bias_after):.1f}")
        print(f"  改善率: {improvement:.1f}%")
        print(f"  t = {t_stat:.3f}, p = {p_value:.4f}, d = {cohens_d:.3f}")
        print(f"  显著: {'是' if p_value < 0.05 else '否'}")

        results[model_name] = {
            "bias_before": float(np.mean(bias_before)),
            "bias_after": float(np.mean(bias_after)),
            "improvement_pct": float(improvement),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": bool(p_value < 0.05),
        }

    return results


# ============================================================
# 综合报告
# ============================================================
def generate_report(all_results: dict):
    """生成综合统计报告"""
    print("\n" + "=" * 60)
    print("综合统计报告")
    print("=" * 60)

    report = {
        "experiments": {},
        "overall": {},
    }

    # 遍历所有实验
    for exp_name, results in all_results.items():
        if not results:
            continue

        valid = [r for r in results if r.get("order_quantity") is not None]
        if not valid:
            report["experiments"][exp_name] = {"n": 0, "error": "No valid results"}
            continue

        q_values = np.array([r["order_quantity"] for r in valid])
        models = set(r.get("model", "unknown") for r in valid)

        exp_stats = {
            "n_total": len(results),
            "n_valid": len(valid),
            "parse_rate": len(valid) / len(results) if results else 0,
            "models": list(models),
            "conditions": sorted(set(r.get("condition", "") for r in valid)),
            "overall_mean": float(np.mean(q_values)),
            "overall_std": float(np.std(q_values)),
            "mean_bias": float(np.mean(q_values - OPTIMAL_Q)),
            "mean_abs_bias": float(np.mean(np.abs(q_values - OPTIMAL_Q))),
            "mean_rel_bias": float(np.mean(np.abs(q_values - OPTIMAL_Q) / OPTIMAL_Q * 100)),
        }

        print(f"\n{exp_name}:")
        print(f"  记录数: {exp_stats['n_total']} (有效: {exp_stats['n_valid']}, {exp_stats['parse_rate']:.1%})")
        print(f"  模型: {exp_stats['models']}")
        print(f"  条件: {exp_stats['conditions']}")
        print(f"  均值: {exp_stats['overall_mean']:.1f} (最优: {OPTIMAL_Q})")
        print(f"  平均绝对偏差: {exp_stats['mean_abs_bias']:.1f} ({exp_stats['mean_rel_bias']:.1f}%)")

        report["experiments"][exp_name] = exp_stats

    # 总体统计
    total_records = sum(e["n_total"] for e in report["experiments"].values())
    total_valid = sum(e["n_valid"] for e in report["experiments"].values())
    report["overall"] = {
        "total_records": total_records,
        "total_valid": total_valid,
        "parse_rate": total_valid / total_records if total_records > 0 else 0,
    }

    print(f"\n总计: {total_records} 条记录, {total_valid} 条有效 ({report['overall']['parse_rate']:.1%})")

    return report


# ============================================================
# 主函数
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="运行所有统计检验")
    parser.add_argument("--report", action="store_true", help="生成综合报告")
    args = parser.parse_args()

    all_data = load_all_results()

    if not all_data:
        print("No data found. Run experiments first.")
        return

    print(f"Loaded {len(all_data)} experiment datasets")

    # 综合报告
    if args.report or args.all:
        generate_report(all_data)

    if not args.all:
        return

    # 各实验统计检验
    if "exp1_baseline" in all_data:
        test_baseline_vs_optimal(all_data["exp1_baseline"])

    if "exp2_anchoring" in all_data:
        exp1 = all_data.get("exp1_baseline", None)
        test_anchoring_effect(all_data["exp2_anchoring"], exp1)

    if "exp4_instruction" in all_data:
        test_instruction_sensitivity(all_data["exp4_instruction"])

    if "exp5_overconfidence" in all_data:
        test_overconfidence(all_data["exp5_overconfidence"])

    # 纠偏效果
    if "exp6_cot" in all_data and "exp2_anchoring" in all_data:
        test_debiasing_effect(all_data["exp6_cot"], all_data["exp2_anchoring"], "E6: CoT")

    if "exp7_debate" in all_data and "exp2_anchoring" in all_data:
        test_debiasing_effect(all_data["exp7_debate"], all_data["exp2_anchoring"], "E7: Debate")

    if "exp8_human_calibration" in all_data and "exp2_anchoring" in all_data:
        test_debiasing_effect(all_data["exp8_human_calibration"], all_data["exp2_anchoring"], "E8: Calibration")


if __name__ == "__main__":
    main()