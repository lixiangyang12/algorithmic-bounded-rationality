"""
实验运行脚本 - 所有8个实验的统一入口

ICMSE 2026: 算法有限理性 — 大语言模型Agent运营决策偏差识别与纠偏机制研究

用法:
    python run_all.py                    # 运行所有实验
    python run_all.py --exp E1           # 只运行实验1
    python run_all.py --exp E1 E2 E3     # 运行指定实验
    python run_all.py --models gpt-4o    # 只使用指定模型
    python run_all.py --n 10             # 减少重复次数（快速测试）
    python run_all.py --dry-run          # 干运行（只打印配置，不调用API）
"""
import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Optional

# 添加项目根目录和子模块到Python路径
_project_root = os.path.dirname(os.path.abspath(__file__))
for _p in [_project_root, os.path.join(_project_root, "03_Experiments")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (
    LLM_CONFIGS, EXPERIMENT_PARAMS, NEWSVENDOR_PARAMS,
    OPTIMAL_Q, OUTPUT_DIRS
)
from agents.llm_agent import LLMAgent, create_agents, save_results, load_results
from agents.response_parser import parse_response, validate_response, parse_success_rate
from agents.prompt_templates import (
    newsvendor_base_prompt,
    newsvendor_anchored_prompt,
    newsvendor_hallucination_prompt,
    newsvendor_instruction_variants,
    newsvendor_confidence_prompt,
    cot_prompt,
    debate_prompt_agent,
    debate_synthesis_prompt,
    human_calibration_prompt,
)
from tasks.newsvendor import (
    compute_bias, compute_anchoring_effect,
    compute_calibration_error, compute_instruction_sensitivity,
    compute_debiasing_effect, compute_summary_stats,
)

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(_project_root, "04_Data", f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("run_all")

# ============================================================
# 实验基类
# ============================================================
class ExperimentRunner:
    """实验运行器基类"""

    def __init__(self, agents: dict, n_repetitions: int = 30, dry_run: bool = False):
        self.agents = agents
        self.n_repetitions = n_repetitions
        self.dry_run = dry_run
        self.nv_params = NEWSVENDOR_PARAMS

    def _get_nv_args(self):
        """获取报童参数"""
        return (
            self.nv_params["unit_cost"],
            self.nv_params["unit_price"],
            self.nv_params["salvage_value"],
            self.nv_params["demand_mean"],
            self.nv_params["demand_std"],
        )

    def _run_condition(self, model_name: str, prompts: list, condition_name: str) -> list:
        """
        运行一个实验条件

        Args:
            model_name: 模型名称
            prompts: Prompt列表（每个prompt对应一次重复）
            condition_name: 条件名称（用于日志）

        Returns:
            list of dict: 解析后的结果
        """
        agent = self.agents.get(model_name)
        if agent is None:
            logger.warning(f"[{condition_name}] Agent {model_name} not available, skipping")
            return []

        results = []
        for i, prompt in enumerate(prompts):
            if self.dry_run:
                logger.info(f"[DRY-RUN] {model_name}/{condition_name} #{i+1}: prompt={len(prompt)} chars")
                results.append({
                    "model": model_name,
                    "condition": condition_name,
                    "rep": i + 1,
                    "order_quantity": None,
                    "parse_success": False,
                    "dry_run": True,
                })
                continue

            logger.info(f"[{condition_name}] {model_name} #{i+1}/{len(prompts)}")
            api_result = agent.call_with_retry(prompt)

            if api_result["success"]:
                parsed = parse_response(api_result["response"])
                parsed["model"] = model_name
                parsed["condition"] = condition_name
                parsed["rep"] = i + 1
                parsed["tokens"] = api_result.get("tokens", 0)
                parsed["cost"] = api_result.get("cost", 0)
                parsed["api_success"] = True
                results.append(parsed)
            else:
                results.append({
                    "model": model_name,
                    "condition": condition_name,
                    "rep": i + 1,
                    "order_quantity": None,
                    "parse_success": False,
                    "api_success": False,
                    "error": api_result.get("error", "Unknown"),
                })

            # Rate limit
            if i < len(prompts) - 1:
                time.sleep(EXPERIMENT_PARAMS["api_call_interval"])

        return results

    def _save_and_summarize(self, results: list, filename: str):
        """保存结果并输出汇总统计"""
        save_results(results, os.path.join(OUTPUT_DIRS["raw"], filename))

        valid_results = [r for r in results if r.get("order_quantity") is not None]
        if valid_results:
            stats = compute_summary_stats(valid_results)
            logger.info(f"[{filename}] N={stats['n']}, Mean={stats['mean']:.1f}, "
                        f"Bias={stats['mean_relative_bias']:.1f}%, "
                        f"Parse={stats['parse_success_rate']:.1%}")
        return results


# ============================================================
# E1: 基准测试
# ============================================================
def run_exp1_baseline(agents, n_repetitions=30, dry_run=False):
    """实验1：报童问题基准测试"""
    logger.info("=" * 60)
    logger.info("E1: 报童问题基准测试")
    logger.info("=" * 60)

    runner = ExperimentRunner(agents, n_repetitions, dry_run)
    args = runner._get_nv_args()
    all_results = []

    for model_name in agents:
        prompts = [newsvendor_base_prompt(*args, lang="zh") for _ in range(n_repetitions)]
        results = runner._run_condition(model_name, prompts, "E1_baseline")
        all_results.extend(results)

    runner._save_and_summarize(all_results, "exp1_baseline.jsonl")
    return all_results


# ============================================================
# E2: 锚定效应
# ============================================================
def run_exp2_anchoring(agents, n_repetitions=30, dry_run=False):
    """实验2：锚定效应"""
    logger.info("=" * 60)
    logger.info("E2: 锚定效应实验")
    logger.info("=" * 60)

    runner = ExperimentRunner(agents, n_repetitions, dry_run)
    args = runner._get_nv_args()
    all_results = []

    anchor_conditions = [
        ("low", 50),
        ("high", 200),
        ("random", 73),
    ]

    for model_name in agents:
        for anchor_type, anchor_value in anchor_conditions:
            condition_name = f"E2_anchor_{anchor_type}"
            prompts = [
                newsvendor_anchored_prompt(*args, anchor_value=anchor_value, anchor_type=anchor_type, lang="zh")
                for _ in range(n_repetitions)
            ]
            results = runner._run_condition(model_name, prompts, condition_name)
            all_results.extend(results)

    runner._save_and_summarize(all_results, "exp2_anchoring.jsonl")
    return all_results


# ============================================================
# E3: 幻觉偏差
# ============================================================
def run_exp3_hallucination(agents, n_repetitions=30, dry_run=False):
    """实验3：幻觉偏差"""
    logger.info("=" * 60)
    logger.info("E3: 幻觉偏差实验")
    logger.info("=" * 60)

    runner = ExperimentRunner(agents, n_repetitions, dry_run)
    args = runner._get_nv_args()
    all_results = []

    for model_name in agents:
        for fake_id in [1, 2, 3]:
            condition_name = f"E3_fake_{fake_id}"
            prompts = [
                newsvendor_hallucination_prompt(*args, fake_info_id=fake_id, lang="zh")
                for _ in range(n_repetitions)
            ]
            results = runner._run_condition(model_name, prompts, condition_name)
            all_results.extend(results)

    runner._save_and_summarize(all_results, "exp3_hallucination.jsonl")
    return all_results


# ============================================================
# E4: 指令敏感性
# ============================================================
def run_exp4_instruction(agents, n_repetitions=20, dry_run=False):
    """实验4：指令敏感性"""
    logger.info("=" * 60)
    logger.info("E4: 指令敏感性实验")
    logger.info("=" * 60)

    runner = ExperimentRunner(agents, n_repetitions, dry_run)
    args = runner._get_nv_args()
    all_results = []

    for model_name in agents:
        for variant_id in [1, 2, 3, 4, 5]:
            condition_name = f"E4_variant_{variant_id}"
            prompts = [
                newsvendor_instruction_variants(*args, variant_id=variant_id, lang="zh")
                for _ in range(n_repetitions)
            ]
            results = runner._run_condition(model_name, prompts, condition_name)
            all_results.extend(results)

    runner._save_and_summarize(all_results, "exp4_instruction.jsonl")
    return all_results


# ============================================================
# E5: 过度自信
# ============================================================
def run_exp5_overconfidence(agents, n_repetitions=30, dry_run=False):
    """实验5：过度自信"""
    logger.info("=" * 60)
    logger.info("E5: 过度自信实验")
    logger.info("=" * 60)

    runner = ExperimentRunner(agents, n_repetitions, dry_run)
    all_results = []

    # 三种难度级别
    difficulty_levels = [
        ("easy", 10),
        ("medium", 30),
        ("hard", 60),
    ]

    for model_name in agents:
        for diff_name, sigma in difficulty_levels:
            condition_name = f"E5_{diff_name}"
            cost, price, salvage, mu, _ = runner._get_nv_args()
            prompts = [
                newsvendor_confidence_prompt(cost, price, salvage, mu, sigma, lang="zh")
                for _ in range(n_repetitions)
            ]
            results = runner._run_condition(model_name, prompts, condition_name)
            all_results.extend(results)

    runner._save_and_summarize(all_results, "exp5_overconfidence.jsonl")
    return all_results


# ============================================================
# E6: CoT纠偏
# ============================================================
def run_exp6_cot(agents, n_repetitions=30, dry_run=False):
    """实验6：Chain-of-Thought纠偏"""
    logger.info("=" * 60)
    logger.info("E6: CoT纠偏实验")
    logger.info("=" * 60)

    runner = ExperimentRunner(agents, n_repetitions, dry_run)
    args = runner._get_nv_args()
    all_results = []

    # 使用高锚定条件作为偏差条件（预期偏差最显著）
    for model_name in agents:
        # CoT条件
        prompts = [
            cot_prompt(newsvendor_anchored_prompt, *args, anchor_value=200, anchor_type="high", lang="zh")
            for _ in range(n_repetitions)
        ]
        results = runner._run_condition(model_name, prompts, "E6_cot")
        all_results.extend(results)

    runner._save_and_summarize(all_results, "exp6_cot.jsonl")
    return all_results


# ============================================================
# E7: 多Agent辩论
# ============================================================
def run_exp7_debate(agents, n_repetitions=15, dry_run=False):
    """实验7：多Agent辩论纠偏"""
    logger.info("=" * 60)
    logger.info("E7: 多Agent辩论实验")
    logger.info("=" * 60)

    runner = ExperimentRunner(agents, n_repetitions, dry_run)
    args = runner._get_nv_args()
    all_results = []

    roles = ["conservative", "aggressive", "analytical"]

    for model_name in agents:
        agent = agents.get(model_name)
        if agent is None:
            continue

        for rep in range(n_repetitions):
            logger.info(f"[E7] {model_name} Debate #{rep+1}/{n_repetitions}")

            if dry_run:
                all_results.append({
                    "model": model_name, "condition": "E7_debate", "rep": rep + 1,
                    "order_quantity": None, "parse_success": False, "dry_run": True,
                })
                continue

            # Round 1: 三个Agent独立决策
            round1_answers = []
            for role in roles:
                prompt = debate_prompt_agent(
                    newsvendor_anchored_prompt, role,
                    *args, anchor_value=200, anchor_type="high", lang="zh"
                )
                api_result = agent.call_with_retry(prompt)
                if api_result["success"]:
                    parsed = parse_response(api_result["response"])
                    parsed["role"] = role
                    parsed["round"] = 1
                    round1_answers.append(parsed)
                time.sleep(1.0)

            # Round 2: 综合决策
            if len(round1_answers) >= 2:
                synthesis_prompt = debate_synthesis_prompt(round1_answers, lang="zh")
                api_result = agent.call_with_retry(synthesis_prompt)
                if api_result["success"]:
                    parsed = parse_response(api_result["response"])
                    parsed["model"] = model_name
                    parsed["condition"] = "E7_debate"
                    parsed["rep"] = rep + 1
                    parsed["round"] = 2
                    parsed["round1_answers"] = [
                        {"role": a["role"], "order_quantity": a["order_quantity"]}
                        for a in round1_answers
                    ]
                    parsed["tokens"] = api_result.get("tokens", 0)
                    parsed["api_success"] = True
                    all_results.append(parsed)
            else:
                all_results.append({
                    "model": model_name, "condition": "E7_debate", "rep": rep + 1,
                    "order_quantity": None, "parse_success": False,
                    "api_success": False, "error": "Insufficient round1 answers",
                })

            time.sleep(1.0)

    runner._save_and_summarize(all_results, "exp7_debate.jsonl")
    return all_results


# ============================================================
# E8: 人类反馈校准
# ============================================================
def run_exp8_calibration(agents, n_repetitions=20, dry_run=False):
    """实验8：人类反馈校准"""
    logger.info("=" * 60)
    logger.info("E8: 人类反馈校准实验")
    logger.info("=" * 60)

    runner = ExperimentRunner(agents, n_repetitions, dry_run)
    args = runner._get_nv_args()
    all_results = []

    feedback_levels = ["light", "strong"]

    for model_name in agents:
        agent = agents.get(model_name)
        if agent is None:
            continue

        for rep in range(n_repetitions):
            logger.info(f"[E8] {model_name} Calibration #{rep+1}/{n_repetitions}")

            if dry_run:
                for fb in feedback_levels:
                    all_results.append({
                        "model": model_name, "condition": f"E8_{fb}", "rep": rep + 1,
                        "order_quantity": None, "parse_success": False, "dry_run": True,
                    })
                continue

            # Round 1: 无反馈基准
            base_prompt = newsvendor_anchored_prompt(*args, anchor_value=200, anchor_type="high", lang="zh")
            api_result = agent.call_with_retry(base_prompt)
            if not api_result["success"]:
                continue
            first_answer = parse_response(api_result["response"])
            time.sleep(1.0)

            # Round 2: 反馈校准
            for fb_level in feedback_levels:
                fb_prompt = human_calibration_prompt(
                    newsvendor_anchored_prompt, fb_level, first_answer,
                    *args, anchor_value=200, anchor_type="high", lang="zh"
                )
                api_result2 = agent.call_with_retry(fb_prompt)
                if api_result2["success"]:
                    parsed = parse_response(api_result2["response"])
                    parsed["model"] = model_name
                    parsed["condition"] = f"E8_{fb_level}"
                    parsed["rep"] = rep + 1
                    parsed["first_answer"] = first_answer.get("order_quantity")
                    parsed["tokens"] = api_result2.get("tokens", 0)
                    parsed["api_success"] = True
                    all_results.append(parsed)
                time.sleep(1.0)

    runner._save_and_summarize(all_results, "exp8_human_calibration.jsonl")
    return all_results


# ============================================================
# 主函数
# ============================================================
EXPERIMENT_MAP = {
    "E1": run_exp1_baseline,
    "E2": run_exp2_anchoring,
    "E3": run_exp3_hallucination,
    "E4": run_exp4_instruction,
    "E5": run_exp5_overconfidence,
    "E6": run_exp6_cot,
    "E7": run_exp7_debate,
    "E8": run_exp8_calibration,
}


def main():
    parser = argparse.ArgumentParser(description="ICMSE 2026 实验运行脚本")
    parser.add_argument("--exp", nargs="+", default=["E1"],
                        help="要运行的实验 (E1-E8, 或 'all')")
    parser.add_argument("--models", nargs="+", default=None,
                        help="要使用的模型 (默认: 所有已配置API key的模型)")
    parser.add_argument("--n", type=int, default=None,
                        help="重复次数 (默认: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="干运行模式（不调用API）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有实验")
    parser.add_argument("--cost", action="store_true",
                        help="显示预计API成本")

    args = parser.parse_args()

    if args.list:
        print("\n实验列表:")
        for name, func in EXPERIMENT_MAP.items():
            print(f"  {name}: {func.__doc__}")
        return

    if args.cost:
        _estimate_cost(args)
        return

    # 创建Agent
    all_agents = create_agents(LLM_CONFIGS)
    if not all_agents:
        logger.error("No agents available! Please configure API keys in .env")
        return

    # 过滤模型
    if args.models:
        agents = {m: a for m, a in all_agents.items() if m in args.models}
        if not agents:
            logger.error(f"No matching models: {args.models}")
            logger.info(f"Available: {list(all_agents.keys())}")
            return
    else:
        agents = all_agents

    logger.info(f"Models: {list(agents.keys())}")

    # 确定重复次数
    n_repetitions = args.n or EXPERIMENT_PARAMS["n_repetitions"]
    logger.info(f"Repetitions: {n_repetitions}")

    # 确定要运行的实验
    if "all" in args.exp:
        exps_to_run = list(EXPERIMENT_MAP.keys())
    else:
        exps_to_run = [e for e in args.exp if e in EXPERIMENT_MAP]

    if not exps_to_run:
        logger.error(f"No valid experiments: {args.exp}")
        return

    logger.info(f"Experiments: {exps_to_run}")
    logger.info(f"Dry run: {args.dry_run}")

    # 运行实验
    start_time = time.time()
    all_results = {}

    for exp_name in exps_to_run:
        exp_func = EXPERIMENT_MAP[exp_name]
        logger.info(f"\n{'#' * 60}")
        logger.info(f"Starting {exp_name}...")
        logger.info(f"{'#' * 60}")

        exp_start = time.time()
        results = exp_func(agents, n_repetitions, args.dry_run)
        all_results[exp_name] = results
        exp_elapsed = time.time() - exp_start

        logger.info(f"{exp_name} completed in {exp_elapsed:.0f}s ({exp_elapsed/60:.1f}min)")

    # 总统计
    total_elapsed = time.time() - start_time
    total_calls = sum(len(r) for r in all_results.values())
    logger.info(f"\n{'=' * 60}")
    logger.info(f"All experiments completed!")
    logger.info(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    logger.info(f"Total results: {total_calls}")

    # Agent统计
    for name, agent in agents.items():
        stats = agent.get_stats()
        logger.info(f"[{name}] Calls: {stats['total_calls']}, "
                    f"Tokens: {stats['total_tokens']}, "
                    f"Cost: ${stats['total_cost']:.2f}")

    logger.info(f"Results saved to: {OUTPUT_DIRS['raw']}")


def _estimate_cost(args):
    """估算API成本"""
    n_repetitions = args.n or EXPERIMENT_PARAMS["n_repetitions"]

    cost_table = {
        "E1": {"calls": 4 * n_repetitions, "desc": "基准测试"},
        "E2": {"calls": 4 * n_repetitions * 3, "desc": "锚定效应"},
        "E3": {"calls": 4 * n_repetitions * 3, "desc": "幻觉偏差"},
        "E4": {"calls": 4 * n_repetitions * 5, "desc": "指令敏感性"},
        "E5": {"calls": 4 * n_repetitions * 3, "desc": "过度自信"},
        "E6": {"calls": 4 * n_repetitions, "desc": "CoT纠偏"},
        "E7": {"calls": 4 * n_repetitions * 4, "desc": "多Agent辩论"},
        "E8": {"calls": 4 * n_repetitions * 3, "desc": "人类校准"},
    }

    print(f"\n预计API成本 (n={n_repetitions}):")
    print(f"{'实验':<6} {'描述':<12} {'调用次数':<10} {'预计成本':<12}")
    print("-" * 44)

    total_calls = 0
    total_cost = 0
    for exp_name, info in cost_table.items():
        calls = info["calls"]
        cost = calls * 0.012  # 平均每次$0.012
        total_calls += calls
        total_cost += cost
        print(f"{exp_name:<6} {info['desc']:<12} {calls:<10} ¥{cost*7.2:.0f}")

    print("-" * 44)
    print(f"{'总计':<6} {'':<12} {total_calls:<10} ¥{total_cost*7.2:.0f}")
    print(f"\n注: 实际成本取决于模型、token消耗和API价格")


if __name__ == "__main__":
    main()