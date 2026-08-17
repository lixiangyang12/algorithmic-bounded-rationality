"""
Prompt模板库
ICMSE 2026: 算法有限理性 — 大语言模型Agent运营决策偏差识别与纠偏机制研究

包含所有实验的Prompt模板（中英文双语版本）
"""
from config import NEWSVENDOR_PARAMS as NV


# ============================================================
# 辅助函数
# ============================================================
def _fmt_params(cost=None, price=None, salvage=None, mu=None, sigma=None):
    """格式化报童参数"""
    return {
        "cost": cost or NV["unit_cost"],
        "price": price or NV["unit_price"],
        "salvage": salvage or NV["salvage_value"],
        "mu": mu or NV["demand_mean"],
        "sigma": sigma or NV["demand_std"],
    }


# ============================================================
# E1: 基准报童问题 Prompt
# ============================================================
def newsvendor_base_prompt(cost=5, price=15, salvage=2, demand_mean=100, demand_std=30, lang="zh"):
    """基础报童问题 - 无偏差条件"""
    if lang == "zh":
        return f"""你是一位运营管理专家。请解决以下库存决策问题：

【问题描述】
一家零售商需要决定某季节性商品的订货量。
- 单位采购成本：{cost}元
- 单位销售价格：{price}元
- 季末未售出商品的残值：{salvage}元/件
- 市场需求服从正态分布，均值为{demand_mean}件，标准差为{demand_std}件

【决策任务】
请确定最优订货量Q，使得期望利润最大化。

【输出要求】
请以JSON格式回答：
{{"order_quantity": <整数>, "reasoning": "<你的推理过程>", "confidence": <0-1之间的置信度>}}"""
    else:
        return f"""You are an operations management expert. Please solve the following inventory decision problem:

[Problem Description]
A retailer needs to decide the order quantity for a seasonal product.
- Unit purchase cost: ${cost}
- Unit selling price: ${price}
- Salvage value for unsold units: ${salvage}/unit
- Demand follows a normal distribution with mean {demand_mean} units and standard deviation {demand_std} units

[Decision Task]
Determine the optimal order quantity Q to maximize expected profit.

[Output Format]
Respond in JSON format:
{{"order_quantity": <integer>, "reasoning": "<your reasoning process>", "confidence": <0-1>}}"""


# ============================================================
# E2: 锚定效应 Prompt
# ============================================================
def newsvendor_anchored_prompt(cost=5, price=15, salvage=2, demand_mean=100, demand_std=30,
                                anchor_value=200, anchor_type="high", lang="zh"):
    """带锚定的报童问题"""
    anchor_texts = {
        "zh": {
            "high": f"【背景信息】据行业分析师预测，该类商品今年市场需求可能高达{anchor_value}件。",
            "low": f"【背景信息】据初步调研，该类商品今年市场需求可能仅为{anchor_value}件。",
            "random": f"【背景信息】该公司去年的员工满意度调查得分为{anchor_value}分（满分1000分）。",
        },
        "en": {
            "high": f"[Background] According to industry analysts, the market demand for this product could reach as high as {anchor_value} units this year.",
            "low": f"[Background] According to preliminary research, the market demand for this product may be only {anchor_value} units.",
            "random": f"[Background] The company's employee satisfaction survey score last year was {anchor_value} (out of 1000).",
        }
    }
    anchor_text = anchor_texts[lang][anchor_type]

    if lang == "zh":
        return f"""你是一位运营管理专家。请解决以下库存决策问题：

{anchor_text}

【问题描述】
一家零售商需要决定某季节性商品的订货量。
- 单位采购成本：{cost}元
- 单位销售价格：{price}元
- 季末未售出商品的残值：{salvage}元/件
- 市场需求服从正态分布，均值为{demand_mean}件，标准差为{demand_std}件

【决策任务】
请确定最优订货量Q，使得期望利润最大化。

【输出要求】
请以JSON格式回答：
{{"order_quantity": <整数>, "reasoning": "<你的推理过程>", "confidence": <0-1之间的置信度>}}"""
    else:
        return f"""You are an operations management expert. Please solve the following inventory decision problem:

{anchor_text}

[Problem Description]
A retailer needs to decide the order quantity for a seasonal product.
- Unit purchase cost: ${cost}
- Unit selling price: ${price}
- Salvage value for unsold units: ${salvage}/unit
- Demand follows a normal distribution with mean {demand_mean} units and standard deviation {demand_std} units

[Decision Task]
Determine the optimal order quantity Q to maximize expected profit.

[Output Format]
Respond in JSON format:
{{"order_quantity": <integer>, "reasoning": "<your reasoning process>", "confidence": <0-1>}}"""


# ============================================================
# E3: 幻觉偏差 Prompt
# ============================================================
def newsvendor_hallucination_prompt(cost=5, price=15, salvage=2, demand_mean=100, demand_std=30,
                                     fake_info_id=1, lang="zh"):
    """嵌入虚假信息的报童问题"""
    fake_info_texts = {
        "zh": {
            1: "【重要市场信息】据2025年供应链报告，该商品原材料即将涨价50%，采购成本可能大幅上升。",
            2: "【重要政策信息】该商品所在行业已被列入限制类目录，明年将禁止销售。",
            3: "【重要竞争信息】主要竞争对手已宣布退出该市场，市场份额将大幅增加。",
        },
        "en": {
            1: "[Important Market Information] According to the 2025 supply chain report, the raw material price for this product is about to increase by 50%, and procurement costs may rise significantly.",
            2: "[Important Policy Information] This product's industry has been listed in the restricted category and will be banned from sale next year.",
            3: "[Important Competitive Information] Major competitors have announced their exit from this market, and market share will increase significantly.",
        }
    }
    fake_text = fake_info_texts[lang][fake_info_id]

    if lang == "zh":
        return f"""你是一位运营管理专家。请解决以下库存决策问题：

{fake_text}

【问题描述】
一家零售商需要决定某季节性商品的订货量。
- 单位采购成本：{cost}元
- 单位销售价格：{price}元
- 季末未售出商品的残值：{salvage}元/件
- 市场需求服从正态分布，均值为{demand_mean}件，标准差为{demand_std}件

【决策任务】
请确定最优订货量Q，使得期望利润最大化。

【输出要求】
请以JSON格式回答：
{{"order_quantity": <整数>, "reasoning": "<你的推理过程>", "confidence": <0-1之间的置信度>}}"""
    else:
        return f"""You are an operations management expert. Please solve the following inventory decision problem:

{fake_text}

[Problem Description]
A retailer needs to decide the order quantity for a seasonal product.
- Unit purchase cost: ${cost}
- Unit selling price: ${price}
- Salvage value for unsold units: ${salvage}/unit
- Demand follows a normal distribution with mean {demand_mean} units and standard deviation {demand_std} units

[Decision Task]
Determine the optimal order quantity Q to maximize expected profit.

[Output Format]
Respond in JSON format:
{{"order_quantity": <integer>, "reasoning": "<your reasoning process>", "confidence": <0-1>}}"""


# ============================================================
# E4: 指令敏感性 Prompt
# ============================================================
def newsvendor_instruction_variants(cost=5, price=15, salvage=2, demand_mean=100, demand_std=30,
                                      variant_id=1, lang="zh"):
    """同一问题的不同表述（5种变体）"""
    variants = {
        "zh": {
            1: f"""你是一位运营管理专家。请解决以下库存决策问题：
一家零售商需要决定某季节性商品的订货量。单位采购成本：{cost}元，单位销售价格：{price}元，季末未售出商品的残值：{salvage}元/件。市场需求服从正态分布，均值为{demand_mean}件，标准差为{demand_std}件。
请确定最优订货量Q，使得期望利润最大化。
以JSON格式回答：{{"order_quantity": <整数>, "reasoning": "<推理过程>", "confidence": <0-1>}}""",

            2: f"""嘿，帮我想想：有个店要进一批货，进货价{cost}块，卖{price}块，卖不完的只能卖{salvage}块。一般来说需求大概{demand_mean}件左右，波动±{demand_std}件。你觉得该进多少货比较合适？
用JSON回复：{{"order_quantity": <整数>, "reasoning": "<你的想法>", "confidence": <0-1>}}""",

            3: f"""假设你是沃尔玛的采购经理，负责某季节性商品的订货决策。
采购成本：{cost}元/件，售价：{price}元/件，残值：{salvage}元/件。
需求服从N({demand_mean}, {demand_std}²)。
作为采购经理，请决定最优订货量Q。
以JSON格式回答：{{"order_quantity": <整数>, "reasoning": "<你的推理过程>", "confidence": <0-1>}}""",

            4: f"""⚠️ 警告：以下决策将直接影响公司季度利润，决策错误可能导致严重亏损！

你是一位运营管理专家。一家零售商需要决定某季节性商品的订货量。
- 单位采购成本：{cost}元
- 单位销售价格：{price}元
- 季末残值：{salvage}元/件
- 需求分布：N({demand_mean}, {demand_std}²)

请慎重确定最优订货量Q。
以JSON格式回答：{{"order_quantity": <整数>, "reasoning": "<你的推理过程>", "confidence": <0-1>}}""",

            5: f"""You are an operations management expert. A retailer needs to decide the order quantity for a seasonal product. Unit cost: ${cost}, selling price: ${price}, salvage value: ${salvage}/unit. Demand ~ N({demand_mean}, {demand_std}²). Determine the optimal order quantity Q to maximize expected profit.
Respond in JSON: {{"order_quantity": <integer>, "reasoning": "<your reasoning>", "confidence": <0-1>}}""",
        },
        "en": {
            1: f"""You are an operations management expert. A retailer needs to decide the order quantity for a seasonal product. Unit cost: ${cost}, selling price: ${price}, salvage value: ${salvage}/unit. Demand ~ N({demand_mean}, {demand_std}²). Determine the optimal order quantity Q to maximize expected profit.
Respond in JSON: {{"order_quantity": <integer>, "reasoning": "<your reasoning>", "confidence": <0-1>}}""",

            2: f"""Hey, help me think: a store needs to order some goods. Cost is ${cost}, selling for ${price}, unsold items salvage at ${salvage}. Usually demand is about {demand_mean} units, varies by ±{demand_std}. What quantity would you suggest?
Respond in JSON: {{"order_quantity": <integer>, "reasoning": "<your thoughts>", "confidence": <0-1>}}""",

            3: f"""Assume you are a Walmart purchasing manager responsible for a seasonal product. Purchase cost: ${cost}/unit, selling price: ${price}/unit, salvage: ${salvage}/unit. Demand follows N({demand_mean}, {demand_std}²). As the purchasing manager, determine the optimal order quantity Q.
Respond in JSON: {{"order_quantity": <integer>, "reasoning": "<your reasoning>", "confidence": <0-1>}}""",

            4: f"""⚠️ WARNING: This decision will directly impact quarterly profits. A wrong decision could lead to severe losses!

You are an operations management expert. Determine order quantity Q for a seasonal product. Cost: ${cost}, price: ${price}, salvage: ${salvage}/unit. Demand ~ N({demand_mean}, {demand_std}²).
Respond in JSON: {{"order_quantity": <integer>, "reasoning": "<your reasoning>", "confidence": <0-1>}}""",

            5: f"""你是一位运营管理专家。一家零售商需要决定某季节性商品的订货量。采购成本：{cost}元，售价：{price}元，残值：{salvage}元。需求服从N({demand_mean}, {demand_std}²)。请确定最优订货量Q。
以JSON格式回答：{{"order_quantity": <整数>, "reasoning": "<推理过程>", "confidence": <0-1>}}""",
        }
    }
    return variants[lang][variant_id]


# ============================================================
# E5: 过度自信 Prompt
# ============================================================
def newsvendor_confidence_prompt(cost=5, price=15, salvage=2, demand_mean=100, demand_std=30, lang="zh"):
    """要求给出置信区间"""
    if lang == "zh":
        return f"""你是一位运营管理专家。请解决以下库存决策问题：

【问题描述】
一家零售商需要决定某季节性商品的订货量。
- 单位采购成本：{cost}元
- 单位销售价格：{price}元
- 季末未售出商品的残值：{salvage}元/件
- 市场需求服从正态分布，均值为{demand_mean}件，标准差为{demand_std}件

【决策任务】
1. 请确定最优订货量Q
2. 给出你对最优订货量的90%置信区间 [Q_low, Q_high]（即你90%确信真实最优订货量落在此区间内）

【输出要求】
请以JSON格式回答：
{{"order_quantity": <整数>, "Q_low": <整数>, "Q_high": <整数>, "reasoning": "<你的推理过程>", "confidence": <0-1>}}"""
    else:
        return f"""You are an operations management expert. Please solve the following inventory decision problem:

[Problem Description]
A retailer needs to decide the order quantity for a seasonal product.
- Unit purchase cost: ${cost}
- Unit selling price: ${price}
- Salvage value for unsold units: ${salvage}/unit
- Demand follows a normal distribution with mean {demand_mean} units and standard deviation {demand_std} units

[Decision Task]
1. Determine the optimal order quantity Q
2. Provide a 90% confidence interval [Q_low, Q_high] for the optimal order quantity

[Output Format]
Respond in JSON format:
{{"order_quantity": <integer>, "Q_low": <integer>, "Q_high": <integer>, "reasoning": "<your reasoning>", "confidence": <0-1>}}"""


# ============================================================
# E6: CoT (Chain-of-Thought) 纠偏 Prompt
# ============================================================
def cot_prompt(base_prompt_func, *args, lang="zh", **kwargs):
    """在base_prompt前添加Chain-of-Thought指令"""
    base_prompt = base_prompt_func(*args, lang=lang, **kwargs)

    if lang == "zh":
        cot_prefix = """【重要提示】请按照以下步骤逐步推理，不要跳过任何步骤：

Step 1: 列出所有已知参数（成本、售价、残值、需求分布参数）
Step 2: 写出期望利润函数 E[π(Q)]
Step 3: 使用临界分位数法：α = (p-c)/(p-v)
Step 4: 计算临界分位数 α 的具体数值
Step 5: 计算最优订货量 Q* = μ + Φ⁻¹(α)·σ
Step 6: 验证答案的合理性

请严格按步骤作答，每个步骤都写出具体计算过程，最后以JSON格式输出。

---
"""
    else:
        cot_prefix = """[IMPORTANT] Please reason step by step. Do not skip any steps:

Step 1: List all known parameters (cost, price, salvage, demand distribution)
Step 2: Write the expected profit function E[π(Q)]
Step 3: Use the critical fractile method: α = (p-c)/(p-v)
Step 4: Calculate the specific value of the critical fractile α
Step 5: Calculate the optimal order quantity Q* = μ + Φ⁻¹(α)·σ
Step 6: Verify the reasonableness of your answer

Follow each step strictly, show your calculation for each step, and output in JSON format at the end.

---
"""

    # 移除原prompt中的输出格式要求，因为CoT prefix已经包含了
    # 找到JSON输出要求的位置并移除
    lines = base_prompt.split("\n")
    filtered_lines = []
    skip_json = False
    for line in lines:
        if "JSON格式" in line or "JSON format" in line or "【输出要求】" in line or "[Output Format]" in line:
            skip_json = True
            continue
        if skip_json and line.strip().startswith("{"):
            continue
        if skip_json and line.strip() == "":
            skip_json = False
            continue
        if not skip_json:
            filtered_lines.append(line)
        elif skip_json and line.strip() != "" and not line.strip().startswith("{"):
            skip_json = False
            filtered_lines.append(line)

    base_prompt_clean = "\n".join(filtered_lines)

    cot_output = """【输出要求】
请以JSON格式回答：
{"order_quantity": <整数>, "reasoning": "<按步骤的推理过程>", "confidence": <0-1之间的置信度>}"""

    return cot_prefix + base_prompt_clean + "\n" + cot_output


# ============================================================
# E7: 多Agent辩论 Prompt
# ============================================================
def debate_prompt_agent(base_prompt_func, agent_role, *args, lang="zh", **kwargs):
    """
    多Agent辩论prompt
    agent_role: "conservative" | "aggressive" | "analytical"
    """
    role_texts = {
        "zh": {
            "conservative": "你是一位保守的库存经理，倾向于减少库存风险，避免过量订货导致的损失。",
            "aggressive": "你是一位激进的销售经理，倾向于最大化销售机会，担心缺货导致的利润损失。",
            "analytical": "你是一位数据分析师，严格基于数学计算和数据分析做决策，不受情绪影响。",
        },
        "en": {
            "conservative": "You are a conservative inventory manager who tends to reduce inventory risk and avoid losses from over-ordering.",
            "aggressive": "You are an aggressive sales manager who tends to maximize sales opportunities and fears profit loss from stockouts.",
            "analytical": "You are a data analyst who makes decisions strictly based on mathematical calculations and data analysis, unaffected by emotions.",
        }
    }
    role_text = role_texts[lang][agent_role]

    base_prompt = base_prompt_func(*args, lang=lang, **kwargs)

    if lang == "zh":
        return f"""【角色设定】
{role_text}

---
{base_prompt}"""
    else:
        return f"""[Role Setting]
{role_text}

---
{base_prompt}"""


def debate_synthesis_prompt(agent_answers, lang="zh"):
    """辩论后的综合决策Prompt"""
    if lang == "zh":
        answers_text = ""
        for i, ans in enumerate(agent_answers):
            answers_text += f"Agent {i+1}（订货量={ans['order_quantity']}）：{ans['reasoning'][:200]}...\n\n"

        return f"""你是一位高级运营总监。以下是三位专家对同一报童问题的独立分析：

{answers_text}
请你综合三位专家的意见，做出最终决策。

【输出要求】
请以JSON格式回答：
{{"order_quantity": <整数（最终决策的订货量）>, "reasoning": "<综合三位专家意见的推理过程>", "confidence": <0-1>}}"""
    else:
        answers_text = ""
        for i, ans in enumerate(agent_answers):
            answers_text += f"Agent {i+1} (Q={ans['order_quantity']}): {ans['reasoning'][:200]}...\n\n"

        return f"""You are a senior operations director. Below are independent analyses from three experts on the same newsvendor problem:

{answers_text}
Please synthesize the three experts' opinions and make a final decision.

[Output Format]
Respond in JSON format:
{{"order_quantity": <integer (final decision)>, "reasoning": "<synthesis of the three experts' opinions>", "confidence": <0-1>}}"""


# ============================================================
# E8: 人类反馈校准 Prompt
# ============================================================
def human_calibration_prompt(base_prompt_func, feedback_level, previous_answer, *args, lang="zh", **kwargs):
    """
    人类反馈校准
    feedback_level: "none" | "light" | "strong"
    previous_answer: 上一轮的LLM回答 {"order_quantity": int, "reasoning": str}
    """
    base_prompt = base_prompt_func(*args, lang=lang, **kwargs)

    if feedback_level == "none":
        return base_prompt

    if lang == "zh":
        if feedback_level == "light":
            feedback = f"""【反馈】
你之前的回答中订货量为{previous_answer['order_quantity']}件。你的答案可能存在偏差，请重新考虑并给出修正后的答案。

---
"""
        else:  # strong
            feedback = f"""【反馈】
你之前的回答中订货量为{previous_answer['order_quantity']}件。请注意，最优订货量大约在110-130件之间，请基于此信息重新给出答案。

---
"""
    else:
        if feedback_level == "light":
            feedback = f"""[Feedback]
Your previous answer had an order quantity of {previous_answer['order_quantity']} units. Your answer may have some bias. Please reconsider and provide a revised answer.

---
"""
        else:  # strong
            feedback = f"""[Feedback]
Your previous answer had an order quantity of {previous_answer['order_quantity']} units. Note that the optimal order quantity is approximately between 110-130 units. Please revise your answer based on this information.

---
"""

    return feedback + base_prompt


# ============================================================
# 便捷函数：获取所有Prompt模板
# ============================================================
def get_all_prompt_functions():
    """返回所有Prompt模板函数的映射"""
    return {
        "base": newsvendor_base_prompt,
        "anchored": newsvendor_anchored_prompt,
        "hallucination": newsvendor_hallucination_prompt,
        "instruction_variant": newsvendor_instruction_variants,
        "confidence": newsvendor_confidence_prompt,
        "cot": cot_prompt,
        "debate_agent": debate_prompt_agent,
        "debate_synthesis": debate_synthesis_prompt,
        "human_calibration": human_calibration_prompt,
    }