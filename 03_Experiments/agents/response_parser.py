"""
LLM Agent 响应解析器
解析LLM返回的JSON响应，容错处理非标准JSON格式
"""
import json
import re
import logging

logger = logging.getLogger(__name__)


def parse_response(response_text: str) -> dict:
    """
    解析LLM响应，提取JSON格式的决策数据。

    容错策略：
    1. 尝试直接解析整个文本为JSON
    2. 用正则提取JSON块
    3. 用正则逐个提取字段

    返回:
        dict with keys: order_quantity, reasoning, confidence, (Q_low, Q_high)
    """
    result = {
        "order_quantity": None,
        "reasoning": "",
        "confidence": None,
        "Q_low": None,
        "Q_high": None,
        "raw_response": response_text,
        "parse_success": False,
        "parse_method": "none",
    }

    if not response_text:
        return result

    # 策略1: 尝试直接解析整个文本
    try:
        data = json.loads(response_text)
        result.update(_extract_fields(data))
        result["parse_success"] = True
        result["parse_method"] = "direct_json"
        return result
    except json.JSONDecodeError:
        pass

    # 策略2: 用正则提取JSON块
    json_patterns = [
        r'\{[^{}]*"order_quantity"[^{}]*\}',  # 含order_quantity的JSON
        r'\{[^{}]*\}',  # 任意JSON块
        r'```json\s*(\{.*?\})\s*```',  # markdown代码块
        r'```\s*(\{.*?\})\s*```',  # 无语言标记的代码块
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                data = json.loads(json_str)
                result.update(_extract_fields(data))
                result["parse_success"] = True
                result["parse_method"] = "regex_json"
                return result
            except (json.JSONDecodeError, AttributeError):
                continue

    # 策略3: 逐个提取字段
    result = _extract_fields_regex(response_text, result)
    if result["order_quantity"] is not None:
        result["parse_success"] = True
        result["parse_method"] = "regex_fields"

    return result


def _extract_fields(data: dict) -> dict:
    """从JSON字典中提取字段"""
    fields = {}
    for key in ["order_quantity", "order_quantity", "Q", "q", "quantity", "order"]:
        if key in data and data[key] is not None:
            try:
                fields["order_quantity"] = int(float(str(data[key]).replace(",", "")))
            except (ValueError, TypeError):
                fields["order_quantity"] = None
            break

    for key in ["reasoning", "reason", "explanation", "analysis"]:
        if key in data and data[key]:
            fields["reasoning"] = str(data[key])
            break

    for key in ["confidence", "conf", "certainty"]:
        if key in data and data[key] is not None:
            try:
                fields["confidence"] = float(data[key])
            except (ValueError, TypeError):
                fields["confidence"] = None
            break

    for key in ["Q_low", "q_low", "lower_bound", "low"]:
        if key in data and data[key] is not None:
            try:
                fields["Q_low"] = int(float(str(data[key]).replace(",", "")))
            except (ValueError, TypeError):
                fields["Q_low"] = None
            break

    for key in ["Q_high", "q_high", "upper_bound", "high"]:
        if key in data and data[key] is not None:
            try:
                fields["Q_high"] = int(float(str(data[key]).replace(",", "")))
            except (ValueError, TypeError):
                fields["Q_high"] = None
            break

    return fields


def _extract_fields_regex(text: str, result: dict) -> dict:
    """用正则逐个提取字段"""
    # 提取订货量
    q_patterns = [
        r'"order_quantity"\s*:\s*(\d+)',
        r'"Q"\s*:\s*(\d+)',
        r'订货量[：:]\s*(\d+)',
        r'order quantity[：:]\s*(\d+)',
        r'Q\s*=\s*(\d+)',
        r'订货(\d+)件',
        r'order\s+(\d+)\s+units',
    ]
    for pattern in q_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["order_quantity"] = int(match.group(1))
            break

    # 提取推理过程
    reasoning_patterns = [
        r'"reasoning"\s*:\s*"([^"]*)"',
        r'"reason"\s*:\s*"([^"]*)"',
    ]
    for pattern in reasoning_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result["reasoning"] = match.group(1)
            break

    if not result["reasoning"]:
        # 如果没有提取到reasoning字段，使用整个文本（去除JSON部分）
        result["reasoning"] = text[:500]

    # 提取置信度
    conf_patterns = [
        r'"confidence"\s*:\s*([\d.]+)',
        r'"conf"\s*:\s*([\d.]+)',
    ]
    for pattern in conf_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                result["confidence"] = float(match.group(1))
            except ValueError:
                pass
            break

    # 提取置信区间
    for key, patterns in [
        ("Q_low", [r'"Q_low"\s*:\s*(\d+)', r'"lower_bound"\s*:\s*(\d+)']),
        ("Q_high", [r'"Q_high"\s*:\s*(\d+)', r'"upper_bound"\s*:\s*(\d+)']),
    ]:
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                result[key] = int(match.group(1))
                break

    return result


def validate_response(parsed: dict, min_q=0, max_q=500) -> bool:
    """
    验证解析结果的合理性

    返回:
        True if valid, False otherwise
    """
    if parsed["order_quantity"] is None:
        return False
    if not (min_q <= parsed["order_quantity"] <= max_q):
        logger.warning(f"Order quantity {parsed['order_quantity']} out of range [{min_q}, {max_q}]")
        return False
    if parsed["confidence"] is not None and not (0 <= parsed["confidence"] <= 1):
        parsed["confidence"] = max(0, min(1, parsed["confidence"]))
    return True


def batch_parse(responses: list) -> list:
    """批量解析响应"""
    results = []
    for resp in responses:
        parsed = parse_response(resp)
        results.append(parsed)
    return results


def parse_success_rate(parsed_results: list) -> float:
    """计算解析成功率"""
    if not parsed_results:
        return 0.0
    success = sum(1 for r in parsed_results if r["parse_success"])
    return success / len(parsed_results)