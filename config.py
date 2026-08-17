"""
全局配置文件
ICMSE 2026: 算法有限理性 — 大语言模型Agent运营决策偏差识别与纠偏机制研究
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ============ LLM API 配置 ============
LLM_CONFIGS = {
    "deepseek-v4-flash": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "api_base": "https://api.deepseek.com",
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    "gpt-4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    "deepseek-v3": {
        "provider": "dashscope",
        "model": "deepseek-v3",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    "qwen-max": {
        "provider": "dashscope",
        "model": "qwen-max",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 2048,
    },
    "glm-4": {
        "provider": "zhipuai",
        "model": "glm-4",
        "api_key": os.getenv("ZHIPU_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 2048,
    },
}

# ============ 实验参数 ============
EXPERIMENT_PARAMS = {
    "n_repetitions": 30,          # 每个条件重复30次
    "n_models": 4,                # 4个LLM模型
    "n_bias_conditions": 4,       # 4种偏差条件
    "n_debias_methods": 3,        # 3种纠偏方法
    "random_seed": 42,
    "api_call_interval": 1.0,     # 每次调用间隔（秒），避免rate limit
    "max_retries": 3,             # 失败重试次数
    "concurrency": 1,             # 并发数（保守设置，避免限流）
}

# ============ 报童问题参数 ============
NEWSVENDOR_PARAMS = {
    "unit_cost": 5,               # 单位成本 c=5
    "unit_price": 15,             # 单位售价 p=15
    "salvage_value": 2,           # 残值 v=2
    "demand_distribution": "normal",
    "demand_mean": 100,           # 需求均值 mu=100
    "demand_std": 30,             # 需求标准差 sigma=30
    "n_periods": 20,              # 多周期问题的周期数
}

# 计算最优解 Q* = mu + Phi^{-1}((p-c)/(p-v)) * sigma
from scipy.stats import norm as _norm
_cf = (NEWSVENDOR_PARAMS["unit_price"] - NEWSVENDOR_PARAMS["unit_cost"]) / \
      (NEWSVENDOR_PARAMS["unit_price"] - NEWSVENDOR_PARAMS["salvage_value"])
OPTIMAL_Q = round(NEWSVENDOR_PARAMS["demand_mean"] + 
                  _norm.ppf(_cf) * NEWSVENDOR_PARAMS["demand_std"])
# OPTIMAL_Q ≈ 122

# ============ 输出路径 ============
OUTPUT_DIRS = {
    "raw": "04_Data/raw/",
    "processed": "04_Data/processed/",
    "results": "04_Data/results/",
    "figures": "06_Writing/figures/",
    "tables": "06_Writing/tables/",
}

# 确保输出目录存在
for _dir in OUTPUT_DIRS.values():
    os.makedirs(_dir, exist_ok=True)