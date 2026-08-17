"""
LLM Agent 统一调用类
支持 OpenAI / DashScope (DeepSeek, Qwen) / ZhipuAI (GLM-4)
"""
import os
import json
import time
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class LLMAgent:
    """统一的LLM Agent调用类"""

    def __init__(self, model_name: str, config: dict):
        """
        初始化Agent

        Args:
            model_name: 模型名称 (gpt-4o, deepseek-v4-flash, deepseek-v3, qwen-max, glm-4)
            config: LLM配置字典（来自config.py的LLM_CONFIGS）
        """
        self.model_name = model_name
        self.config = config
        self.provider = config["provider"]
        self.model = config["model"]
        self.api_key = config.get("api_key", "")
        self.api_base = config.get("api_base", "")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)

        self._client = None
        self._init_client()

        # 统计
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def _init_client(self):
        """初始化各平台的客户端"""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info(f"[{self.model_name}] OpenAI client initialized")
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                self._client = None
            except Exception as e:
                logger.error(f"[{self.model_name}] OpenAI init failed: {e}")
                self._client = None

        elif self.provider == "deepseek":
            # DeepSeek uses OpenAI-compatible API
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base or "https://api.deepseek.com",
                )
                logger.info(f"[{self.model_name}] DeepSeek client initialized (base={self.api_base})")
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                self._client = None
            except Exception as e:
                logger.error(f"[{self.model_name}] DeepSeek init failed: {e}")
                self._client = None

        elif self.provider == "dashscope":
            # DashScope使用HTTP API直接调用
            self._dashscope_ready = bool(self.api_key)
            if self._dashscope_ready:
                logger.info(f"[{self.model_name}] DashScope client ready")
            else:
                logger.warning(f"[{self.model_name}] DashScope API key not set")

        elif self.provider == "zhipuai":
            try:
                from zhipuai import ZhipuAI
                self._client = ZhipuAI(api_key=self.api_key)
                logger.info(f"[{self.model_name}] ZhipuAI client initialized")
            except ImportError:
                logger.error("zhipuai package not installed. Run: pip install zhipuai")
                self._client = None
            except Exception as e:
                logger.error(f"[{self.model_name}] ZhipuAI init failed: {e}")
                self._client = None

        else:
            logger.error(f"Unknown provider: {self.provider}")
            self._client = None

    def call(self, prompt: str, temperature: float = None, max_tokens: int = None) -> dict:
        """
        调用LLM

        Args:
            prompt: 提示词
            temperature: 温度参数（覆盖默认值）
            max_tokens: 最大token数（覆盖默认值）

        Returns:
            dict: {"response": str, "tokens": int, "cost": float, "success": bool, "error": str}
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        start_time = time.time()

        try:
            if self.provider == "openai":
                result = self._call_openai(prompt, temp, max_tok)
            elif self.provider == "deepseek":
                result = self._call_openai(prompt, temp, max_tok)  # DeepSeek uses OpenAI-compatible API
            elif self.provider == "dashscope":
                result = self._call_dashscope(prompt, temp, max_tok)
            elif self.provider == "zhipuai":
                result = self._call_zhipuai(prompt, temp, max_tok)
            else:
                return {"response": "", "tokens": 0, "cost": 0, "success": False, "error": f"Unknown provider: {self.provider}"}

            elapsed = time.time() - start_time
            self.total_calls += 1
            self.total_tokens += result.get("tokens", 0)
            self.total_cost += result.get("cost", 0)

            logger.info(
                f"[{self.model_name}] Call #{self.total_calls} | "
                f"Tokens: {result.get('tokens', 0)} | "
                f"Cost: ${result.get('cost', 0):.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{self.model_name}] Call failed after {elapsed:.1f}s: {e}")
            return {"response": "", "tokens": 0, "cost": 0, "success": False, "error": str(e)}

    def _call_openai(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        """调用OpenAI API (also used by DeepSeek via OpenAI-compatible endpoint)"""
        if self._client is None:
            return {"response": "", "tokens": 0, "cost": 0, "success": False, "error": "Client not initialized"}

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        tokens = response.usage.total_tokens

        # 成本估算
        if self.provider == "deepseek":
            # DeepSeek V4 Flash: ~$0.14/$0.28 per 1M input/output tokens
            cost = (response.usage.prompt_tokens * 0.14 + response.usage.completion_tokens * 0.28) / 1_000_000
        else:
            # GPT-4o: $2.50/$10.00 per 1M input/output tokens
            cost = (response.usage.prompt_tokens * 2.5 + response.usage.completion_tokens * 10.0) / 1_000_000

        return {"response": content, "tokens": tokens, "cost": cost, "success": True, "error": None}

    def _call_dashscope(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        """调用DashScope API (DeepSeek/Qwen)"""
        import urllib.request
        import urllib.error

        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                tokens = result.get("usage", {}).get("total_tokens", 0)
                # 成本估算 (简化)
                cost = tokens * 0.002 / 1000
                return {"response": content, "tokens": tokens, "cost": cost, "success": True, "error": None}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            return {"response": "", "tokens": 0, "cost": 0, "success": False, "error": f"HTTP {e.code}: {error_body}"}

    def _call_zhipuai(self, prompt: str, temperature: float, max_tokens: int) -> dict:
        """调用ZhipuAI API (GLM-4)"""
        if self._client is None:
            return {"response": "", "tokens": 0, "cost": 0, "success": False, "error": "Client not initialized"}

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        # 成本估算 (GLM-4: ¥0.1/1M tokens)
        cost = tokens * 0.1 / 1_000_000

        return {"response": content, "tokens": tokens, "cost": cost, "success": True, "error": None}

    def call_with_retry(self, prompt: str, max_retries: int = 3, temperature: float = None,
                        max_tokens: int = None, retry_delay: float = 2.0) -> dict:
        """
        带重试的LLM调用

        Args:
            prompt: 提示词
            max_retries: 最大重试次数
            temperature: 温度参数
            max_tokens: 最大token数
            retry_delay: 重试延迟（秒）

        Returns:
            dict: 同call()方法
        """
        for attempt in range(max_retries):
            result = self.call(prompt, temperature, max_tokens)
            if result["success"]:
                return result

            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # 指数退避
                logger.warning(
                    f"[{self.model_name}] Attempt {attempt+1}/{max_retries} failed: {result.get('error')}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)

        logger.error(f"[{self.model_name}] All {max_retries} attempts failed")
        return result

    def batch_call(self, prompts: list, concurrency: int = 1, **kwargs) -> list:
        """
        批量调用（串行模式，避免rate limit）

        Args:
            prompts: 提示词列表
            concurrency: 仅保留接口兼容性，实际串行执行
            **kwargs: 传递给call_with_retry的参数

        Returns:
            list of dict: 每个prompt的调用结果
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"[{self.model_name}] Batch call {i+1}/{len(prompts)}")
            result = self.call_with_retry(prompt, **kwargs)
            results.append(result)
            # 避免rate limit
            if i < len(prompts) - 1:
                time.sleep(1.0)
        return results

    def get_stats(self) -> dict:
        """获取使用统计"""
        return {
            "model_name": self.model_name,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }

    def reset_stats(self):
        """重置统计"""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0


# ============================================================
# 工厂函数
# ============================================================
def create_agents(configs: dict) -> dict:
    """
    根据配置创建多个Agent

    Args:
        configs: LLM_CONFIGS字典

    Returns:
        dict: {model_name: LLMAgent}
    """
    agents = {}
    for model_name, config in configs.items():
        api_key = config.get("api_key")
        if not api_key or api_key.startswith("your-") or api_key == "":
            logger.warning(f"[{model_name}] API key not configured, skipping")
            continue

        agent = LLMAgent(model_name, config)
        agents[model_name] = agent

    logger.info(f"Created {len(agents)} agents: {list(agents.keys())}")
    return agents


def save_results(results: list, filepath: str):
    """
    保存实验结果为JSONL格式

    Args:
        results: 结果列表，每个元素是一个dict
        filepath: 输出文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} results to {filepath}")


def load_results(filepath: str) -> list:
    """加载JSONL格式的实验结果"""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results