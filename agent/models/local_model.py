"""
御厨·臻享 本地模型封装（langchain-ollama 版）
========================================
当 agent_config.yaml 中 lora.use_lora_adapter: true 时启用。
通过 ChatOllama 调用本地 GGUF 微调模型，与云端 CloudChefModel 保持相同的调用风格。

前置条件：
  1. 已安装并启动 Ollama（ollama serve）
  2. 已将 chef_lora_q4.gguf 注册到 Ollama：
       ollama create chef-lora -f lora_tuning/data/Modelfile
     或直接使用基础模型：
       ollama pull qwen2.5:7b

安装依赖：pip install langchain-ollama
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from utils.logger_handler import get_logger

logger = get_logger("ai_chef.local_model")

DEFAULT_SYSTEM_PROMPT = (
    "你是「御厨·臻享」的专属 AI 私厨助手，精通中西料理、营养搭配与食品安全，"
    "以专业、优雅的语气为用户提供定制化的烹饪建议。"
)


class LocalChefModel:
    """
    基于 ChatOllama 的本地模型封装。

    使用方式：
        model = LocalChefModel(lora_config)
        reply = model.chat("鸡胸肉怎么做才嫩？")
    """

    def __init__(self, lora_config: dict):
        model_name  = lora_config.get("ollama_model", "chef-lora")
        base_url    = lora_config.get("ollama_base_url", "http://localhost:11434")
        temperature = lora_config.get("temperature", 0.7)
        max_tokens  = lora_config.get("max_new_tokens", 512)

        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )
        logger.info(f"[本地模式] ChatOllama 初始化完成，模型：{model_name} @ {base_url}")

