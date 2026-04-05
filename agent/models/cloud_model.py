"""
御厨·臻享 云端模型封装（ChatTongyi / DashScope 版）
===================================================
当 agent_config.yaml 中 lora.use_lora_adapter: false（默认）时启用。
封装通义千问大模型的初始化逻辑，与本地 LocalChefModel 保持对称结构。

前置条件：
  设置环境变量 DASHSCOPE_API_KEY
"""

import os
from langchain_community.chat_models import ChatTongyi
from utils.logger_handler import get_logger

logger = get_logger("ai_chef.cloud_model")


class CloudChefModel:
    """
    基于 ChatTongyi（通义千问）的云端模型封装。
    初始化后通过 .llm 属性将实例传给 Agent 使用。

    使用方式：
        cloud = CloudChefModel(llm_config)
        agent_executor = create_agent(model=cloud.llm, tools=all_tools, ...)
    """

    def __init__(self, llm_config: dict):
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY 环境变量！")

        model_name  = llm_config.get("main_model", "qwen3-max")
        temperature = llm_config.get("main_temperature", 0.7)

        self.llm = ChatTongyi(
            model_name=model_name,
            dashscope_api_key=api_key,
            temperature=temperature,
        )
        logger.info(f"[云端模式] ChatTongyi 初始化完成，模型：{model_name}")
