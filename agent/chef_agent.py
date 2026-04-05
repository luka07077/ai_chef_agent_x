import os
import sys
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from agent.mcp_client_tools import load_mcp_tools
from agent.middleware import all_middleware
from agent.models.local_model import LocalChefModel
from agent.models.cloud_model import CloudChefModel
from multimodal.audio_handler import text_to_speech_tool
from agent.state_manager import StateManager
from rag.agentic_rag_core import search_private_knowledge
from conf import get_agent_config, get_prompt_config
from utils.logger_handler import get_logger

"""
AI 顶级私厨主控大脑 (Chef Agent)
使用 LangChain 的 create_agent 构建 ReAct 工作流，
集成 MCP 工具、多模态感知、Agentic RAG、官方中间件与持久化状态机。

运行模式由 conf/agent_config.yaml 中的 lora.use_lora_adapter 决定：
  false（默认）→ 云端模式：ChatTongyi（qwen3-max），语音/图像可用
  true          → 本地模式：ChatOllama（qwen2.5:7b），MCP/RAG 可用，语音/图像不可用
两种模式共用同一套 Agent 路径，仅 LLM 不同。
"""

logger = get_logger("ai_chef.agent")


def load_system_prompt() -> str:
    """从 prompt_config.yaml 加载系统提示词"""
    prompts = get_prompt_config()
    return prompts.get("chef_system_prompt", "你是一个极具专业素养的 AI 私厨管家。")


async def init_agent_executor():
    """
    初始化 Agent 执行器：
    1. 按配置选择 LLM（云端 ChatTongyi / 本地 ChatOllama）
    2. 加载 MCP 工具 + 本地工具
    3. 通过 create_agent 构建 Agent，挂载官方中间件
    """
    logger.info("正在唤醒 AI 私厨大脑...")

    config      = get_agent_config()
    lora_config = config.get("lora", {})
    use_lora    = lora_config.get("use_lora_adapter", False)
    llm_config  = config["llm"]

    # 1. 按配置选择 LLM（接口一致，仅实例不同）
    if use_lora:
        llm = LocalChefModel(lora_config).llm
        logger.info(f"[本地模式] LLM：{lora_config.get('ollama_model', 'qwen2.5:7b')}")
    else:
        llm = CloudChefModel(llm_config).llm
        logger.info(f"[云端模式] LLM：{llm_config.get('main_model', 'qwen3-max')}")

    # 2. 加载 MCP 与本地工具（两种模式均可用）
    mcp_tools, mcp_client = await load_mcp_tools()
    all_tools = mcp_tools + [text_to_speech_tool, search_private_knowledge]
    logger.info(f"大脑装备完毕！共挂载 {len(all_tools)} 个工具。")

    # 3. 使用 create_agent 构建 Agent，挂载官方中间件
    #    @wrap_tool_call  — 工具监控 + 预警联动
    #    @before_model    — 模型调用前置日志
    #    @dynamic_prompt  — 动态切换系统提示词
    agent_executor = create_agent(
        model=llm,
        tools=all_tools,
        middleware=all_middleware
    )

    return agent_executor, mcp_client


async def interactive_chat():
    """
    终端交互式对话入口，云端/本地模式共用同一路径。
    模式切换只影响 init_agent_executor() 中的 LLM 选择。
    """
    config      = get_agent_config()
    lora_config = config.get("lora", {})
    use_lora    = lora_config.get("use_lora_adapter", False)
    llm_config  = config.get("llm", {})

    logger.info("=" * 55)
    logger.info(f"  运行模式  : {'本地模式（Ollama）' if use_lora else '云端模式（DashScope API）'}")
    if use_lora:
        logger.info(f"  本地模型  : {lora_config.get('ollama_model', 'qwen2.5:7b')}")
        logger.info(f"  Ollama 地址: {lora_config.get('ollama_base_url', 'http://localhost:11434')}")
        logger.info("  语音/图像  : 不可用（阿里百炼额度限制）")
    else:
        logger.info(f"  云端模型  : {llm_config.get('main_model', 'qwen3-max')}")
    logger.info("=" * 55)

    print("\n" + "=" * 50)
    if use_lora:
        print("  御厨·臻享 AI 私厨终端 [本地模式 · Ollama]")
        print("  MCP / RAG 可用 | 语音 / 图像不可用")
    else:
        print("  御厨·臻享 AI 私厨终端 [云端模式]")
    print("  输入 'quit' 退出，'clear' 清除记忆")
    print("=" * 50 + "\n")

    memory_manager      = StateManager()
    current_session_id  = config["user"]["default_session_id"]

    try:
        agent_executor, mcp_client = await init_agent_executor()
        system_msg = SystemMessage(content=load_system_prompt())
        logger.info(f"已加载用户 [{current_session_id}] 的历史记忆")

        while True:
            user_input = input("\n您: ")

            if user_input.lower() in ['quit', 'exit']:
                print("私厨管家: 期待下次为您服务！祝您胃口好！")
                break

            if user_input.lower() == 'clear':
                memory_manager.clear_memory(current_session_id)
                continue

            if not user_input.strip():
                continue

            try:
                chat_history = memory_manager.load_history(current_session_id)
                messages = [system_msg] + chat_history + [HumanMessage(content=user_input)]

                response = await agent_executor.ainvoke({"messages": messages})
                bot_reply = response["messages"][-1].content
                print(f"\n私厨: {bot_reply}")

                memory_manager.add_conversation(
                    session_id=current_session_id,
                    human_text=user_input,
                    ai_text=bot_reply
                )

            except Exception as e:
                logger.error(f"Agent 执行异常: {e}")
                print(f"\n大脑短路了: {str(e)}")

    finally:
        if 'mcp_client' in locals():
            logger.info("正在安全断开微服务连接...")


async def get_chef_response(agent_executor, messages) -> str:
    """
    非流式调用：获取 Agent 完整回复。
    官方中间件通过 middleware 参数挂载到 Agent，在调用过程中自动生效。
    """
    response = await agent_executor.ainvoke({"messages": messages})
    return response["messages"][-1].content


async def get_chef_response_stream(agent_executor, messages):
    """
    混合方案：用 astream 获取 Agent 每一步的输出，
    只对最终的 AI 文本回复做逐字 yield。
    """
    final_content = ""

    async for chunk in agent_executor.astream(
            {"messages": messages},
            stream_mode="updates"
    ):
        if not isinstance(chunk, dict):
            continue
        for node_name, node_output in chunk.items():
            if not node_output or not isinstance(node_output, dict):
                continue
            if "messages" not in node_output:
                continue
            for msg in node_output["messages"]:
                if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                    final_content = msg.content

    if final_content:
        for char in final_content:
            yield char
            await asyncio.sleep(0.02)  # 控制打字速度


if __name__ == "__main__":
    asyncio.run(interactive_chat())
