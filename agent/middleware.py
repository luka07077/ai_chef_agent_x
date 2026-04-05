import time

from langchain.agents import AgentState
from langchain.agents.middleware import before_model, wrap_tool_call, dynamic_prompt
from langgraph.runtime import Runtime

from utils.logger_handler import get_logger
from conf import get_prompt_config

"""
Agent 中间件层 (LangChain Official Middleware)
利用 LangChain 1.0 的官方中间件装饰器，拦截 Agent 主循环中的模型调用与工具调用。

三个核心中间件：
1. @wrap_tool_call   — 监控所有工具执行（包括 MCP 工具），记录入参/出参/耗时/异常
                        同时检测预警类工具调用，自动设置 warning_mode 标志
2. @before_model     — 模型调用前记录上下文信息（消息数量、轮次）
3. @dynamic_prompt   — 根据运行时状态动态切换系统提示词（如安全预警模式）

注意：所有中间件函数均为 async，以兼容 Agent 的异步调用模式（astream / ainvoke）。
"""

logger = get_logger("ai_chef.middleware")

# 触发 warning_mode 的工具名称集合
_WARNING_TOOLS = {"check_fridge_warnings", "check_allergen_safety"}

# 跨中间件共享的运行时上下文
# @wrap_tool_call / @before_model / @dynamic_prompt 各自拿到的 runtime 实例不同，
# 用模块级字典作为唯一共享状态，保证三者读写同一份数据。
_shared_context: dict = {}


# ==========================================
# 中间件 1：工具调用监控 + 预警状态联动
# ==========================================
@wrap_tool_call
async def monitor_tool(request, handler):
    """
    拦截 Agent 循环内的每一次工具调用。
    - 记录工具名、入参、出参、耗时
    - 当检测到预警类工具返回了风险信息时，自动设置 warning_mode 标志，
      触发 @dynamic_prompt 在下一轮模型调用时切换为安全预警提示词
    """
    tool_name = request.tool_call['name']
    tool_args = request.tool_call['args']

    logger.info(f"┌─ TOOL CALL: {tool_name}")
    logger.info(f"│  入参: {tool_args}")

    start_time = time.time()
    try:
        result = await handler(request)
        elapsed = time.time() - start_time

        result_preview = str(result.content if hasattr(result, 'content') else result)[:150]
        logger.info(f"└─ OK ({elapsed:.2f}s): {result_preview}")

        # 预警联动：如果预警工具返回了风险内容，写入模块级共享上下文
        # 注意：MCP 工具的 result.content 可能是 list（多段文本），需强制 str() 转换
        if tool_name in _WARNING_TOOLS:
            result_text = str(result.content) if hasattr(result, 'content') else str(result)
            if "预警" in result_text or "过期" in result_text or "致敏" in result_text:
                _shared_context['warning_mode'] = True
                logger.info("│  >> 检测到风险信号，已激活 warning_mode")

        return result
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"└─ FAILED ({elapsed:.2f}s): {type(e).__name__}: {e}")
        raise


# ==========================================
# 中间件 2：模型调用前置日志
# ==========================================
@before_model
async def log_before_model(state: AgentState, runtime: Runtime) -> None:
    """
    在每次模型调用前记录当前上下文信息。
    第一次模型调用（新一轮对话入口，消息数较少）时重置 warning_mode，
    避免上一轮的预警状态污染下一轮对话。
    """
    msg_count = len(state['messages'])

    # 消息数 <= 2（system + user）说明是新一轮的第一次模型调用，重置预警状态
    if msg_count <= 2:
        _shared_context['warning_mode'] = False

    mode = "安全预警模式" if _shared_context.get('warning_mode', False) else "正常模式"
    logger.info(f"── MODEL CALL: 即将调用模型 [{mode}]，当前上下文共 {msg_count} 条消息")


# ==========================================
# 中间件 3：动态提示词切换
# ==========================================
@dynamic_prompt
async def chef_dynamic_prompt(request):
    """
    根据运行时状态动态切换系统提示词。
    当 warning_mode 被激活时（由 monitor_tool 在检测到预警工具返回风险信号后设置），
    自动从配置文件中读取并追加安全预警专用指令，确保模型在回复中重点关注过敏和过期风险。
    """
    # 1. 加载所有提示词配置
    prompts = get_prompt_config()

    # 2. 获取基础的私厨人设提示词，提供默认后备文案
    base_prompt = prompts.get("chef_system_prompt", "你是一个极具专业素养的 AI 私厨管家。")

    # 3. 从模块级共享上下文读取 warning_mode（由 monitor_tool 在检测到风险时写入）
    is_warning_mode = _shared_context.get('warning_mode', False)

    if is_warning_mode:
        logger.info("── PROMPT SWITCH: 切换为安全预警模式提示词")

        # 4. 从配置中读取预警专用的追加提示词，提供简短的默认后备文案以防配置丢失
        warning_addition = prompts.get(
            "chef_warning_prompt_addition",
            "【当前处于安全预警模式】请重点关注食材风险并给出安全建议。"
        )

        # 5. 将基础提示词与预警指令拼接并返回
        return f"{base_prompt}\n\n{warning_addition}"

    # 正常模式下，仅返回基础提示词
    return base_prompt


# 导出中间件列表，供 chef_agent.py 使用
all_middleware = [monitor_tool, log_before_model, chef_dynamic_prompt]
