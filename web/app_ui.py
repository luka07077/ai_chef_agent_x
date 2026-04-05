import os
import sys
import asyncio
import time
from datetime import datetime

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from agent.chef_agent import init_agent_executor, load_system_prompt, get_chef_response
from agent.state_manager import StateManager
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from fridge_manager import fridge_db
from multimodal.audio_handler import speech_to_text
from multimodal.vision_parser import parse_fridge_image
from rag.vector_stores import ingest_single_document, is_document_ingested
from conf import get_agent_config
from utils.logger_handler import get_logger

logger = get_logger("ai_chef.web")

# ==========================================
# 页面全局配置
# ==========================================
st.set_page_config(
    page_title="AI 顶级私厨与智能食材管家",
    page_icon="👨‍🍳",
    layout="wide"
)

config       = get_agent_config()
lora_config  = config.get("lora", {})
USE_LORA     = lora_config.get("use_lora_adapter", False)
LOCAL_MODEL  = lora_config.get("ollama_model", "qwen2.5:7b")
CLOUD_MODEL  = config.get("llm", {}).get("main_model", "qwen3-max")

memory_manager = StateManager()
USER_ID    = config["user"]["default_user_id"]
SESSION_ID = config["user"]["default_session_id"]

# 防抖 + 状态持久化
for key, default in [
    ("processed_audio_id", None),
    ("processed_img_id", None),
    ("latest_audio_path", None),
    ("processed_kb_id", None),
    ("agent_executor", None),
    ("mcp_client", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ==========================================
# Agent 初始化（缓存，避免每次 rerun 重新连接 MCP）
# 云端/本地模式共用，LLM 由 init_agent_executor 内部按配置选择
# ==========================================
async def _ensure_agent():
    if st.session_state.agent_executor is None:
        executor, client = await init_agent_executor()
        st.session_state.agent_executor = executor
        st.session_state.mcp_client = client
    return st.session_state.agent_executor


def _get_agent_sync():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_ensure_agent())
    finally:
        loop.close()


# ==========================================
# Agent 响应（生成器式流式输出）
# ==========================================
async def _stream_agent_response(user_input: str):
    """异步生成器：流式返回 Agent 的每个 token"""
    from agent.chef_agent import get_chef_response_stream

    agent_executor = await _ensure_agent()
    chat_history = memory_manager.load_history(SESSION_ID)
    system_msg = SystemMessage(content=load_system_prompt())
    messages = [system_msg] + chat_history + [HumanMessage(content=user_input)]

    full_reply = ""
    async for token in get_chef_response_stream(agent_executor, messages):
        full_reply += token
        yield token

    memory_manager.add_conversation(SESSION_ID, user_input, full_reply)


def stream_agent_response(user_input: str):
    """
    同步包装器：将异步生成器转为同步生成器，
    供 Streamlit 的 st.write_stream 使用。
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        agen = _stream_agent_response(user_input)
        while True:
            try:
                token = loop.run_until_complete(agen.__anext__())
                yield token
            except StopAsyncIteration:
                break
    finally:
        loop.close()


# ==========================================
# 侧边栏：控制面板 + 冰箱监控 + 知识库上传
# ==========================================
with st.sidebar:
    st.header("⚙️ 管家控制台")
    if USE_LORA:
        st.info(f"🔌 本地模式 · `{LOCAL_MODEL}`")
        st.caption("语音与图像识别不可用（阿里百炼额度限制）")
    else:
        st.success(f"☁️ 云端模式 · `{CLOUD_MODEL}`")
    st.write(f"当前用户: `{SESSION_ID}`")

    if st.button("🧹 清空大脑记忆", use_container_width=True):
        memory_manager.clear_memory(SESSION_ID)
        st.session_state.latest_audio_path = None
        st.session_state.agent_executor = None
        st.session_state.mcp_client = None
        st.success("记忆已清空，大厨已重新上岗！")
        st.rerun()

    st.divider()

    # 冰箱实时库存
    st.header("❄️ 冰箱实时库存")
    inventory = fridge_db.get_active_inventory(USER_ID)
    if not inventory:
        st.info("冰箱空空如也，快让大厨帮忙买点菜吧！")
    else:
        today = datetime.now().date()
        for item in inventory:
            expiration_date = datetime.strptime(item['expiration_date'], "%Y-%m-%d").date()
            days_to_expire = (expiration_date - today).days
            name = item['item_name']
            quantity = item['quantity']
            unit = item['unit']

            if days_to_expire < 0:
                st.error(f"❌ {name} ({quantity}{unit}) - 已过期！({abs(days_to_expire)} 天)")
            elif days_to_expire <= 2:
                st.warning(f"⚠️ {name} ({quantity}{unit}) - 剩 {days_to_expire} 天过期！")
            elif days_to_expire <= 5:
                st.info(f"🔸 {name} ({quantity}{unit}) - 剩 {days_to_expire} 天")
            else:
                st.success(f"✅ {name} ({quantity}{unit}) - 剩 {days_to_expire} 天")

    st.divider()

    # 知识库录入（支持 txt / pdf / docx，带去重检测）
    st.header("📚 传授私房秘籍 (知识库录入)")
    st.caption("上传菜谱或营养学文档，大厨会将其铭记于心。")

    kb_file = st.file_uploader("支持 .txt / .pdf / .docx 格式", type=["txt", "pdf", "docx"])

    if kb_file is not None and kb_file.file_id != st.session_state.processed_kb_id:
        st.session_state.processed_kb_id = kb_file.file_id

        upload_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        kb_path = os.path.join(upload_dir, kb_file.name)

        with open(kb_path, "wb") as f:
            f.write(kb_file.getbuffer())

        if is_document_ingested(kb_path):
            st.warning(f"《{kb_file.name}》的内容已经录入过知识库了，无需重复上传哦！")
        else:
            with st.spinner(f"🧠 正在解析 {kb_file.name}，将其刻入大脑深处..."):
                success = ingest_single_document(kb_path)
                if success:
                    st.success(f"《{kb_file.name}》已成功融会贯通！")
                else:
                    st.error("秘籍录入失败，请检查文件格式或内容。")

# ==========================================
# 主界面：多模态交互区
# ==========================================
st.title("👨‍🍳 AI 顶级私厨 & 智能食材管家")
st.caption("集成了大模型 Agent、MCP 微服务、Agentic RAG、多模态识别与状态机的全栈 AI 项目。")

# 渲染历史对话
chat_history = memory_manager.load_history(SESSION_ID)
for msg in chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="👨‍🍳"):
            st.write(msg.content)

# 渲染持久化的语音播报（自动播放）
if st.session_state.latest_audio_path and os.path.exists(st.session_state.latest_audio_path):
    st.divider()
    st.subheader("🔊 大厨语音播报")
    try:
        with open(st.session_state.latest_audio_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav", autoplay=True)
    except Exception as e:
        st.warning(f"音频播放失败：{str(e)}")
    st.divider()

# 多模态输入区（语音和图像仅云端模式可用）
user_text = st.chat_input("想吃点什么？或者有什么吩咐...")

if USE_LORA:
    st.caption("🔌 本地模式下语音与图像识别不可用（阿里百炼额度限制）")
    audio_file = None
    img_file   = None
else:
    col1, col2 = st.columns(2)
    with col1:
        audio_file = st.file_uploader("🎙️ 上传语音指令 (wav/mp3)", type=["wav", "mp3", "m4a"])
    with col2:
        img_file = st.file_uploader("📸 上传冰箱照片", type=["jpg", "png", "jpeg"])

# ==========================================
# 核心逻辑：处理用户输入
# ==========================================
final_input = ""

# 优先级 1：文本输入
if user_text:
    final_input = user_text

# 优先级 2：语音输入（仅云端模式）
elif audio_file is not None and audio_file.file_id != st.session_state.processed_audio_id:
    st.session_state.processed_audio_id = audio_file.file_id

    audio_dir = os.path.join(PROJECT_ROOT, "data", "audio")
    os.makedirs(audio_dir, exist_ok=True)
    temp_audio_path = os.path.join(audio_dir, "temp_upload.wav")

    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    with st.spinner("🎧 正在听您说话..."):
        recognized_text = speech_to_text(temp_audio_path)
        if recognized_text:
            final_input = f"【语音指令】: {recognized_text}"
            st.toast(f"识别成功: {recognized_text}", icon="✅")

# 优先级 3：图像输入（仅云端模式）
elif img_file is not None and img_file.file_id != st.session_state.processed_img_id:
    st.session_state.processed_img_id = img_file.file_id

    upload_dir = os.path.join(PROJECT_ROOT, "data", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    temp_img_path = os.path.join(upload_dir, "temp_fridge_upload.jpg")

    with open(temp_img_path, "wb") as f:
        f.write(img_file.getbuffer())

    with st.spinner("👁️ 正在用视觉神经扫描您的冰箱..."):
        try:
            extracted_items = parse_fridge_image(temp_img_path)

            if extracted_items:
                st.toast(f"成功识别出 {len(extracted_items)} 种食材！", icon="✅")

                added_items_str = ""
                for item in extracted_items:
                    fridge_db.add_food_item(
                        user_id=USER_ID,
                        item_name=item.get("item_name", "未知食材"),
                        quantity=item.get("quantity", 1),
                        unit=item.get("unit", "个"),
                        days_to_expire=item.get("days_to_expire", 3)
                    )
                    added_items_str += (
                        f"- {item.get('item_name')}: "
                        f"{item.get('quantity')}{item.get('unit')} "
                        f"(预计保鲜 {item.get('days_to_expire')} 天)\n"
                    )

                final_input = (
                    f"【视觉感知与自动入库完毕】我刚刚上传了一张冰箱照片，"
                    f"视觉模块识别出了以下食材并已自动存入虚拟冰箱：\n\n"
                    f"{added_items_str}\n"
                    f"请结合冰箱里现有的所有食材，给我推荐今晚的菜谱。"
                )
            else:
                st.error("视觉模块没有识别出任何有效食材。")
        except Exception as e:
            st.error(f"视觉解析过程出现异常: {str(e)}")

# ==========================================
# 统一发送到 Agent（云端/本地模式相同路径）
# ==========================================
if final_input:
    with st.chat_message("user", avatar="👤"):
        st.write(final_input)

    with st.chat_message("assistant", avatar="👨‍🍳"):
        spinner_text = (
            f"🔌 本地模型（{LOCAL_MODEL}）思考中..."
            if USE_LORA else
            "🧠 大厨正在思考并调度微服务..."
        )
        with st.spinner(spinner_text):
            reply = st.write_stream(stream_agent_response(final_input))

        # 语音检测（仅云端模式有语音生成）
        if not USE_LORA:
            audio_dir = os.path.join(PROJECT_ROOT, "data", "audio")
            if os.path.exists(audio_dir):
                now = time.time()
                wav_files = [
                    os.path.join(audio_dir, f)
                    for f in os.listdir(audio_dir)
                    if f.endswith(".wav") and f != "temp_upload.wav"
                ]
                if wav_files:
                    latest_audio = max(wav_files, key=os.path.getctime)
                    if now - os.path.getctime(latest_audio) < 30:
                        st.session_state.latest_audio_path = latest_audio
                        with open(latest_audio, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav", autoplay=True)

    st.rerun()
