import os
import time
import dashscope
from dashscope.audio.tts import SpeechSynthesizer
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from conf import get_project_root, get_agent_config, get_prompt_config
from utils.logger_handler import get_logger

"""
语音处理模块 (Audio Handler)
处理多模态中的"语音"输入（ASR）与输出（TTS）。
ASR 采用 LCEL Chain 模式：model | StrOutputParser
所有生成的音频统一存放在 data/audio/ 目录下。
"""

logger = get_logger("ai_chef.audio")

AUDIO_DIR = os.path.join(get_project_root(), "data", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


def speech_to_text(audio_path: str, api_key: str = None) -> str:
    """
    ASR (语音转文本)：通过 LCEL Chain 调用 ChatTongyi 识别语音。

    Chain 结构: 多模态音频消息 -> ChatTongyi(qwen-audio) -> StrOutputParser
    """
    if not os.path.exists(audio_path):
        logger.error(f"找不到音频文件: {audio_path}")
        return ""

    actual_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not actual_api_key:
        raise ValueError("未找到 DASHSCOPE_API_KEY，请在环境变量中配置。")

    config = get_agent_config()
    prompts = get_prompt_config()

    logger.info(f"正在通过 ASR 识别语音: {os.path.basename(audio_path)}")

    # 1. 构建 LCEL Chain: model | str_parser
    chat_audio = ChatTongyi(
        model_name=config["llm"]["audio_asr_model"],
        dashscope_api_key=actual_api_key,
        temperature=0.1,
    )
    str_parser = StrOutputParser()
    chain = chat_audio | str_parser

    # 2. 构造多模态音频消息
    abs_audio_path = os.path.abspath(audio_path)
    file_uri = f"file://{abs_audio_path}"

    messages = [
        SystemMessage(content=prompts["asr_system_prompt"]),
        HumanMessage(
            content=[
                {"text": prompts["asr_user_prompt"]},
                {"audio": file_uri}
            ]
        )
    ]

    try:
        # 3. 链式调用，直接得到 str
        result_text = chain.invoke(messages)
        logger.info(f"ASR 识别完成: {result_text[:50]}...")
        return result_text.strip()

    except Exception as e:
        logger.error(f"语音识别错误: {e}")
        return ""


@tool
def text_to_speech_tool(text: str) -> str:
    """
    将文本合成为语音文件的工具。
    当需要向用户播报菜谱、提醒过期食材或进行口语交流时，调用此工具。

    Args:
        text: 需要转换的文本内容。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return "失败：未找到 DASHSCOPE_API_KEY 环境变量。"

    config = get_agent_config()
    dashscope.api_key = api_key
    logger.info(f"正在合成语音: '{text[:30]}...'")

    try:
        result = SpeechSynthesizer.call(
            model=config["llm"]["audio_tts_model"],
            text=text,
            sample_rate=config["llm"]["audio_sample_rate"],
            format='wav'
        )

        if result.get_audio_data() is not None:
            timestamp = int(time.time())
            filename = f"chef_reply_{timestamp}.wav"
            final_output_path = os.path.join(AUDIO_DIR, filename)

            with open(final_output_path, 'wb') as f:
                f.write(result.get_audio_data())

            logger.info(f"语音合成成功: {final_output_path}")
            return f"语音合成成功，文件已保存至: {final_output_path}"
        else:
            return f"语音合成失败: 状态码 {result.get_status_code()}, 信息 {result.get_message()}"

    except Exception as e:
        logger.error(f"TTS 异常: {e}")
        return f"工具执行异常: {str(e)}"


if __name__ == "__main__":
    test_text = "你好！我是你的智能私厨管家。"
    print("--- TTS 测试 ---")
    tts_result = text_to_speech_tool.invoke({"text": test_text})
    print(tts_result)
