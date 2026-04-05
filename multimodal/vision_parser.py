import json
import os

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from conf import get_agent_config, get_prompt_config
from utils.logger_handler import get_logger

"""
视觉解析模块 (Vision Parser)
利用 qwen-vl-max 识别冰箱图片中的食材，提取为 JSON 结构化数据。
采用 LCEL Chain 模式：prompt | model | output_parser
"""

logger = get_logger("ai_chef.vision")


def _extract_text_from_response(raw_content) -> str:
    """从模型多模态响应中提取纯文本"""
    if isinstance(raw_content, list):
        parts = []
        for block in raw_content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(raw_content)


def _clean_json_output(text: str) -> str:
    """清理模型输出中的 Markdown 标签，提取纯 JSON"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_fridge_image(image_path: str, api_key: str = None) -> list:
    """
    核心视觉解析函数：读取图片，通过 LCEL Chain 调用 Qwen-VL 识别食材。

    Chain 结构: 多模态消息构造 -> ChatTongyi(qwen-vl) -> StrOutputParser -> JSON 解析

    Args:
        image_path: 冰箱或食材图片的路径
        api_key: DashScope API Key（不传则读取环境变量）

    Returns:
        list: 包含食材信息的字典列表
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片文件: {image_path}")

    actual_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not actual_api_key:
        raise ValueError("未找到 DASHSCOPE_API_KEY，请在环境中配置。")

    config = get_agent_config()
    llm_config = config["llm"]
    prompts = get_prompt_config()

    # 1. 构建 LCEL Chain 组件
    chat_model = ChatTongyi(
        model_name=llm_config["vision_model"],
        dashscope_api_key=actual_api_key,
        max_tokens=llm_config["vision_max_tokens"],
        temperature=llm_config["vision_temperature"],
    )
    str_parser = StrOutputParser()

    # 2. 构造多模态消息（视觉模型需要直接传 messages，无法用 ChatPromptTemplate）
    abs_image_path = os.path.abspath(image_path)
    file_uri = f"file://{abs_image_path}"

    messages = [
        SystemMessage(content=prompts["vision_system_prompt"]),
        HumanMessage(
            content=[
                {"text": "请帮我盘点一下这些图片里的食材，并按要求返回 JSON。"},
                {"image": file_uri}
            ]
        )
    ]

    try:
        logger.info("正在调用视觉模型进行食材识别...")

        # 3. LCEL Chain 调用: model -> str_parser
        #    视觉模型的多模态消息无法通过 prompt template 构建，
        #    所以这里用 (model | parser) 的链式调用
        chain = chat_model | str_parser
        raw_output = chain.invoke(messages)

        # 4. 清理并解析 JSON
        clean_json = _clean_json_output(raw_output)
        parsed_data = json.loads(clean_json)

        logger.info(f"视觉识别成功，提取到 {len(parsed_data)} 种食材")
        return parsed_data

    except json.JSONDecodeError:
        logger.error(f"JSON 解析失败，模型输出: {raw_output}")
        return []
    except Exception as e:
        logger.error(f"视觉解析过程中发生错误: {e}")
        return []


if __name__ == "__main__":
    from conf import get_project_root

    test_img_path = os.path.join(get_project_root(), "multimodal", "test_fridge.jpg")

    if os.path.exists(test_img_path):
        print("--- 开始测试视觉解析器 ---")
        items = parse_fridge_image(test_img_path)
        print(json.dumps(items, indent=4, ensure_ascii=False))
    else:
        print(f"找不到测试图片: {test_img_path}")
