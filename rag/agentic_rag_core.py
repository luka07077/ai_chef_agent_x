import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from rag.vector_stores import search_knowledge
from conf import get_agent_config, get_prompt_config
from utils.logger_handler import get_logger

"""
Agentic RAG 核心模块
实现真正的 Agentic RAG 三步走：
  1. Query Rewriting (查询改写) - 优化用户问题以提升召回率
  2. Multi-path Retrieval (多路召回) - 原始查询 + 改写查询同时检索并去重
  3. Self-Reflection (自我反思) - 质检员模型判断检索结果是否相关并提纯
"""

logger = get_logger("ai_chef.agentic_rag")


def _get_evaluator_llm():
    """获取轻量质检模型（不占用主脑算力）"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    config = get_agent_config()
    return ChatTongyi(
        model_name=config["llm"]["evaluator_model"],
        dashscope_api_key=api_key,
        temperature=config["llm"]["evaluator_temperature"]
    )


def _rewrite_query(query: str, llm) -> list:
    """
    Step 1: Query Rewriting (查询改写)
    将用户的自然语言问题改写为更适合向量检索的关键词组合。
    """
    prompts = get_prompt_config()
    rewrite_prompt = ChatPromptTemplate.from_template(prompts["rag_query_rewrite_prompt"])
    chain = rewrite_prompt | llm

    try:
        result = chain.invoke({"query": query})
        rewritten = [q.strip() for q in result.content.strip().split("\n") if q.strip()]
        logger.info(f"[Query Rewriting] 原始: '{query}' -> 改写: {rewritten}")
        return rewritten
    except Exception as e:
        logger.warning(f"[Query Rewriting] 改写失败，使用原始查询: {e}")
        return []


def _multi_path_retrieve(query: str, rewritten_queries: list, k: int = 3) -> list:
    """
    Step 2: Multi-path Retrieval (多路召回)
    原始查询和改写查询同时检索，结果去重合并。
    """
    all_docs = []
    seen_contents = set()

    # 原始查询检索
    for doc in search_knowledge(query, k=k):
        content_key = doc.page_content[:100]
        if content_key not in seen_contents:
            seen_contents.add(content_key)
            all_docs.append(doc)

    # 改写查询检索
    for rq in rewritten_queries:
        for doc in search_knowledge(rq, k=k):
            content_key = doc.page_content[:100]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                all_docs.append(doc)

    logger.info(f"[Multi-path Retrieval] 共召回 {len(all_docs)} 个去重文档片段")
    return all_docs


def _self_reflect(query: str, docs: list, llm) -> str:
    """
    Step 3: Self-Reflection (自我反思与提纯)
    质检员判断检索结果是否真的能回答用户问题。
    """
    raw_context = "\n\n".join(
        [f"【来源: {d.metadata.get('source', '未知')}】\n{d.page_content}" for d in docs]
    )

    prompts = get_prompt_config()
    reflection_prompt = ChatPromptTemplate.from_template(prompts["rag_reflection_prompt"])
    chain = reflection_prompt | llm

    result = chain.invoke({"query": query, "context": raw_context})
    return result.content.strip()


@tool
def search_private_knowledge(query: str) -> str:
    """
    当用户询问具体的菜谱做法、养生建议、营养搭配等需要从知识库检索的问题时，调用此工具。
    此工具实现了完整的 Agentic RAG 流程：查询改写 -> 多路召回 -> 自我反思质检。

    Args:
        query: 用户的原始问题或检索关键词。
    """
    logger.info(f"[Agentic RAG] 启动完整检索流程，原始查询: '{query}'")

    # 获取质检模型（复用同一个实例）
    evaluator_llm = _get_evaluator_llm()

    # Step 1: 查询改写
    rewritten_queries = _rewrite_query(query, evaluator_llm)

    # Step 2: 多路召回
    config = get_agent_config()
    k = config["rag"]["retrieval_top_k"]
    docs = _multi_path_retrieve(query, rewritten_queries, k=k)

    if not docs:
        logger.warning("[Agentic RAG] 知识库中未检索到任何相关内容")
        return "本地知识库中未检索到相关内容，请依靠通用知识回答。"

    # Step 3: 自我反思与提纯
    logger.info("[Agentic RAG] 内部质检员正在进行自我反思 (Self-Reflection)...")
    output = _self_reflect(query, docs, evaluator_llm)

    # 路由判断
    if "NOT_FOUND" in output:
        logger.info("[Agentic RAG] 反思结论：检索文档与问题无关，已阻断。")
        return "检索到的文档与问题无关，请依靠通用知识回答用户。"
    else:
        logger.info("[Agentic RAG] 反思结论：内容高度相关，已提纯并交付。")
        return f"【来自知识库的高纯度信息】\n{output}"


if __name__ == "__main__":
    print("=== Agentic RAG 独立测试 ===")
    test_q = "根据我的体检报告，我能吃海鲜吗？"
    result = search_private_knowledge.invoke({"query": test_q})
    print(f"\n最终返回:\n{result}")
