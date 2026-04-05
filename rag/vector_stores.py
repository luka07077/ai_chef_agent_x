import os
import hashlib

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from conf import get_project_root, get_agent_config
from utils.logger_handler import get_logger

"""
向量数据库模块 (Vector Stores)
负责用户上传的菜谱、营养学文档等公开知识的向量化与检索。
支持 .txt / .pdf / .docx 格式。
注意：local_privacy 目录下的私密文件不进入此向量库，由 MCP 文件系统服务按需读取。
"""

logger = get_logger("ai_chef.rag")

# 数据路径
PROJECT_ROOT = get_project_root()
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "data", "chroma_db")
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

# 记录已入库文档的指纹，防止重复注入
_INGESTED_HASHES_FILE = os.path.join(CHROMA_PERSIST_DIR, ".ingested_hashes")


def _load_ingested_hashes() -> set:
    """加载已入库文档的哈希集合"""
    if os.path.exists(_INGESTED_HASHES_FILE):
        with open(_INGESTED_HASHES_FILE, "r") as f:
            return set(f.read().splitlines())
    return set()


def _save_ingested_hash(file_hash: str):
    """记录新入库文档的哈希"""
    os.makedirs(os.path.dirname(_INGESTED_HASHES_FILE), exist_ok=True)
    with open(_INGESTED_HASHES_FILE, "a") as f:
        f.write(file_hash + "\n")


def _file_hash(file_path: str) -> str:
    """计算文件内容的 MD5 哈希"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_document_ingested(file_path: str) -> bool:
    """
    检查文件是否已经入库过（供 UI 层调用做前置判断）。
    基于文件内容 MD5 哈希，相同内容 = 已入库。
    """
    if not os.path.exists(file_path):
        return False
    return _file_hash(file_path) in _load_ingested_hashes()


def get_embeddings_model():
    """初始化通义千问的文本向量化模型"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY 环境变量！")

    config = get_agent_config()
    return DashScopeEmbeddings(
        dashscope_api_key=api_key,
        model=config["llm"]["embedding_model"]
    )


def get_vector_store():
    """获取或初始化 Chroma 向量数据库实例"""
    config = get_agent_config()
    embeddings = get_embeddings_model()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=config["rag"]["collection_name"]
    )


def _get_text_splitter():
    """获取配置化的文本切分器"""
    config = get_agent_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=config["rag"]["chunk_size"],
        chunk_overlap=config["rag"]["chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
    )


def search_knowledge(query: str, k: int = None):
    """提供给 RAG 核心或者 Agent 调用的检索接口"""
    if k is None:
        config = get_agent_config()
        k = config["rag"]["retrieval_top_k"]

    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)


def ingest_single_document(file_path: str) -> bool:
    """
    接收单个文件 (支持 txt / pdf / docx)，切分并注入到向量数据库中。
    内置去重机制：相同内容的文件不会重复入库。

    Returns:
        True: 入库成功（或已存在，跳过）
        False: 入库失败
    """
    if not os.path.exists(file_path):
        logger.error(f"找不到文件: {file_path}")
        return False

    # 去重检查
    fhash = _file_hash(file_path)
    if fhash in _load_ingested_hashes():
        logger.info(f"文件已入库过，跳过: {os.path.basename(file_path)}")
        return True

    logger.info(f"正在解析并录入: {os.path.basename(file_path)}")

    try:
        # 根据文件后缀选择 Loader
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.txt':
            loader = TextLoader(file_path, autodetect_encoding=True)
        elif ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            logger.error(f"不支持的文件格式: {ext}")
            return False

        documents = loader.load()
        split_docs = _get_text_splitter().split_documents(documents)

        vector_store = get_vector_store()
        vector_store.add_documents(split_docs)

        _save_ingested_hash(fhash)
        logger.info(f"《{os.path.basename(file_path)}》已成功注入！生成 {len(split_docs)} 个向量块。")
        return True

    except Exception as e:
        logger.error(f"录入失败: {str(e)}")
        return False


if __name__ == "__main__":
    print("=== 向量数据库测试 ===")

    test_query = "适合阴雨天喝的汤"
    print(f"搜索: '{test_query}'")
    docs = search_knowledge(test_query, k=1)
    if docs:
        print(f"检索到 (来源: {docs[0].metadata.get('source')}):")
        print(docs[0].page_content)
    else:
        print("未检索到相关内容。")
