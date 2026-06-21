"""
检索器工具模块 —— 提供带缓存的嵌入模型与向量存储的检索器。

本模块通过两级缓存机制（嵌入模型缓存 + 向量存储缓存）避免重复创建
昂贵的嵌入模型和向量数据库连接，提升检索性能。

依赖关系:
    - agent.utils.config: 应用配置（嵌入提供商、模型名、向量维度等）
    - agent.utils.embeddings: 嵌入模型工厂函数
    - agent.utils.vdb: Qdrant 向量数据库客户端与稀疏嵌入
"""

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from agent.utils.config import config
from agent.utils.embeddings import get_embedding_model
from agent.utils.vdb import initialize_vector_db, qdrant_client, sparse_embeddings

# ── 嵌入模型缓存 ──────────────────────────────────────────────────────────────
# _embeddings_cache: dict[tuple[str, str], Embeddings]
#   键 = (embedding_provider, embedding_model_name)，值 = 对应的 Embeddings 实例。
#   确保同一（提供商, 模型名）组合只创建一次嵌入模型。
_embeddings_cache: dict[tuple[str, str], Embeddings] = {}


def _get_cached_embedding() -> Embeddings:
    """
    获取缓存的嵌入模型实例。

    根据全局配置中的 embedding_provider 和 embedding_model_name 构建缓存键，
    若缓存中不存在则调用 get_embedding_model() 创建新实例并缓存。

    Returns:
        Embeddings: 对应配置的嵌入模型实例（如 OpenAI 嵌入模型）。
    """
    key = (config.embedding_provider, config.embedding_model_name)
    if key not in _embeddings_cache:
        _embeddings_cache[key] = get_embedding_model(config)
    return _embeddings_cache[key]


# ── 向量存储缓存 ─────────────────────────────────────────────────────────────
# _vector_store_cache: dict[str, QdrantVectorStore]
#   键 = collection_name（集合名称），值 = 对应的 QdrantVectorStore 实例。
#   确保同一集合只初始化一次 QdrantVectorStore 连接。
_vector_store_cache: dict[str, QdrantVectorStore] = {}


def _get_cached_vector_store(collection_name: str) -> QdrantVectorStore:
    """
    获取缓存的 Qdrant 向量存储实例。

    先检查缓存中是否已存在指定集合的 VectorStore，若不存在则：
        1. 调用 initialize_vector_db() 确保集合在 Qdrant 中已就绪。
        2. 创建 QdrantVectorStore 实例，配置混合检索模式（稠密 + 稀疏向量）。
        3. 将实例存入缓存供后续复用。

    Args:
        collection_name: Qdrant 集合名称，用于区分不同的文档集合。

    Returns:
        QdrantVectorStore: 配置好的向量存储实例，支持混合检索。
    """
    if collection_name not in _vector_store_cache:
        # 连接前确保集合已存在（不存在则自动创建）
        initialize_vector_db(collection_name=collection_name, embeddings_size=config.embedding_size)
        _vector_store_cache[collection_name] = QdrantVectorStore(
            client=qdrant_client,                        # Qdrant 客户端连接
            collection_name=collection_name,              # 目标集合名称
            embedding=_get_cached_embedding(),            # 稠密嵌入模型（用于语义搜索）
            sparse_embedding=sparse_embeddings,           # 稀疏嵌入模型（用于关键词匹配，BM25 风格）
            retrieval_mode=RetrievalMode.HYBRID,          # 检索模式：混合检索（稠密 + 稀疏）
            sparse_vector_name="fast-sparse-bm25",        # Qdrant 中稀疏向量的名称
        )
    return _vector_store_cache[collection_name]


def get_retriever(k: int = 4, collection_name: str = "default") -> BaseRetriever:
    """
    创建基于向量数据库的检索器，使用混合检索策略。

    结合稠密向量语义检索与稀疏向量关键词检索的优势，提供更全面的
    文档召回效果。通过缓存机制复用嵌入模型和向量存储连接。

    Args:
        k: 检索返回的文档数量（Top-K），默认 4 条。
        collection_name: 要检索的 Qdrant 集合名称，默认 "default"。

    Returns:
        BaseRetriever: 配置好的 LangChain 检索器实例，可直接用于文档检索。

    使用示例:
        >>> retriever = get_retriever(k=5, collection_name="knowledge_base")
        >>> docs = retriever.invoke("什么是混合检索？")
    """
    vector_db = _get_cached_vector_store(collection_name)
    return vector_db.as_retriever(search_kwargs={"k": k})
