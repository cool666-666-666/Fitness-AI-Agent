"""RAG 搜索工具 —— 仅负责检索 + 重排序，不生成最终回答。

从 Qdrant 向量库中检索相关文档片段，由 Fitness Agent 的 LLM
负责基于这些片段合成最终答案。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool
from loguru import logger

from agent.utils.config import config
from agent.utils.retriever import get_retriever


@tool
async def rag_search_tool(
    query: str,
    collection_name: str = "fitness_kb",
    top_k: int = 5,
) -> str:
    """搜索健身知识库，获取相关信息。

    当用户询问以下内容时使用此工具：
    - 运动理论、正确姿势或技巧
    - 训练计划、周期化训练或编程
    - 营养、饮食或补剂
    - 恢复、伤病预防或活动度
    - 运动科学或运动生理学

    此工具返回相关文档片段。请勿用它来记录训练或查询个人训练历史，
    这些需求应使用 add_workout_tool 或 query_workout_tool。

    Args:
        query: 要搜索的具体问题。
        collection_name: Qdrant 集合名称（默认 "fitness_kb"）。
        top_k: 返回的文档片段数量（默认 5）。

    Returns:
        包含相关文档片段及元数据的 JSON 字符串。
    """
    try:
        # 粗筛阶段：多拉候选文档（默认 40 条），充分发挥混合检索的召回优势
        # 稠密向量覆盖语义相关，BM25 覆盖关键词匹配，两者互补
        retriever = get_retriever(k=config.retrieval_k, collection_name=collection_name)
        docs = await retriever.ainvoke(query)

        # 精排阶段：用 FlashRank 对粗筛结果重排序，截断到 top_k 条
        if config.rerank_provider != "none":
            from agent.utils.reranker import get_reranker
            reranker = get_reranker(
                provider=config.rerank_provider,
                top_k=min(top_k, len(docs)),
            )
            docs = reranker(docs, query)
        else:
            # 未启用重排序时直接截断
            docs = docs[:top_k]

        # 格式化结果 —— 返回内容 + 元数据，不生成最终回答
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            category = doc.metadata.get("category", "general")
            results.append({
                "index": i + 1,
                "content": doc.page_content,
                "source": source,
                "category": category,
            })

        logger.info(f"RAG search returned {len(results)} chunks for query: {query[:60]}")
        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return json.dumps({"error": f"Search failed: {e}"}, ensure_ascii=False)
