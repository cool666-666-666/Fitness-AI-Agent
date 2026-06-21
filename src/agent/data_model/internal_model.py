"""该脚本包含内部数据处理使用的 Pydantic 数据模型。

本模块定义的模型不直接暴露给 API 路由，而是在后端服务、工具函数之间
传递和处理数据时使用，起到统一内部数据结构的作用。
"""

from pydantic import BaseModel


class RetrievalResults(BaseModel):
    """检索结果内部模型。

    将 Qdrant 原始查询结果转换为项目中统一使用的标准格式。
    相比直接传递 LangChain Document 对象，使用该模型可以提供明确的
    类型约束和更好的可读性。
    """

    # 检索到的文档正文内容（即 chunk 的文本）。
    document: str

    # 该文档片段的元数据字典，包含页码、来源文件名等附加信息。
    # 由 Qdrant 中存储的 payload 直接映射而来，键值取决于存入时的定义。
    metadata: dict

    # 该文档与查询之间的语义相似度评分（0~1），数值越高表示越相关。
    # 来自 Qdrant 的向量相似度计算结果。
    score: float
