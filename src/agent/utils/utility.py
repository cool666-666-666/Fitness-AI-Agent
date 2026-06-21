"""
工具函数模块 —— 提供通用的辅助函数，包括文本处理、提示模板加载、
向量数据库结果转换、临时目录创建以及文档格式化等。

本模块中的函数均为纯工具函数，不依赖模块内部状态，可独立调用。
"""

import uuid
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from loguru import logger

from agent.data_model.internal_model import RetrievalResults


def combine_text_from_list(input_list: list) -> str:
    """将字符串列表合并为一个字符串。

    遍历列表中的每个元素，验证其均为字符串类型后，
    用换行符（\\n）连接成一个完整的字符串。

    Args:
        input_list: 由字符串组成的列表。

    Raises:
        TypeError: 如果列表中包含非字符串类型的元素。

    Returns:
        合并后的字符串，每个元素之间以换行符分隔。

    示例:
        >>> combine_text_from_list(["你好", "世界"])
        '你好\\n世界'
    """
    logger.info(f"List: {input_list}")

    for text in input_list:
        if not isinstance(text, str):
            msg = "Input list must contain only strings"
            raise TypeError(msg)

    return "\n".join(input_list)


def load_prompt_template(prompt_name: str, task: str) -> PromptTemplate:
    """加载指定任务的提示模板文件。

    从 'prompts/<task>/<prompt_name>' 路径读取模板文件，
    并返回 LangChain 的 PromptTemplate 对象。

    Args:
        prompt_name: 提示模板的文件名。
        task: 任务类别名称（如 "chat"），对应 prompts 目录下的子目录名。

    Raises:
        FileNotFoundError: 如果指定的模板文件不存在。

    Returns:
        加载完成的 PromptTemplate 对象，可直接用于格式化提示。
    """
    try:
        with Path(Path("prompts") / task / prompt_name).open(encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError as e:
        msg = f"Prompt file '{prompt_name}' not found."
        raise FileNotFoundError(msg) from e

    return PromptTemplate.from_template(prompt_template)


def convert_qdrant_result_to_retrieval_results(docs: list) -> list[RetrievalResults]:
    """将 Qdrant 原始查询结果转换为内部模型 RetrievalResults 列表。

    Qdrant 的相似度搜索返回包含 (Document, score) 元组的列表，
    此函数将其映射为项目中统一使用的 RetrievalResults 数据模型。

    Args:
        docs: Qdrant 返回的原始结果列表，
              每个元素为 (Document, score) 元组。

    Returns:
        RetrievalResults 对象列表，每个包含文档内容、相似度分数和元数据。
    """
    return [
        RetrievalResults(document=doc[0].page_content, score=doc[1], metadata=doc[0].metadata)
        for doc in docs
    ]


def create_tmp_folder() -> str:
    """创建一个临时文件夹用于存放临时文件。

    在当前工作目录下创建以 'tmp_<UUID>' 命名的临时文件夹，
    每次调用生成唯一名称，避免命名冲突。

    Returns:
        创建的临时文件夹的绝对路径字符串。

    Raises:
        ValueError: 如果文件夹创建失败。
    """
    tmp_dir = Path.cwd() / f"tmp_{uuid.uuid4()}"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new folder {tmp_dir}.")
    except ValueError as e:
        logger.error(f"Failed to create directory {tmp_dir}. Error: {e}")
        raise
    return str(tmp_dir)


def format_docs_for_citations(docs: Sequence[Document]) -> str:
    """将文档列表格式化为带引用的 XML 标记格式。

    为每个文档添加 <doc id='索引'> 标签，使其在生成回答时
    可以引用具体的文档来源（citation）。

    Args:
        docs: 从向量数据库检索到的 LangChain Document 序列。

    Returns:
        格式化后的字符串，每个文档被 <doc> 标签包裹，索引从 0 开始编号。

    示例输出:
        <doc id='0'>文档内容...</doc>
        <doc id='1'>文档内容...</doc>
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)
