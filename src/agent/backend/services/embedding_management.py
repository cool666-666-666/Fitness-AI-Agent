"""文档嵌入管理服务。

负责文档嵌入到向量数据库的完整流程：
从磁盘加载文档 → 分割为文本块 → 生成嵌入向量 → 存入 Qdrant 向量数据库。
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFium2Loader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from agent.utils.config import config
from agent.utils.embeddings import get_embedding_model
from agent.utils.vdb import generate_collection, init_vdb, initialize_vector_db

load_dotenv()


class EmbeddingManagement:
    """管理文档嵌入和向量数据库操作。

    职责包括：
    - 连接到 Qdrant 向量数据库并指定集合（collection）。
    - 从目录加载文档并分割为文本块（支持 .pdf / .txt / .md 格式）。
    - 批量生成嵌入向量并存入向量数据库。
    - 在向量数据库中创建新的集合。

    Attributes
    ----------
    cfg : Config
        应用配置对象，包含 embedding 模型、批处理大小等参数。
    collection_name : str
        要使用的 Qdrant 集合名称，由构造函数参数传入。
    vector_db : Qdrant
        Qdrant 向量存储客户端，用于添加文档嵌入。
    """

    def __init__(self, collection_name: str | None) -> None:
        """初始化嵌入管理服务。

        完成以下步骤：
        1. 加载应用配置。
        2. 确保目标 Qdrant 集合已存在。
        3. 初始化嵌入模型。
        4. 连接到向量数据库。

        Parameters
        ----------
        collection_name : str or None
            Qdrant 集合名称。若为 None，则从应用配置中读取。
        """
        # 应用配置（embedding 模型、批处理大小等）
        self.cfg = config

        if collection_name:
            self.collection_name = collection_name

        # 确保集合存在后再连接
        initialize_vector_db(collection_name=self.collection_name, embeddings_size=self.cfg.embedding_size)

        # 加载 embedding 模型（LiteLLM / OpenAI 兼容接口）
        embedding = get_embedding_model(self.cfg)

        # 初始化当前集合的 Qdrant 向量存储客户端
        self.vector_db = init_vdb(collection_name=self.collection_name, embedding=embedding)

    def embed_documents(self, directory: str, file_ending: str = ".pdf") -> None:
        """从目录加载文档，分割为文本块，生成嵌入并存入向量数据库。

        支持 PDF（.pdf）、纯文本（.txt）和 Markdown（.md）格式。
        使用 RecursiveCharacterTextSplitter 按配置的 chunk 大小和重叠量进行分割。
        元数据中的 source 路径会被截取为仅保留文件名。
        嵌入向量通过分批的方式写入，避免对数据库造成过大压力。

        Parameters
        ----------
        directory : str
            待嵌入文档所在的目录路径。
        file_ending : str, default ".pdf"
            文件扩展名过滤。支持的值：".pdf"、".txt"、".md"。

        Raises
        ------
        ValueError
            当传入不支持的文件扩展名时抛出。
        """
        # TODO: 后续可重构为使用 markdownit 进行解析
        # 根据文件类型选择合适的加载器
        if file_ending == ".pdf":
            loader = DirectoryLoader(directory, glob="*" + file_ending, loader_cls=PyPDFium2Loader)
        elif file_ending in (".txt", ".md"):
            loader = DirectoryLoader(directory, glob="*" + file_ending, loader_cls=TextLoader, loader_kwargs={"autodetect_encoding": True})
        else:
            msg = f"File ending '{file_ending}' not supported."
            raise ValueError(msg)

        # 使用递归文本分割器，chunk_size=750, overlap=200，
        # 优先在段落、换行、句号和感叹号处分隔，保证语义完整性
        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=200, length_function=len, separators=["\n\n", "\n", ".", "!"])

        docs = loader.load_and_split(splitter)

        logger.info(f"Loaded {len(docs)} documents.")
        # 提取每个文本块的正文和元数据
        text_list = [doc.page_content for doc in docs]
        metadata_list = [doc.metadata for doc in docs]

        # 规范化元数据：仅保留文件名，去除目录路径
        for m in metadata_list:
            if "/" in m["source"]:
                m["source"] = m["source"].split("/")[-1]

        # 按批次插入向量数据库（若 batch_size <= 0 则一次性全量写入）
        batch_size = self.cfg.embedding_batch_size
        if batch_size > 0:
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i : i + batch_size]
                batch_metadata = metadata_list[i : i + batch_size]
                self.vector_db.add_texts(texts=batch_texts, metadatas=batch_metadata)
                logger.info(f"Embedded batch {i // batch_size + 1}/{(len(text_list) + batch_size - 1) // batch_size} ({len(batch_texts)} texts)")
        else:
            self.vector_db.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Texts embedded.")