# 导入 Pydantic 字段定义工具
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Loading the settings with pydantic."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    openai_api_type: str = "openai"
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    openai_api_key: str = ""
    openai_api_version: str = ""
    gemini_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_model_name: str = "deepseek-chat"
    qwen_api_key: str = ""
    dashscope_api_key: str = ""
    qwen_embedding_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_embedding_api_base: str = ""

    # Model Configuration
    model_name: str = "gemini/gemini-2.5-flash"
    embedding_provider: str = "google"
    embedding_model_name: str = "gemini-embedding-002"
    embedding_size: int = 768
    embedding_batch_size: int = 0  # 0 = no batching; set to 10 for Qwen/Dashscope limit

    # Database
    database_url: str = "mysql+aiomysql://root:root@localhost:3306/fitness_agent"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Fitness Agent
    fitness_collection_name: str = "fitness_kb"

    # Reranker Configuration
    rerank_provider: str = "flashrank"  # "flashrank" or "none"
    rerank_top_k: int = 5

    # Retrieval Configuration
    retrieval_k: int = 40  # Number of documents to retrieve initially
    retrieval_k_retry: int = 100  # Number of documents to retrieve on retry

    # QDRANT
    qdrant_url: str = "http://localhost"
    qdrant_api_key: str | None = Field(default=None, validation_alias=AliasChoices("qdrant_api_key", "qdrant_cloud_api_key"))
    qdrant_port: int = 6333
    qdrant_prefer_grpc: bool = False
    phoenix_collector_endpoint: str = "http://phoenix:4318/v1/traces"
    qdrant_collection_name: str = "default"


config = Config()
