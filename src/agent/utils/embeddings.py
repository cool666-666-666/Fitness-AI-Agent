"""Embedding 模型工具模块。

该模块根据配置文件(Config)中指定的 embedding_provider，
动态创建并返回对应的 Embedding 模型实例。

支持：
- Google Gemini
- OpenAI
- 通义千问（Qwen）
- DashScope
"""

# LangChain 中所有 Embedding 模型的统一父类
from langchain_core.embeddings import Embeddings

# 项目的配置类，保存各种 API Key、模型名称等配置
from agent.utils.config import Config


def get_embedding_model(cfg: Config) -> Embeddings:
    """
    根据配置返回对应的 Embedding 模型对象。

    参数：
        cfg: 项目配置对象(Config)

    返回：
        一个实现了 LangChain Embeddings 接口的实例。

    例如：
        GoogleGenerativeAIEmbeddings
        OpenAIEmbeddings
    """

    # 获取当前配置的 Embedding 提供商
    # 例如："openai"、"google"
    provider = cfg.embedding_provider

    # 获取具体使用的 Embedding 模型名称
    # 例如：
    # text-embedding-3-small
    # text-embedding-3-large
    model_name = cfg.embedding_model_name

    # Python 3.10+ 的 match-case，相当于 switch-case
    match provider:

        # ==============================
        # Google Gemini Embedding
        # ==============================
        case "google":

            try:
                from langchain_google_genai import (
                    GoogleGenerativeAIEmbeddings,
                )  # noqa: PLC0415

            # 如果没有安装依赖，则提示用户安装
            except ImportError as exc:
                msg = "langchain-google-genai is required for Google embeddings."
                raise ImportError(msg) from exc

            # 创建 Google Embedding 模型
            return GoogleGenerativeAIEmbeddings(
                model=model_name,

                # Gemini API Key
                google_api_key=cfg.gemini_api_key or None,

                # 输出向量维度
                output_dimensionality=cfg.embedding_size,
            )

        # ==============================
        # OpenAI Embedding
        # ==============================
        case "openai":

            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415

            # 创建 OpenAI Embedding 模型
            return OpenAIEmbeddings(
                model=model_name,

                # OpenAI API Key
                api_key=cfg.openai_api_key or None,
            )

        # ==============================
        # 通义千问 / DashScope Embedding
        # ==============================
        #
        # 阿里云兼容 OpenAI API，
        # 所以直接复用 OpenAIEmbeddings。
        #
        case "qwen" | "dashscope":

            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415

            return OpenAIEmbeddings(
                model=model_name,

                # 优先使用 qwen_api_key
                # 没有则使用 dashscope_api_key
                api_key=cfg.qwen_api_key or cfg.dashscope_api_key or None,

                # 指定兼容 OpenAI 的 Base URL
                base_url=(
                    cfg.qwen_embedding_api_base
                    or cfg.dashscope_embedding_api_base
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),

                # 指定输出向量维度
                dimensions=cfg.embedding_size,

                # 不检查文本长度限制
                check_embedding_ctx_length=False,
            )

        # ==============================
        # 未配置支持的 Provider
        # ==============================
        case _:

            msg = "No suitable embedding Model configured!"

            raise KeyError(msg)