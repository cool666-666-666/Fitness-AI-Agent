"""Fitness Agent —— 基于 LangGraph 的 ReAct 智能体，配备健身领域工具集。

构建一个工具调用型 Agent，融合三种能力：
- 个人训练记录（通过工具操作 MySQL）
- 健身知识库（通过 rag_search_tool 检索 Qdrant 向量库）
- LLM 推理（DeepSeek / LiteLLM 作为大脑）
"""

from __future__ import annotations

from langchain_litellm import ChatLiteLLM
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from loguru import logger

from agent.backend.prompts import get_system_prompt
from agent.tools import analyze_progress_tool, query_workout_tool, rag_search_tool
from agent.utils.config import Config


def _create_llm(cfg: Config) -> ChatOpenAI | ChatLiteLLM:
    """创建 LLM 实例，沿用现有的多模型切换模式。

    优先使用 DeepSeek（通过 OpenAI 兼容接口），
    未配置 DeepSeek API Key 时回退到 LiteLLM 统一网关。
    两种模式均开启流式输出。
    """
    if cfg.deepseek_api_key:
        # DeepSeek 模式：使用 ChatOpenAI 客户端 + 自定义 base_url
        return ChatOpenAI(
            model=cfg.deepseek_model_name,
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_api_base,
            streaming=True,
        )
    # LiteLLM 回退模式：支持多种模型供应商的统一网关
    return ChatLiteLLM(model_name=cfg.model_name, streaming=True)


def build_fitness_agent(cfg: Config | None = None) -> object:
    """构建并返回 Fitness AI ReAct 智能体。

    Agent 会根据用户意图自动选择合适的工具：
    - query_workout_tool —— 查询训练历史
    - analyze_progress_tool —— 计算训练进展统计
    - rag_search_tool —— 搜索健身知识库

    注意：add_workout_tool 不在此列表中，因为它由前端侧边栏直接调用，
    不经过 Agent 推理链路。

    Args:
        cfg: 应用配置，为 None 时使用全局 Config 单例。

    Returns:
        编译好的 LangGraph ReAct Agent（Runnable 对象）。
    """
    if cfg is None:
        cfg = Config()

    # 创建 LLM 推理引擎
    llm = _create_llm(cfg)

    # Agent 可调用的工具列表
    tools = [
        query_workout_tool,
        analyze_progress_tool,
        rag_search_tool,
    ]

    # 内存检查点：保存对话历史，支持多轮对话的上下文记忆
    memory = MemorySaver()

    # 创建 ReAct Agent：
    # - system_prompt 定义 Agent 的行为人设和规则
    # - checkpointer 负责在多轮对话间持久化状态
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=get_system_prompt(),
        checkpointer=memory,
    )

    logger.info(
        f"Fitness Agent built with {len(tools)} tools, "
        f"model={cfg.deepseek_model_name if cfg.deepseek_api_key else cfg.model_name}"
    )
    return agent


# 模块级 Agent 单例 —— 在 import 时构建一次，整个应用生命周期内复用
# 避免每次请求都重建 LLM 连接和加载工具定义
fitness_agent = build_fitness_agent()
