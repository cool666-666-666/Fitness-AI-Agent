"""Tools 包 —— 为 Fitness AI Agent 提供工具集。

本包包含 Agent 可调用的各类工具：
- add_workout_tool: 添加训练记录
- query_workout_tool: 查询训练记录
- analyze_progress_tool: 分析训练进展
- rag_search_tool: 基于 RAG 的知识检索
"""


import contextvars

from agent.tools.add_workout import add_workout_tool
from agent.tools.analyze_progress import analyze_progress_tool
from agent.tools.query_workout import query_workout_tool
from agent.tools.rag_search import rag_search_tool

# 上下文变量，用于在工具调用中强制使用正确的 user_id。
# 当在 Agent 调用内部设置后，工具必须使用此值，
# 而不是 LLM 可能虚构的任意 user_id。
current_user_id: contextvars.ContextVar[str] = contextvars.ContextVar("current_user_id", default="")

__all__ = [
    "add_workout_tool",
    "query_workout_tool",
    "analyze_progress_tool",
    "rag_search_tool",
    "current_user_id",
]
