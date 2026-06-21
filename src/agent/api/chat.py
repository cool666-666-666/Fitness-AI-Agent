"""健身 AI 助手聊天接口。

提供两个端点：
- POST /chat        : 非流式问答，等待完整结果后返回。
- POST /chat/stream : 流式问答，实时推送工具调用状态和生成内容。
"""

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from agent.agent.fitness_agent import fitness_agent
from agent.tools import current_user_id

router = APIRouter()


class ChatMessage(BaseModel):
    """单条对话消息，对应 OpenAI 消息格式的子集。"""
    role: str = Field(description="消息角色：'user' 或 'assistant'")
    content: str = Field(description="消息文本内容")


class ChatRequest(BaseModel):
    """聊天请求体，包含对话历史和用户标识。"""
    messages: list[ChatMessage] = Field(description="对话历史（含当前用户消息）")
    user_id: str = Field(description="用户 UUID，用于数据隔离和鉴权")
    collection_name: str = Field(default="fitness_kb", description="RAG 检索用的 Qdrant 集合名称")


class ChatResponse(BaseModel):
    """非流式聊天的响应体。"""
    answer: str = Field(description="Agent 的最终回答文本")


@router.post("", tags=["chat"], summary="Chat with the Fitness AI Agent")
async def chat(request: ChatRequest) -> ChatResponse:
    """向 Fitness AI Agent 发送消息（非流式）。

    Agent 自动决定调用哪些工具：
    - 记录训练（add_workout_tool）
    - 查询训练历史（query_workout_tool）
    - 分析进展（analyze_progress_tool）
    - 搜索健身知识（rag_search_tool）

    所有工具调用均使用请求中提供的 user_id 进行数据隔离。
    """
    # 构建 LangChain 消息列表
    langchain_messages: list[AnyMessage] = []
    user_messages_count = sum(1 for m in request.messages if m.role == "user")

    # 仅在首轮对话时注入 SystemMessage（含 user_id），
    # 后续轮次 MemorySaver 已记住，无需重复注入。
    if user_messages_count <= 1:
        langchain_messages.append(
            SystemMessage(
                content=f"当前用户的 user_id 是: {request.user_id}。"
                f"调用任何工具时都必须使用这个 user_id。"
            )
        )

    # 将前端消息转换为 LangChain 标准消息格式
    for msg in request.messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))

    # LangGraph 配置：thread_id = user_id，实现按用户隔离的对话记忆
    config = {
        "configurable": {
            "thread_id": request.user_id,
        }
    }

    logger.info(f"Chat request: user={request.user_id}, messages={len(langchain_messages)}")

    # 设置上下文变量，强制本次调用内所有工具使用正确的 user_id。
    # 这是防止 LLM 幻觉出其他用户 ID 的安全机制。
    token = current_user_id.set(request.user_id)
    try:
        result = await fitness_agent.ainvoke(
            {"messages": langchain_messages},
            config=config,
        )
    finally:
        # 无论成功或异常，调用结束后必须重置上下文变量
        current_user_id.reset(token)

    # 提取最后一条 AI 消息作为回答
    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    logger.info(f"Chat response: {answer[:100]}...")
    return ChatResponse(answer=answer)


@router.post("/stream", tags=["chat"], summary="流式问答：实时推送工具调用状态和生成内容")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """发送消息给健身 AI 助手，通过 NDJSON 流式返回进度和回答。

    每行一个 JSON 对象，type 字段区分消息类型：
    - "status"   : 阶段状态（开始分析 → 工具调用中 → 生成回答 → 完成）
    - "content"  : 逐 token 流式返回的回答文本（前端可逐字渲染）
    - "tool"     : 工具调用详情（工具名、参数、结果预览）

    Parameters
    ----------
    request : ChatRequest
        包含 messages（对话历史）、user_id（用户 ID）、collection_name（知识库集合）。

    Returns
    -------
    StreamingResponse
        media_type 为 application/x-ndjson 的流式响应。
    """
    # 构建 LangChain 消息列表（逻辑与 /chat 端点一致）
    langchain_messages: list[AnyMessage] = []
    user_messages_count = sum(1 for m in request.messages if m.role == "user")
    if user_messages_count <= 1:
        langchain_messages.append(
            SystemMessage(
                content=f"当前用户的 user_id 是: {request.user_id}。"
                f"调用任何工具时都必须使用这个 user_id。"
            )
        )
    for msg in request.messages:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))

    # LangGraph 配置：thread_id = user_id，实现按用户隔离的对话记忆
    config = {
        "configurable": {
            "thread_id": request.user_id,
        }
    }

    async def stream() -> AsyncGenerator:
        """内部异步生成器，逐事件产出 NDJSON 行。

        利用 LangGraph 的 astream_events(v2) 订阅 Agent 内部运行事件，
        将每个事件实时序列化为 NDJSON 行推送给前端。
        """
        # 阶段 1: 分析阶段开始
        yield json.dumps({"type": "status", "data": "正在分析问题..."}, ensure_ascii=False) + "\n"

        # 设置上下文变量，强制工具使用正确的 user_id
        token = current_user_id.set(request.user_id)
        try:
            async for event in fitness_agent.astream_events(
                {"messages": langchain_messages},
                config=config,
                version="v2",  # v2 版本事件更详细，支持 on_tool_start/end 等细粒度事件
            ):
                kind = event["event"]
                name = event.get("name", "")

                # ---- 工具调用开始 ----
                if kind == "on_tool_start":
                    tool_name = name
                    tool_input = event["data"].get("input", {})
                    # 裁剪过长的输入（如 5000 字的查询），避免流式传输膨胀
                    safe_input = {k: (v[:200] + "..." if isinstance(v, str) and len(v) > 200 else v) for k, v in tool_input.items()}
                    yield json.dumps(
                        {"type": "tool", "data": {"tool": tool_name, "input": safe_input, "status": "start"}},
                        ensure_ascii=False,
                    ) + "\n"

                # ---- 工具调用完成 ----
                elif kind == "on_tool_end":
                    tool_output = event["data"].get("output", "")
                    # 只传递结果的前 100 个字符作为预览，完整结果由 LLM 处理后输出
                    preview = tool_output[:100] + "..." if isinstance(tool_output, str) and len(tool_output) > 100 else str(tool_output)[:100]
                    yield json.dumps(
                        {"type": "tool", "data": {"tool": name, "output_preview": preview, "status": "end"}},
                        ensure_ascii=False,
                    ) + "\n"

                # ---- LLM 开始生成回答文本 ----
                elif kind == "on_chat_model_start":
                    yield json.dumps({"type": "status", "data": "正在生成回答..."}, ensure_ascii=False) + "\n"

                # ---- 逐 token 流式输出回答内容 ----
                elif kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield json.dumps({"type": "content", "data": content}, ensure_ascii=False) + "\n"

        finally:
            # 无论成功或异常，调用结束后必须重置上下文变量
            current_user_id.reset(token)

        # 阶段 5: 流结束信号
        yield json.dumps({"type": "status", "data": "完成"}, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
