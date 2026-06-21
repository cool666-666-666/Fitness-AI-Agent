"""Fitness AI Agent 前端 API 客户端 —— 封装对后端所有接口的 HTTP 调用。

负责三个模块的通信：
- Chat  : 流式聊天（NDJSON）和非流式对话
- Workout: 训练记录的增删改查
- Documents: 知识库文档上传

注意：本模块由 Streamlit 前端（app.py）调用，运行在独立的 Docker 容器中，
通过 BACKEND_HOST 环境变量指向 agent 后端服务。
"""

import os
from collections.abc import AsyncGenerator

import httpx


class AgentClient:
    """Fitness AI Agent 后端 API 的 HTTP 客户端。

    封装了与后端 FastAPI 服务的通信逻辑：
    - base_url 自动解析（Docker 环境用服务名，本地开发用 localhost）
    - 针对不同接口设置合理的超时时间
    - 流式接口返回异步生成器，非流式接口返回 dict
    """

    def __init__(self, base_url: str | None = None) -> None:
        """初始化客户端，确定后端服务地址。

        base_url 的优先级：
        1. 显式传入 base_url 参数
        2. 环境变量 BACKEND_HOST + BACKEND_PORT
        3. 默认值 localhost:8001（本地开发）

        Docker 环境中 BACKEND_HOST=agent（服务名即 DNS），
        本地开发时不设置该变量，自动回退到 localhost。
        """
        if base_url:
            # 直接指定（测试或自定义场景）
            self.base_url = base_url
        else:
            # Docker 环境中 BACKEND_HOST=agent，本地默认 localhost
            host = os.getenv("BACKEND_HOST", "localhost")
            port = os.getenv("BACKEND_PORT", "8001")
            self.base_url = f"http://{host}:{port}"

    # ===========================
    #  聊天模块
    # ===========================

    async def chat_stream(
        self, messages: list[dict], user_id: str, collection_name: str = "fitness_kb"
    ) -> AsyncGenerator[str, None]:
        """流式发送消息到健身 AI 助手，实时接收 NDJSON 格式的进度和回答。

        使用 HTTP 流式请求（Server-Sent Events 的变体），每行是一个完整的
        JSON 对象（NDJSON 格式），前端逐行解析后增量渲染到聊天界面。

        NDJSON 消息类型：
        - {"type": "status",  "data": "..."}   → 阶段状态提示
        - {"type": "tool",    "data": {...}}    → 工具调用详情
        - {"type": "content", "data": "..."}    → 逐 token 回答文本

        Args:
            messages: 对话历史列表 [{"role":"user","content":"..."}, ...]
            user_id: 用户唯一标识，用于数据隔离和鉴权
            collection_name: Qdrant 知识库集合名称

        Yields:
            每行 NDJSON 字符串（不含换行符），上游逐行 await 消费
        """
        payload = {
            "messages": messages,
            "user_id": user_id,
            "collection_name": collection_name,
        }
        # timeout=600s：LLM 推理 + 多次工具调用可能耗时较长（极端情况下数分钟）
        async with httpx.AsyncClient(timeout=600.0) as client, client.stream(
            "POST", f"{self.base_url}/chat/stream", json=payload
        ) as response:
            # 若后端返回 4xx/5xx，立即抛异常，阻止生成器继续
            response.raise_for_status()
            # aiter_lines() 逐行读取响应体，匹配后端 NDJSON yield 的分行节奏
            async for line in response.aiter_lines():
                yield line

    # ===========================
    #  文档上传模块
    # ===========================

    async def upload_documents(
        self, files: list[tuple], collection_name: str = "fitness_kb", category: str = "general"
    ) -> dict:
        """上传健身知识文档到知识库。

        调用 POST /documents/upload，后端自动完成：
        1. 校验文件格式（仅 .pdf / .txt / .md）
        2. 写入临时目录
        3. 切分为文本块并生成嵌入向量
        4. 存入 Qdrant 向量库

        Args:
            files: 文件列表，格式为 [("files", (file_name, file_bytes, mime_type)), ...]
            collection_name: 目标 Qdrant 集合名称
            category: 文档分类标签 —— strength / nutrition / recovery / supplement / general

        Returns:
            后端响应的 JSON dict，格式：
            {"status": "success|partial", "collection": "...", "files": [...]}
        """
        params = {"collection_name": collection_name, "category": category}
        # timeout=6000s（100 分钟）：大量大文件的上传 + 嵌入可能非常耗时
        async with httpx.AsyncClient(timeout=6000.0) as client:
            resp = await client.post(
                f"{self.base_url}/documents/upload",
                params=params,
                files=files,
            )
            resp.raise_for_status()
            return resp.json()

    # ===========================
    #  训练记录模块
    # ===========================

    async def add_workout(self, user_id: str, exercises: list[dict]) -> dict:
        """通过结构化数据记录一次训练。

        由前端侧边栏直接调用，不经过 Agent 推理链路（add_workout_tool
        不在 fitness_agent 的工具列表中，见 fitness_agent.py 第 65-69 行）。

        Args:
            user_id: 用户唯一标识
            exercises: 训练动作列表，每项含：
                - exercise_name (str): 动作名称，如 "深蹲"
                - sets (int): 完成组数
                - reps (int): 每组次数
                - weight_kg (float): 使用重量（公斤）
                - set_details (str, 可选): 每组详细记录，如 "10,8,6"
        """
        payload = {"user_id": user_id, "exercises": exercises}
        # timeout=60s：单次写入应很快完成，60s 是充足的保守值
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/workout/add",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()

