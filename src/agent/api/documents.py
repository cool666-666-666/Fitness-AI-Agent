"""文档上传接口 —— 健身知识库文档摄入，自动嵌入并存储到 Qdrant。"""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from loguru import logger

from agent.backend.services.embedding_management import EmbeddingManagement
from agent.utils.utility import create_tmp_folder

router = APIRouter()

# 支持上传的文件格式（PDF / 纯文本 / Markdown）
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


class DocumentUploadResponse:
    """文档上传接口的响应模型。"""

    def __init__(self, status: str, files: list[dict]):
        self.status = status     # 上传状态：success / partial
        self.files = files       # 每个文件的处理结果列表


async def _write_and_get_path(file: UploadFile, tmp_dir: str) -> Path:
    """将上传文件写入磁盘临时目录，返回文件路径。

    用于在嵌入处理前暂存文件，处理完成后整个 tmp_dir 会被清理。
    """
    # 读取上传文件的全部内容（小文件，一次性读入内存）
    content = await file.read()
    # 若文件名为空，生成一个随机名防止冲突
    filename = file.filename or f"upload_{uuid.uuid4().hex[:8]}"
    file_path = Path(tmp_dir) / filename
    # 异步写入磁盘
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)
    return file_path


@router.post(
    "/upload",
    tags=["documents"],
    summary="Upload fitness knowledge documents",
)
async def upload_documents(
    files: Annotated[list[UploadFile], File(description="PDF, TXT, or Markdown files")],
    collection_name: str = "fitness_kb",
    category: str = "general",
) -> dict:
    """上传健身相关的文档到知识库。

    支持的格式：PDF、TXT、Markdown (.md)。

    每个文档的处理流程：
    1. 暂存到临时目录
    2. 切分为文本块（chunk）
    3. 调用嵌入模型生成向量，存入 Qdrant，并附带元数据：
       - filename（文件名）、source（来源）、upload_time（上传时间）、category（分类）

    Args:
        files: 要上传的文件列表。
        collection_name: 目标 Qdrant 集合名称。
        category: 内容分类（strength / nutrition / recovery / supplement / general）。
    """
    # 无文件上传时直接返回 400
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未上传任何文件。",
        )

    # 创建临时目录，用于暂存上传文件（处理完成后自动清理）
    tmp_dir = create_tmp_folder()
    # 统一的上传时间戳，同一批次所有文件共享
    upload_time = datetime.now(timezone.utc).isoformat()
    results = []

    for file in files:
        # 跳过空文件名的上传项
        if not file.filename:
            continue

        # 校验文件扩展名，不支持的类型直接跳过并记录原因
        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "skipped",
                "reason": f"不支持的格式: {ext}。支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}",
            })
            continue

        try:
            # 写入磁盘 → 嵌入 → 存入 Qdrant
            file_path = await _write_and_get_path(file, tmp_dir)
            logger.info(f"处理中: {file.filename} (category={category})")

            service = EmbeddingManagement(collection_name=collection_name)
            # embedding 操作可能涉及网络 I/O，放到线程池中执行避免阻塞事件循环
            await asyncio.to_thread(
                service.embed_documents,
                directory=str(tmp_dir),
                file_ending=ext,
            )

            results.append({
                "filename": file.filename,
                "status": "success",
                "category": category,
                "upload_time": upload_time,
            })
        except Exception as e:
            logger.error(f"嵌入失败 {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "reason": str(e),
            })

    # 全部成功返回 "success"，部分失败返回 "partial"
    return {
        "status": "success" if all(r["status"] == "success" for r in results) else "partial",
        "collection": collection_name,
        "files": results,
    }
