"""查询训练记录工具 —— 从 MySQL 检索训练历史。"""

from __future__ import annotations

import json
import uuid
from datetime import date, timedelta

from langchain_core.tools import tool
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from agent.db.models import User, WorkoutItem, WorkoutSession
from agent.db.mysql import async_session_factory


def _resolve_user_id(user_id: str) -> uuid.UUID:
    """将 user_id 字符串转换为确定性的 UUID。

    支持两种输入形式：
    - 标准 UUID 字符串：直接解析返回
    - 普通用户名：通过 uuid5 确定性生成，同一用户名每次生成相同 UUID
    """
    # 情况 1：输入已是 UUID 格式（如前端传来的标准 UUID）
    try:
        return uuid.UUID(user_id)
    # 情况 2：输入是普通用户名，用 uuid5 确定性映射为 UUID
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}@fitness-agent")


@tool
async def query_workout_tool(
    user_id: str,
    exercise_name: str | None = None,
    days_back: int = 30,
    limit: int = 20,
) -> str:
    """查询用户的训练历史。

    当用户询问过往训练情况时使用此工具，例如：
    - "我最近练了什么？"
    - "上周深蹲练了多少？"
    - "这个月卧推的记录"

    返回结构化的训练数据，包括动作、组数、次数和重量。

    Args:
        user_id: 用户的唯一标识符（UUID 字符串）。
        exercise_name: 按动作名称筛选（可选，如 "深蹲"）。
        days_back: 回溯多少天（默认 30，上限 365）。
        limit: 最多返回多少条训练记录（默认 20，上限 50）。

    Returns:
        包含训练记录及动作明细的 JSON 字符串。
    """
    # 当在 Agent 内部运行时，强制使用正确的 user_id（防止 LLM 幻觉）
    from agent.tools import current_user_id  # noqa: PLC0415

    enforced = current_user_id.get()
    if enforced:
        user_id = enforced

    user_uuid = _resolve_user_id(user_id)

    # 对参数做上限保护，防止查询范围过大拖垮数据库
    days_back = min(days_back, 365)
    limit = min(limit, 50)
    # 计算查询的起始日期（今天往回推 N 天）
    cutoff = date.today() - timedelta(days=days_back)

    async with async_session_factory() as session:
        # 用户不存在则自动创建
        user = await session.get(User, user_uuid)
        if user is None:
            user = User(id=user_uuid, username=user_id)
            session.add(user)
            await session.flush()
            logger.info(f"Auto-created user: {user_id} ({user_uuid})")

        # 查询训练记录，并预加载关联的动作明细（避免 N+1 查询）
        # selectinload: 用一条独立的 SELECT IN 语句批量加载所有 WorkoutItem，
        # 而不是逐个 session 再发查询（那样 20 条记录会变成 1+20=21 次查询）
        stmt = (
            select(WorkoutSession)
            .where(
                WorkoutSession.user_id == user_uuid,
                WorkoutSession.session_date >= cutoff,  # 只查 cutoff 之后的记录
            )
            .options(selectinload(WorkoutSession.items))  # 预加载关联的 WorkoutItem 列表
            .order_by(WorkoutSession.session_date.desc())  # 按日期降序，最近的在前
            .limit(limit)
        )
        result = await session.execute(stmt)
        sessions = result.scalars().all()

        # 将查询结果组装为 JSON 友好的 dict 列表
        output = []
        for s in sessions:
            items = s.items
            # 如果指定了动作名，对每个 session 内的动作做客户端过滤
            if exercise_name:
                items = [i for i in items if exercise_name.lower() in i.exercise_name.lower()]
                # 该 session 中没有匹配的动作则跳过整条记录
                if not items:
                    continue

            output.append({
                "session_id": str(s.id),
                "date": str(s.session_date),
                "notes": s.notes,
                "items": [
                    {
                        "exercise": i.exercise_name,
                        "sets": i.sets,
                        "reps": i.reps,
                        "weight_kg": i.weight_kg,
                        "set_details": i.set_details,
                    }
                    for i in items
                ],
            })

        logger.info(
            f"Query returned {len(output)} sessions for user {user_id}"
            + (f" (exercise: {exercise_name})" if exercise_name else "")
        )
        return json.dumps(output, ensure_ascii=False, indent=2)
