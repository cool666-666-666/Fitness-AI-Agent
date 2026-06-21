"""添加训练记录工具 — 接收结构化数据直接写入 MySQL。"""

from __future__ import annotations

import json
import uuid
from datetime import date

from langchain_core.tools import tool
from loguru import logger
from sqlalchemy import or_, select

from agent.db.models import ExerciseCatalog, User, WorkoutItem, WorkoutSession
from agent.db.mysql import async_session_factory
from agent.parser.workout_parser import build_workout


def _looks_like_uuid(s: str) -> bool:
    """判断字符串是否类似 UUID 格式。"""
    try:
        uuid.UUID(s)
        return True
    except (ValueError, AttributeError):
        return False


@tool
async def add_workout_tool(
    user_id: str,
    exercises: list[dict],
) -> str:
    """记录一次训练，接收结构化动作数据直接写入数据库。

    由前端侧边栏直接调用，也可由 Agent 调用。
    每个 dict 需包含:
        - name (str): 动作名称
        - sets (int): 组数
        - reps (int): 每组次数
        - weight_kg (float, 可选): 重量（公斤），0 表示自重

    Args:
        user_id: 用户标识符（UUID 字符串或用户名）
        exercises: 结构化训练动作列表

    Returns:
        包含训练记录确认信息的 JSON 字符串
    """
    # 确保 Agent 调用时使用正确的 user_id
    from agent.tools import current_user_id  # noqa: PLC0415

    enforced = current_user_id.get()
    if enforced:
        user_id = enforced

    # 将 user_id 转为 UUID
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}@fitness-agent")

    username = user_id if not _looks_like_uuid(user_id) else f"user-{user_uuid.hex[:8]}"

    # 从结构化数据构建 ParsedWorkout
    try:
        parsed = build_workout(exercises)
    except (ValueError, KeyError) as e:
        return json.dumps(
            {"error": f"训练数据格式错误: {e}"},
            ensure_ascii=False,
        )

    # 写入数据库
    async with async_session_factory() as session:
        # 确保用户存在
        user = await session.get(User, user_uuid)
        if user is None:
            user = User(id=user_uuid, username=username)
            session.add(user)
            await session.flush()
            logger.info(f"自动创建用户: {username} ({user_uuid})")

        # 创建训练记录
        workout = WorkoutSession(
            user_id=user_uuid,
            session_date=date.today(),
            notes=None,
        )
        session.add(workout)
        await session.flush()

        # 逐项写入训练动作
        items_saved = []
        for ex in parsed.exercises:
            # 查找动作目录匹配项
            catalog_entry = None
            if ex.name:
                result = await session.execute(
                    select(ExerciseCatalog).where(
                        or_(
                            ExerciseCatalog.name.ilike(f"%{ex.name}%"),
                            ExerciseCatalog.alias.contains(ex.name),
                        )
                    )
                )
                catalog_entry = result.scalars().first()

            item = WorkoutItem(
                session_id=workout.id,
                exercise_name=ex.name,
                exercise_catalog_id=catalog_entry.id if catalog_entry else None,
                sets=ex.sets,
                reps=ex.reps if ex.reps and ex.reps > 0 else None,
                weight_kg=ex.weight_kg,
                set_details=[s.model_dump() for s in ex.set_details] if ex.set_details else None,
                duration_seconds=None,
                distance_meters=None,
                notes=ex.notes,
            )
            session.add(item)
            items_saved.append({
                "exercise": ex.name,
                "sets": ex.sets,
                "reps": ex.reps,
                "weight_kg": ex.weight_kg,
                "set_details": [s.model_dump() for s in ex.set_details] if ex.set_details else None,
            })

        await session.commit()

        result = {
            "status": "success",
            "session_id": str(workout.id),
            "date": str(workout.session_date),
            "exercises": items_saved,
        }
        logger.info(f"已保存训练记录 {workout.id}，共 {len(items_saved)} 项动作")
        return json.dumps(result, ensure_ascii=False, indent=2)
