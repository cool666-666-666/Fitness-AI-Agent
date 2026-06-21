"""Workout API — 直接记录和查询训练数据（绕过 Agent）。"""

import json

from fastapi import APIRouter, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field

from agent.tools.add_workout import add_workout_tool
from agent.tools.query_workout import query_workout_tool

router = APIRouter()


class ExerciseItem(BaseModel):
    """单个训练动作的结构化输入。"""

    name: str = Field(description="动作名称，如 深蹲、卧推", min_length=1)
    sets: int = Field(description="组数", ge=1)
    reps: int = Field(description="总次数", ge=0)
    weight_kg: float = Field(default=0.0, description="平均重量（公斤），0 表示自重", ge=0.0)
    set_details: list[dict] | None = Field(
        default=None,
        description="每组明细 [{\"reps\": 10, \"weight_kg\": 80.0}, ...]，每组次数与重量一一对应",
    )
    notes: str | None = Field(default=None, description="备注")


class WorkoutAddRequest(BaseModel):
    """训练记录请求 — 结构化动作列表。"""

    user_id: str = Field(description="用户 UUID")
    exercises: list[ExerciseItem] = Field(description="训练动作列表", min_length=1)


class WorkoutAddResponse(BaseModel):
    """训练记录响应。"""

    status: str
    session_id: str | None = None
    parsed: list[dict] = Field(default_factory=list)
    message: str = ""


class WorkoutHistoryResponse(BaseModel):
    """训练历史响应。"""

    sessions: list[dict] = Field(default_factory=list)


@router.post("/add", tags=["workout"], summary="记录训练（结构化输入）")
async def add_workout(request: WorkoutAddRequest) -> WorkoutAddResponse:
    """接收结构化训练数据并写入数据库。

    前端通过侧边栏的 4 个输入框录入：
    - 动作名称（如"深蹲"）
    - 组数
    - 每组次数
    - 重量（kg，可选）
    """
    # 将 Pydantic 模型转为 dict 列表传给 tool
    exercises = [
        {
            "name": ex.name,
            "sets": ex.sets,
            "reps": ex.reps,
            "weight_kg": ex.weight_kg if ex.weight_kg > 0 else None,
            "set_details": ex.set_details,
            "notes": ex.notes,
        }
        for ex in request.exercises
    ]

    result_str = await add_workout_tool.ainvoke({
        "user_id": request.user_id,
        "exercises": exercises,
    })

    try:
        result = json.loads(result_str)
    except json.JSONDecodeError:
        logger.error(f"Tool 返回非法 JSON: {result_str[:200]}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="内部错误：无法解析训练结果。",
        ) from None

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "未知错误"),
        )

    return WorkoutAddResponse(
        status=result.get("status", "unknown"),
        session_id=result.get("session_id"),
        parsed=result.get("exercises", []),
        message=f"已记录 {len(result.get('exercises', []))} 项训练",
    )


@router.get("/history", tags=["workout"], summary="查询训练历史")
async def get_workout_history(
    user_id: str = Query(description="用户 UUID"),
    exercise_name: str | None = Query(default=None, description="按动作名称筛选"),
    days_back: int = Query(default=30, ge=1, le=365, description="回溯天数"),
    limit: int = Query(default=20, ge=1, le=50, description="最大返回条数"),
) -> WorkoutHistoryResponse:
    """从数据库查询用户的训练历史。"""
    result_str = await query_workout_tool.ainvoke({
        "user_id": user_id,
        "exercise_name": exercise_name,
        "days_back": days_back,
        "limit": limit,
    })

    try:
        sessions = json.loads(result_str)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法解析训练历史数据",
        ) from None

    return WorkoutHistoryResponse(sessions=sessions if isinstance(sessions, list) else [])
