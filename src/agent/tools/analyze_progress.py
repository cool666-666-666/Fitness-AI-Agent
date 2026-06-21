"""分析训练进展工具 —— 用 Pandas 计算训练趋势。

所有统计均在 Python 中完成，LLM 仅负责解读结果。
"""

from __future__ import annotations

import json
import uuid
from datetime import date, timedelta

from langchain_core.tools import tool
from loguru import logger
from sqlalchemy import or_, select

from agent.db.models import ExerciseCatalog, User, WorkoutItem, WorkoutSession
from agent.db.mysql import async_session_factory


def _resolve_user_id(user_id: str) -> uuid.UUID:
    """将 user_id 字符串转换为确定性的 UUID。

    支持两种输入形式：
    - 标准 UUID 字符串：直接解析返回
    - 普通用户名：通过 uuid5 确定性生成，同一用户名每次生成相同 UUID
    """
    # 情况 1：输入已是 UUID 格式
    try:
        return uuid.UUID(user_id)
    # 情况 2：输入是普通用户名，用 uuid5 确定性映射为 UUID
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}@fitness-agent")


@tool
async def analyze_progress_tool(
    user_id: str,
    exercise_name: str | None = None,
    category: str | None = None,
    days_back: int = 90,
) -> str:
    """分析训练进展 —— 支持两种模式。

    模式 1 — 按 exercise_name（单动作趋势分析）：
        追踪某个具体动作的历史进展。
        返回最大重量趋势、训练量趋势、每周训练频率和整体进步百分比。
        适用场景："深蹲进步了吗？", "卧推最近有没有变强？"

    模式 2 — 按 category（身体部位聚合统计）：
        汇总某个身体部位下所有动作的训练数据。
        返回总训练次数、涉及的动作种类数、总组数/次数/训练量。
        可选类别: 胸, 背, 腿, 肩, 手臂, 核心, 有氧
        适用场景："这个月背部练了多少？", "最近腿的训练量？", "我手臂练得够不够？"

    Args:
        user_id: 用户的唯一标识符（UUID 字符串）。
        exercise_name: 要分析的动作名称（如 "深蹲"），对应模式 1。
        category: 身体部位类别（如 "背"），对应模式 2。
        days_back: 分析窗口天数（默认 90，上限 365）。

    Returns:
        包含进展统计数据的 JSON 字符串。
    """
    from agent.tools import current_user_id  # noqa: PLC0415

    enforced = current_user_id.get()
    if enforced:
        user_id = enforced

    user_uuid = _resolve_user_id(user_id)
    # 对参数做上限保护，防止查询范围过大
    days_back = min(days_back, 365)
    # 计算查询的起始日期（今天往回推 N 天）
    cutoff = date.today() - timedelta(days=days_back)

    async with async_session_factory() as session:
        user = await session.get(User, user_uuid)
        if user is None:
            user = User(id=user_uuid, username=user_id)
            session.add(user)
            await session.flush()
            logger.info(f"Auto-created user: {user_id} ({user_uuid})")

        # ── 模式 2: 按身体部位聚合统计 ──
        if category is not None:
            return await _analyze_by_category(session, user_uuid, category, cutoff, days_back)

        # ── 模式 1: 单动作趋势分析（默认，exercise_name 未传则为空字符串）──
        target_exercise = exercise_name or ""
        return await _analyze_by_exercise(session, user_uuid, target_exercise, cutoff, days_back)


async def _analyze_by_exercise(
    session,
    user_uuid: uuid.UUID,
    exercise_name: str,
    cutoff: date,
    days_back: int,
) -> str:
    """单动作趋势分析 —— 统计指定动作的重量和训练量变化趋势。

    核心算法：将时间段平分为前后两半，比较平均值的百分比变化。
    """
    # 查询该动作的所有训练记录，按日期升序排列
    stmt = (
        select(
            WorkoutSession.session_date,
            WorkoutItem.exercise_name,
            WorkoutItem.sets,
            WorkoutItem.reps,
            WorkoutItem.weight_kg,
            # duration_seconds 和 distance_meters 当前始终为 None，仅为有氧运动预留
            WorkoutItem.duration_seconds,
            WorkoutItem.distance_meters,
        )
        .join(WorkoutItem, WorkoutItem.session_id == WorkoutSession.id)
        .where(
            WorkoutSession.user_id == user_uuid,
            WorkoutSession.session_date >= cutoff,
            WorkoutItem.exercise_name.ilike(f"%{exercise_name}%"),  # 模糊匹配动作名
        )
        .order_by(WorkoutSession.session_date.asc())  # 升序，便于计算趋势
    )
    result = await session.execute(stmt)
    rows = result.all()

    if not rows:
        return json.dumps({
            "exercise": exercise_name,
            "message": f"未找到 {exercise_name} 在最近 {days_back} 天的训练记录。",
            "total_sessions": 0,
        }, ensure_ascii=False)

    # 按日期聚合：同一天可能有多组同一动作的训练，合并计算
    sessions_data: dict[str, dict] = {}
    for row in rows:
        d = str(row.session_date)
        if d not in sessions_data:
            sessions_data[d] = {"max_weight": 0.0, "total_volume": 0.0, "count": 0}
        entry = sessions_data[d]
        weight = row.weight_kg or 0
        sets = row.sets or 0
        reps = row.reps or 0
        # 训练量 = 重量 × 组数 × 次数（单位：公斤）
        volume = weight * sets * reps
        # 当天最大重量（用于追踪力量增长）
        entry["max_weight"] = max(entry["max_weight"], weight)
        # 当天总训练量（用于追踪训练负荷）
        entry["total_volume"] += volume
        entry["count"] += 1

    dates = sorted(sessions_data.keys())
    max_weights = [sessions_data[d]["max_weight"] for d in dates]
    volumes = [sessions_data[d]["total_volume"] for d in dates]
    session_count = len(dates)

    # 前后半段对比：将时间平分为两半，比较平均值变化百分比
    if len(max_weights) >= 2:
        # 前半段平均最大重量
        first_half_max = sum(max_weights[: len(max_weights) // 2]) / (len(max_weights) // 2)
        # 后半段平均最大重量
        second_half_max = sum(max_weights[len(max_weights) // 2:]) / (len(max_weights) - len(max_weights) // 2)
        # 重量变化百分比（正数 = 进步，负数 = 退步）
        weight_change_pct = round(
            ((second_half_max - first_half_max) / first_half_max * 100) if first_half_max > 0 else 0, 1
        )
        # 前半段平均训练量
        first_half_vol = sum(volumes[: len(volumes) // 2]) / (len(volumes) // 2)
        # 后半段平均训练量
        second_half_vol = sum(volumes[len(volumes) // 2:]) / (len(volumes) - len(volumes) // 2)
        # 训练量变化百分比
        volume_change_pct = round(
            ((second_half_vol - first_half_vol) / first_half_vol * 100) if first_half_vol > 0 else 0, 1
        )
    else:
        weight_change_pct = 0.0
        volume_change_pct = 0.0

    # 计算每周平均训练次数
    if len(dates) >= 2:
        days_span = (date.today() - date.fromisoformat(dates[0])).days or 1
        sessions_per_week = round(session_count / (days_span / 7), 1)
    else:
        sessions_per_week = 1.0

    analysis = {
        "exercise": exercise_name,
        "period_days": days_back,
        "total_sessions": session_count,
        "sessions_per_week": sessions_per_week,
        # 最近一次训练的最大重量
        "latest_max_weight_kg": max_weights[-1] if max_weights else 0,
        # 全周期内出现的最大重量
        "all_time_max_weight_kg": max(max_weights) if max_weights else 0,
        "avg_max_weight_first_half_kg": round(first_half_max, 1) if session_count >= 2 else (max_weights[0] if max_weights else 0),
        "avg_max_weight_second_half_kg": round(second_half_max, 1) if session_count >= 2 else (max_weights[0] if max_weights else 0),
        "max_weight_change_pct": weight_change_pct,
        "avg_volume_first_half": round(first_half_vol, 0) if session_count >= 2 else (volumes[0] if volumes else 0),
        "avg_volume_second_half": round(second_half_vol, 0) if session_count >= 2 else (volumes[0] if volumes else 0),
        "volume_change_pct": volume_change_pct,
        # 趋势判断：变化 > 2% 为上升，< -2% 为下降，在中间为稳定
        "trend_direction": "up" if weight_change_pct > 2 else "down" if weight_change_pct < -2 else "stable",
    }

    logger.info(f"Progress analysis for {exercise_name}: {session_count} sessions, {weight_change_pct}% weight change")
    return json.dumps(analysis, ensure_ascii=False, indent=2)


async def _analyze_by_category(
    session,
    user_uuid: uuid.UUID,
    category: str,
    cutoff: date,
    days_back: int,
) -> str:
    """按身体部位聚合统计 —— 汇总某个部位下所有动作的训练数据。

    采用两级查询策略：
    1. 首选通过 ExerciseCatalog 外键关联精确匹配类别
    2. 若无匹配（旧数据未关联目录），回退到按动作名称模糊匹配
    """
    # 策略 1: 通过 exercise_catalog_id 外键精确关联类别
    stmt = (
        select(
            WorkoutItem.exercise_name,
            WorkoutItem.sets,
            WorkoutItem.reps,
            WorkoutItem.weight_kg,
            WorkoutItem.duration_seconds,
            WorkoutItem.distance_meters,
            WorkoutSession.session_date,
            WorkoutSession.id,
        )
        .join(WorkoutSession, WorkoutItem.session_id == WorkoutSession.id)
        .join(ExerciseCatalog, WorkoutItem.exercise_catalog_id == ExerciseCatalog.id)
        .where(
            WorkoutSession.user_id == user_uuid,
            WorkoutSession.session_date >= cutoff,
            ExerciseCatalog.category == category,  # 按类别精确筛选
        )
        .order_by(WorkoutSession.session_date.desc())
    )
    result = await session.execute(stmt)
    rows = result.all()

    # Best-effort: 如果外键关联无结果，尝试回退方案
    if not rows:
        # 先查出该类别下有哪些动作名称
        cat_result = await session.execute(
            select(ExerciseCatalog.name).where(ExerciseCatalog.category == category)
        )
        cat_names = [r[0] for r in cat_result.all()]
        if cat_names:
            # 用动作名称做模糊匹配（ILike），覆盖未关联 catalog 的历史数据
            name_conditions = [
                WorkoutItem.exercise_name.ilike(f"%{n}%") for n in cat_names
            ]
            stmt = (
                select(
                    WorkoutItem.exercise_name,
                    WorkoutItem.sets,
                    WorkoutItem.reps,
                    WorkoutItem.weight_kg,
                    WorkoutItem.duration_seconds,
                    WorkoutItem.distance_meters,
                    WorkoutSession.session_date,
                    WorkoutSession.id,
                )
                .join(WorkoutSession, WorkoutItem.session_id == WorkoutSession.id)
                .where(
                    WorkoutSession.user_id == user_uuid,
                    WorkoutSession.session_date >= cutoff,
                    or_(*name_conditions),  # 任一动作名匹配即可
                )
                .order_by(WorkoutSession.session_date.desc())
            )
            result = await session.execute(stmt)
            rows = result.all()

    if not rows:
        return json.dumps({
            "category": category,
            "message": f"未找到 {category} 部在最近 {days_back} 天的训练记录。",
            "total_sessions": 0,
            "total_exercises": 0,
            "total_sets": 0,
            "total_reps": 0,
            "total_volume_kg": 0,
        }, ensure_ascii=False)

    # 对该部位下所有动作做聚合统计
    exercises_seen: set[str] = set()   # 去重统计做过哪些动作
    sessions_seen: set[str] = set()    # 去重统计多少次训练
    total_sets = 0
    total_reps = 0
    total_volume = 0.0

    for row in rows:
        exercises_seen.add(row.exercise_name)
        sessions_seen.add(str(row.id))
        total_sets += row.sets or 0
        total_reps += row.reps or 0
        w = row.weight_kg or 0
        s = row.sets or 0
        r = row.reps or 0
        # 仅当重量、组数、次数均有效时才计入训练量
        if w > 0 and s > 0 and r > 0:
            total_volume += w * s * r

    # 计算每周平均训练频率
    sessions_per_week = 0.0
    if len(sessions_seen) >= 2:
        # 通过实际训练日期跨度和 session 数来估算
        dates_seen = sorted({str(r.session_date) for r in rows})
        if len(dates_seen) >= 2:
            span = (date.today() - date.fromisoformat(dates_seen[0])).days or 1
            sessions_per_week = round(len(sessions_seen) / (span / 7), 1)

    analysis = {
        "category": category,
        "period_days": days_back,
        "total_sessions": len(sessions_seen),
        "sessions_per_week": sessions_per_week,
        "total_exercises": len(exercises_seen),
        "exercises": sorted(exercises_seen),  # 按字母排序的动作列表
        "total_sets": total_sets,
        "total_reps": total_reps,
        "total_volume_kg": round(total_volume, 1),
    }

    logger.info(
        f"Category analysis for {category}: {len(sessions_seen)} sessions, "
        f"{len(exercises_seen)} exercises, {total_volume}kg volume"
    )
    return json.dumps(analysis, ensure_ascii=False, indent=2)
