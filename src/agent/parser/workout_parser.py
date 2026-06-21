"""训练内容解析器 — 从结构化输入直接构建训练记录。

前端使用固定输入框（动作名称、组数、每组次数+重量），
不再需要自然语言解析，直接组装为 ParsedWorkout 数据结构。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SetDetail(BaseModel):
    """单组训练明细。"""

    reps: int = Field(description="该组次数", ge=0)
    weight_kg: float = Field(default=0.0, description="该组重量（公斤），0 表示自重", ge=0.0)


class ParsedExercise(BaseModel):
    """从用户输入中解析出的单个训练动作。

    属性：
        name: 动作名称，如"深蹲"、"卧推"
        sets: 组数
        reps: 总次数（所有组次数之和）
        weight_kg: 平均重量（公斤），全自重时为 None
        set_details: 每组明细 [{"reps": 10, "weight_kg": 80.0}, ...]
        notes: 备注信息
    """

    name: str = Field(description="动作名称，如 深蹲、卧推")
    sets: int | None = Field(default=None, description="组数")
    reps: int | None = Field(default=None, description="总次数")
    weight_kg: float | None = Field(default=None, description="平均重量（公斤）")
    set_details: list[SetDetail] | None = Field(default=None, description="每组次数+重量明细")
    notes: str | None = Field(default=None, description="关于该动作的额外备注")


class ParsedWorkout(BaseModel):
    """解析训练描述后的结果。

    属性：
        exercises: 解析出的训练动作列表，至少包含一个动作
    """

    exercises: list[ParsedExercise] = Field(description="解析出的训练动作列表", min_length=1)


def build_workout(exercises: list[dict]) -> ParsedWorkout:
    """从结构化数据直接构建 ParsedWorkout。

    前端按组数显示每组次数+重量输入框，组装为 dict 列表后传入此函数。

    Args:
        exercises: 训练动作列表，每项包含:
            - name (str): 动作名称
            - sets (int): 组数
            - reps (int): 总次数
            - weight_kg (float, 可选): 平均重量（公斤），0 或不传表示自重
            - set_details (list[dict], 可选): 每组明细 [{"reps": N, "weight_kg": W}, ...]
            - notes (str, 可选): 备注

    Returns:
        ParsedWorkout 实例

    Raises:
        ValueError: exercises 为空或格式不正确
    """
    if not exercises:
        raise ValueError("exercises 不能为空")

    parsed = []
    for item in exercises:
        weight = item.get("weight_kg", 0)
        set_details = item.get("set_details")
        # 如果传了 set_details，转换为 SetDetail 对象列表
        if set_details:
            set_details = [{"reps": s["reps"], "weight_kg": s.get("weight_kg", 0)} for s in set_details]
        parsed.append(
            ParsedExercise(
                name=item["name"],
                sets=item.get("sets"),
                reps=item.get("reps"),
                weight_kg=float(weight) if weight else None,
                set_details=set_details,
                notes=item.get("notes"),
            )
        )

    return ParsedWorkout(exercises=parsed)
