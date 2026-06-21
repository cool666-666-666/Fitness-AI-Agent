"""工作输出解析器测试 — 结构化构建。"""

import pytest
from agent.parser.workout_parser import (
    ParsedExercise,
    ParsedWorkout,
    build_workout,
)


class TestBuildWorkout:
    """测试 build_workout 工厂函数。"""

    def test_single_exercise(self):
        """单动作构建。"""
        result = build_workout([
            {"name": "深蹲", "sets": 5, "reps": 10, "weight_kg": 80.0},
        ])
        assert isinstance(result, ParsedWorkout)
        assert len(result.exercises) == 1
        ex = result.exercises[0]
        assert ex.name == "深蹲"
        assert ex.sets == 5
        assert ex.reps == 10
        assert ex.weight_kg == 80.0

    def test_multiple_exercises(self):
        """多动作构建。"""
        result = build_workout([
            {"name": "深蹲", "sets": 5, "reps": 10, "weight_kg": 80.0},
            {"name": "卧推", "sets": 3, "reps": 8, "weight_kg": 60.0},
            {"name": "引体向上", "sets": 4, "reps": 12, "weight_kg": 0},
        ])
        assert len(result.exercises) == 3
        assert result.exercises[1].name == "卧推"
        # weight_kg=0 → None (自重)
        assert result.exercises[2].weight_kg is None

    def test_zero_weight_becomes_none(self):
        """weight_kg=0 或未传时应转为 None（自重训练）。"""
        result = build_workout([
            {"name": "俯卧撑", "sets": 3, "reps": 20},
        ])
        assert result.exercises[0].weight_kg is None

    def test_empty_exercises_raises(self):
        """空列表应抛出 ValueError。"""
        with pytest.raises(ValueError, match="exercises 不能为空"):
            build_workout([])

    def test_missing_name_raises(self):
        """缺少 name 字段应抛出 KeyError。"""
        with pytest.raises(KeyError):
            build_workout([{"sets": 3, "reps": 10}])

    def test_notes_passthrough(self):
        """notes 字段应正确传递。"""
        result = build_workout([
            {"name": "深蹲", "sets": 5, "reps": 38, "weight_kg": 80.0,
             "notes": "最后一组力竭"},
        ])
        assert result.exercises[0].notes == "最后一组力竭"

    def test_set_details_passthrough(self):
        """set_details 应正确传递每组次数与重量的对应关系。"""
        result = build_workout([
            {"name": "深蹲", "sets": 5, "reps": 38, "weight_kg": 75.0,
             "set_details": [
                 {"reps": 10, "weight_kg": 80.0},
                 {"reps": 8, "weight_kg": 80.0},
                 {"reps": 8, "weight_kg": 75.0},
                 {"reps": 6, "weight_kg": 70.0},
                 {"reps": 6, "weight_kg": 70.0},
             ]},
        ])
        ex = result.exercises[0]
        assert ex.set_details is not None
        assert len(ex.set_details) == 5
        # 每组次数与重量对应（SetDetail 为 Pydantic 对象，用属性访问）
        assert ex.set_details[0].reps == 10
        assert ex.set_details[0].weight_kg == 80.0
        assert ex.set_details[2].reps == 8
        assert ex.set_details[2].weight_kg == 75.0

    def test_set_details_default_zero_weight(self):
        """set_details 中未传 weight_kg 应默认为 0（自重）。"""
        result = build_workout([
            {"name": "俯卧撑", "sets": 3, "reps": 30, "weight_kg": 0,
             "set_details": [
                 {"reps": 10},
                 {"reps": 10, "weight_kg": 0},
                 {"reps": 10},
             ]},
        ])
        ex = result.exercises[0]
        assert ex.set_details is not None
        for s in ex.set_details:
            assert s.weight_kg == 0

    def test_reps_can_be_zero(self):
        """reps=0 应被保留（力竭训练等情况）。"""
        result = build_workout([
            {"name": "俯卧撑", "sets": 3, "reps": 0},
        ])
        assert result.exercises[0].reps == 0


class TestParsedWorkoutModel:
    """Pydantic 模型验证测试。"""

    def test_parsed_exercise_defaults(self):
        """ParsedExercise 的默认值应为 None。"""
        ex = ParsedExercise(name="深蹲")
        assert ex.sets is None
        assert ex.reps is None
        assert ex.weight_kg is None
        assert ex.notes is None

    def test_parsed_workout_min_length(self):
        """ParsedWorkout 要求至少包含 1 个动作。"""
        with pytest.raises(ValueError):
            ParsedWorkout(exercises=[])

    def test_parsed_workout_valid(self):
        """正常构建 ParsedWorkout。"""
        w = ParsedWorkout(exercises=[
            ParsedExercise(name="深蹲", sets=5, reps=10, weight_kg=80.0),
        ])
        assert len(w.exercises) == 1
        assert w.exercises[0].name == "深蹲"
