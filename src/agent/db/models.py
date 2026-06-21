"""Fitness AI Agent 的 SQLAlchemy ORM 数据库模型。

本模块定义了 MySQL 数据库中所有表的结构映射。每个继承 Base 的类对应一张表，
类属性对应表中字段。通过 SQLAlchemy ORM，业务代码可以直接操作 Python 对象
而无需编写原始 SQL。

表关系概览：
    User (1) ──< (N) WorkoutSession (1) ──< (N) WorkoutItem (N) >── (1) ExerciseCatalog
"""

import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, JSON, String, Text, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类。

    所有 ORM 模型必须继承此类，SQLAlchemy 通过它发现并管理模型与数据库表之间的映射关系。
    自身不对应任何数据库表。
    """
    pass


class User(Base):
    """用户表。

    存储每个注册用户的基本信息。一个用户可以拥有多条训练记录。
    """
    __tablename__ = "users"

    # 用户唯一标识，UUID v4 自动生成，用于关联 workout_sessions 表。
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # 用户名，全局唯一，不可为空。用于登录和展示。
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # 账户创建时间，由数据库服务器端自动填入当前时间戳。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # 反向关联：该用户的所有训练记录。删除用户时级联删除其全部训练数据。
    workout_sessions: Mapped[list["WorkoutSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class ExerciseCatalog(Base):
    """动作目录表。

    预定义的健身动作知识库，在应用启动时通过 seed 脚本自动填充。
    提供动作的中文名称、所属肌群分类，以及英文/别名用于模糊匹配。
    """
    __tablename__ = "exercise_catalog"

    # 动作唯一标识。
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # 动作的标准中文名称（如 "卧推"、"深蹲"），全局唯一，用于查询和展示。
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)

    # 动作所属肌群分类：胸 / 背 / 腿 / 肩 / 手臂 / 核心 / 有氧。
    # 可为空——某些特殊动作可能尚无明确分类。
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # 动作的别名列表（JSON 数组），用于模糊匹配。
    # 例如 "卧推" 的别名为 ["bench press", "barbell bench press"]。
    # 存储为 JSON 列，支持灵活扩展。
    alias: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)


class WorkoutSession(Base):
    """训练记录表。

    表示用户的一次完整训练（某天的一堂训练课）。一个 Session 可包含多个训练动作项。
    """
    __tablename__ = "workout_sessions"

    # 训练记录唯一标识。
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # 所属用户的 ID，外键关联到 users 表。通过 user 字段可反向获取用户信息。
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False, index=True)

    # 训练日期。用于按日期查询和统计训练频率。
    session_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # 训练备注，用户可选的自由文本。记录当天整体感受、注意事项等。
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 记录创建时间，由数据库服务器端自动填入。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # 反向关联：所属的用户对象。
    user: Mapped["User"] = relationship(back_populates="workout_sessions")

    # 反向关联：本次训练包含的所有动作项。删除训练时级联删除其全部动作项。
    items: Mapped[list["WorkoutItem"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class WorkoutItem(Base):
    """训练动作明细表。

    记录一次训练课中的单个动作项，包含组数、次数、重量等详细训练数据。
    通过外键关联到 WorkoutSession（属于哪次训练）和 ExerciseCatalog（参照哪个标准动作）。
    """
    __tablename__ = "workout_items"

    # 动作项唯一标识。
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # 所属训练记录的 ID，外键关联到 workout_sessions 表。
    session_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("workout_sessions.id"), nullable=False, index=True)

    # 动作名称（当时用户输入的文字，可能为非标准名称）。
    # 与 exercise_catalog_id 配合使用：前者作标准参照，此处保留原始记录。
    exercise_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # 关联的标准动作目录 ID，外键关联到 exercise_catalog 表。
    # 可为空——用户输入的动作可能在目录中找不到匹配项。
    exercise_catalog_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("exercise_catalog.id"), nullable=True)

    # 训练组数。可为空——有氧运动可能不按组数记录。
    sets: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 每组次数。与组数配合构成训练量描述。
    reps: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 使用重量（单位：公斤）。可为空——自重训练可能无重量数据。
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 运动时长（单位：秒）。主要用于有氧运动（如跑步 1800 秒）。
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 运动距离（单位：米）。主要用于有氧运动（如跑步 5000 米）。
    distance_meters: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 自感用力程度 RPE（Rating of Perceived Exertion），0~10 量表。
    # 用于主观衡量训练强度，10 表示极限用力。
    rpe: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 组间休息时间（单位：秒）。
    rest_time_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 动作节奏（如 "3-1-3-0" 表示离心3秒-底端1秒-向心3秒-顶端0秒）。
    tempo: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # 每组明细（次数+重量对应关系），JSON 数组格式：
    # [{"reps": 10, "weight_kg": 80.0}, {"reps": 8, "weight_kg": 75.0}, ...]
    set_details: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)

    # 该动作项的备注说明，可记录动作变形、器械设置等细节。
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 记录创建时间，由数据库服务器端自动填入。
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # 反向关联：所属的训练记录对象，通过 session 可访问该次训练的其他信息。
    session: Mapped["WorkoutSession"] = relationship(back_populates="items")

    # 关联的标准动作目录条目。可为空——用户输入的动作可能无法匹配到标准目录。
    exercise_catalog: Mapped["ExerciseCatalog | None"] = relationship()
