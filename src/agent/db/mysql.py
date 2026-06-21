"""异步 MySQL 引擎与会话管理。

本模块封装了 SQLAlchemy 异步引擎的创建、会话获取和生命周期管理。
所有数据库操作均通过此模块获取异步会话，避免阻塞 FastAPI 事件循环。

核心设计：
    - 引擎和会话工厂采用懒加载（lazy-init），防止导入时数据库未就绪而报错。
    - 提供两套会话获取方式：FastAPI 依赖注入（自动提交/回滚）和
      直接上下文管理器（供 Agent Tool 等非路由场景使用）。
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agent.utils.config import config

# 全局异步引擎实例，懒加载，应用生命周期内复用。None 表示尚未初始化。
_engine = None

# 全局异步会话工厂，与 _engine 配套创建。None 表示尚未初始化。
_async_session_factory = None


def _get_engine():
    """懒加载初始化异步引擎与会话工厂。

    全局单例模式：首次调用时根据配置创建引擎和工厂，后续调用直接返回缓存实例。
    支持 MySQL 连接池参数（pool_size / max_overflow）的自动配置。

    Returns:
        tuple[AsyncEngine, async_sessionmaker]: 引擎实例和会话工厂。
    """
    global _engine, _async_session_factory  # noqa: PLW0603
    if _engine is None:
        db_url = config.database_url
        engine_kwargs: dict = {"echo": False}
        # MySQL 类驱动支持连接池配置，用于控制并发连接数。
        if "mysql" in db_url or "aiomysql" in db_url or "asyncmy" in db_url:
            engine_kwargs["pool_size"] = config.db_pool_size
            engine_kwargs["max_overflow"] = config.db_max_overflow
        _engine = create_async_engine(db_url, **engine_kwargs)
        # expire_on_commit=False: 提交后不使对象过期，允许在提交后继续访问属性。
        _async_session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)
    return _engine, _async_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入用会话生成器。

    用法：在路由函数参数中声明 `session: AsyncSession = Depends(get_db_session)`。
    生成器在请求进入时创建会话，请求结束时自动提交；若发生异常则自动回滚。

    Yields:
        AsyncSession: 一个数据库异步会话实例。

    Raises:
        异常会触发回滚并向上重新抛出，由 FastAPI 全局异常处理器接管。
    """
    _, factory = _get_engine()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def async_session_factory() -> AsyncSession:
    """返回一个异步会话上下文管理器，供 FastAPI 依赖注入之外的场景使用。

    适用场景：Agent Tool、后台任务、CLI 脚本等不在路由函数中的数据库操作。

    用法：
        async with async_session_factory() as session:
            result = await session.execute(...)

    Returns:
        AsyncSession: 一个异步会话上下文管理器（通过 async with 使用）。
    """
    _, factory = _get_engine()
    return factory()


async def init_db() -> None:
    """应用启动时调用，自动创建所有缺失的数据库表。

    通过 SQLAlchemy 的 Base.metadata.create_all 实现，已有的表不会重复创建。
    应在 FastAPI 的 lifespan startup 事件中调用。
    """
    from agent.db.models import Base  # noqa: PLC0415

    engine, _ = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """应用关闭时调用，销毁数据库引擎释放连接池资源。

    应在 FastAPI 的 lifespan shutdown 事件中调用。
    销毁后全局变量重置为 None，下次调用时会重新创建。
    """
    global _engine, _async_session_factory  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
