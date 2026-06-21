"""Fitness AI Agent 应用入口 —— FastAPI 应用工厂、生命周期管理和路由注册。

启动顺序（从上到下，模块级代码在 import 时即执行）：
1. 加载 .env 环境变量
2. 构建 Config 单例（读取所有配置项）
3. 初始化 Phoenix/OpenTelemetry 链路追踪
4. 打印 ASCII 启动横幅
5. 创建 FastAPI app 实例
6. 注册全局异常处理器
7. 挂载各子模块路由
"""

import pyfiglet
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from loguru import logger
# OpenInference: 自动为 LangChain/LangGraph 创建 OpenTelemetry span，
# 将 Agent 内部的 LLM 调用、工具调用、检索等操作作为 tracing span 上报
from openinference.instrumentation.langchain import LangChainInstrumentor
# Phoenix: Arize 的开源 LLM 可观测性平台，收集并可视化 tracing 数据
from phoenix.otel import register

from agent.api import chat, documents, workout
from agent.utils.config import Config
from agent.utils.vdb import initialize_all_vector_dbs

# 加载 .env 文件中的环境变量，override=True 表示 .env 中的值优先于系统环境变量
load_dotenv(override=True)
# 模块级 Config 单例 —— 整个应用共享同一份配置，避免重复解析 .env
config = Config()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期管理器 —— FastAPI 启动和关闭时的初始化与清理逻辑。

    启动阶段（yield 之前）按顺序执行：
    1. 初始化向量数据库连接（Qdrant）
    2. 初始化 MySQL 表结构
    3. 填充动作目录种子数据

    关闭阶段（yield 之后）：
    暂无需手动清理的资源（连接由各模块自行管理）。
    """
    # ---- 启动阶段 ----
    logger.info("Initializing vector databases...")
    # 初始化 Qdrant 客户端，创建/验证集合是否存在
    initialize_all_vector_dbs(config=config)

    logger.info("Initializing MySQL...")
    # 延迟导入：MySQL 初始化需要用到全局配置，放在函数内可以避免
    # import 时触发循环依赖（mysql 模块也依赖 config）
    from agent.db.mysql import init_db  # noqa: PLC0415
    # 创建数据库表（若无），创建成功后 init_db 内部为 no-op
    await init_db()

    logger.info("Seeding exercise catalog...")
    from agent.db.seed import seed_exercise_catalog  # noqa: PLC0415
    # 填充 ExerciseCatalog 表（胸/背/腿/肩/手臂/核心/有氧 七大类别下的标准动作）
    # 若种子数据已存在则跳过
    await seed_exercise_catalog()

    logger.info("Startup complete.")
    # ---- yield 交出控制权给 FastAPI 处理请求 ----
    yield
    # ---- 关闭阶段 ----
    logger.info("Shutting down.")

# ---- OpenTelemetry / Phoenix 链路追踪初始化 ----
# register() 返回一个全局 TracerProvider，后续所有 span 都由此提供者管理
tracer_provider = register(
    project_name="rag",  # Phoenix UI 中显示的项目名称
    endpoint=config.phoenix_collector_endpoint,  # Phoenix 收集器的 gRPC/HTTP 地址
)

# 自动注入 LangChain/LangGraph 的 tracing hook：
# - LLM 调用 → llm span（模型名、token 数、延迟）
# - 工具调用 → tool span（工具名、输入输出）
# - 检索操作 → retrieval span（查询文本、返回文档数）
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# 打印 ASCII 艺术字启动横幅（通过 loguru 输出到控制台）
f = pyfiglet.figlet_format("Fitness AI", font="alligator")
logger.info(f"Welcome to\n\n{f}\n\n")


def my_schema() -> dict:
    """生成自定义 OpenAPI Schema，在基础 spec 上添加项目级元数据。

    该函数通过 app.openapi 属性注入，FastAPI 每次访问 /openapi.json
    时调用它，而非直接使用自动生成的 schema。
    这意味着元数据（title/version/description）可以集中在此维护。
    """
    openapi_schema = get_openapi(
        title="Fitness AI Agent API",
        version="1.0",
        description="Intelligent fitness coach with training records and knowledge base.",
        routes=app.routes,
    )
    # 将生成的 schema 缓存到 app 实例，避免重复计算
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# FastAPI 应用实例创建
# lifespan 负责启停时的初始化/清理，替代已废弃的 on_event("startup"/"shutdown")
app = FastAPI(lifespan=lifespan)
# 注入自定义 OpenAPI 生成函数（延迟调用：首次访问 /docs 或 /openapi.json 时触发）
app.openapi = my_schema


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """全局异常兜底处理器 —— 捕获所有未被路由层处理器的异常。

    作用：
    - 防止 500 错误时返回 HTML traceback（FastAPI 默认行为）
    - 统一返回 JSON 格式的错误信息，方便前端解析
    - 通过 loguru 将完整异常记录到日志，便于排查

    注意：此处理器在路由层异常处理（try/except + HTTPException）之后执行，
    仅捕获那些"穿透"了业务代码的未处理异常。
    """
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)},
    )


logger.info("Loading REST API Finished.")

# ---- 路由注册 ----
# 每个子模块有独立的 router 和 prefix，URL 前缀即 API 版本边界
app.include_router(router=chat.router, prefix="/chat")          # POST /chat, /chat/stream
app.include_router(router=workout.router, prefix="/workout")    # POST /workout/add, /workout/history
app.include_router(router=documents.router, prefix="/documents")  # POST /documents/upload


@app.get(path="/", tags=["root"])
def read_root() -> str:
    """根路径健康检查 —— 返回欢迎信息，引导访问 /docs 查看完整 API 文档。"""
    return "Welcome to the RAG Backend. Please navigate to /docs for the OpenAPI!"


if __name__ == "__main__":
    # 直接执行 python main.py 时的开发模式启动
    # 生产环境应通过 Dockerfile 中的 uvicorn 命令启动（支持热重载、worker 数等配置）
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
