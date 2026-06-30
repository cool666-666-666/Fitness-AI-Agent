# 🏋️ 健身 AI 助手 (Fitness AI Agent)

针对健身爱好者训练数据碎片化、知识获取低效、缺乏趋势分析三大痛点，基于 ReAct Agent 与 RAG 技术提供三大核心功能：结构化录入训练数据，上传健身文档后专业知识精准问答，历史训练数据自动生成力量趋势与训练量化分析。



---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [API 接口](#api-接口)
- [配置说明](#配置说明)
- [本地开发](#本地开发)
- [测试](#测试)
- [工具链](#工具链)

---

## 功能特性

- 🤖 **对话式 AI 教练** — 基于 LangGraph ReAct Agent，自动判断用户意图并调用对应工具
- 📚 **知识库搜索 (RAG)** — 上传健身/运动科学文档，通过语义搜索获取专业知识
- 📝 **训练记录** — 自然语言描述训练内容，自动解析并存入数据库（如 "深蹲 80kg 5组5次"）
- 📊 **进度分析** — 查询历史训练数据，追踪力量增长和训练量变化
- 🔄 **混合检索** — 稠密向量 + BM25 稀疏向量，兼顾语义理解和关键词匹配
- 🎯 **重排序 (Reranking)** — 支持 Cohere 和 FlashRank 两种重排序引擎，提升检索精度
- 🔍 **链路追踪** — 基于 Phoenix 的可观测性，监控每次 LLM 调用和检索过程
- 🌐 **REST API** — FastAPI 构建，自动生成 OpenAPI 文档
- 🖥️ **Web 前端** — Streamlit 构建的对话界面

---

## 系统架构

```
┌──────────────┐     ┌─────────────────────────────────────┐
│   Streamlit  │────►│         FastAPI 后端 (8001)          │
│   前端 (8501) │     │                                     │
└──────────────┘     │  ┌───────────────────────────────┐  │
                     │  │   LangGraph ReAct Agent        │  │
                     │  │   ┌─────┐ ┌─────┐ ┌─────┐    │  │
                     │  │   │聊天 │ │训练 │ │知识 │    │  │
                     │  │   │工具 │ │工具 │ │搜索 │    │  │
                     │  │   └──┬──┘ └──┬──┘ └──┬──┘    │  │
                     │  └──────┼───────┼───────┼───────┘  │
                     │         │       │       │          │
                     └─────────┼───────┼───────┼──────────┘
                               │       │       │
                      ┌────────▼──┐ ┌──▼───────▼──┐
                      │   MySQL    │ │    Qdrant    │
                      │  训练数据   │ │  知识库向量   │
                      └────────────┘ └──────┬──────┘
                                            │
                                     ┌──────▼──────┐
                                     │   Phoenix    │
                                     │   链路追踪    │
                                     └─────────────┘
```

### 三大核心存储

| 数据库 | 用途 | 存储内容 |
|--------|------|---------|
| **MySQL** | 结构化业务数据 | 用户档案、训练课表、动作记录、动作库 |
| **Qdrant** | 非结构化知识库 | 健身文档的向量嵌入，支撑语义搜索和 RAG |
| **Phoenix** | 可观测性 | LLM 调用、检索步骤、Tool 使用的链路追踪 |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| **LLM 网关** | LiteLLM（统一接口，支持 DeepSeek / Gemini / Cohere / OpenAI） |
| **Agent 框架** | LangGraph（ReAct Agent + MemorySaver 对话记忆） |
| **嵌入模型** | FastEmbed / Qwen Embedding / Google Embedding |
| **向量数据库** | Qdrant（混合检索：稠密 + BM25 稀疏向量） |
| **关系数据库** | MySQL 8.4 + SQLAlchemy + aiomysql |
| **重排序** | Cohere Rerank / FlashRank（本地） |
| **链路追踪** | Arize Phoenix + OpenInference + OpenTelemetry |
| **Web 框架** | FastAPI + Uvicorn |
| **前端** | Streamlit |
| **包管理** | uv |
| **容器化** | Docker Compose |

---

## 快速开始

### 前置条件

- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- （可选）[uv](https://docs.astral.sh/uv/getting-started/installation/) 用于本地开发

### 1. 克隆项目

```bash
git clone https://github.com/mfmezger/conversational-agent-langchain.git
cd conversational-agent-langchain
```

### 2. 配置环境变量

复制模板文件并填入 API Key：

```bash
cp template.env .env
```

**必须配置的 Key：**

| 变量 | 说明 |
|------|------|
| `GEMINI_API_KEY` | Google Gemini API 密钥 |
| `COHERE_API_KEY` | Cohere API 密钥（重排序） |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（对话模型，推荐） |
| `QWEN_API_KEY` | 阿里云 DashScope API 密钥（嵌入模型） |

### 3. 启动服务

```bash
docker compose up -d
```

### 4. 访问服务

| 服务 | URL |
|------|-----|
| **API 文档** | http://127.0.0.1:8001/docs |
| **聊天前端** | http://localhost:8501 |
| **Qdrant 控制台** | http://localhost:6333/dashboard |
| **Phoenix 追踪** | http://localhost:6006 |

---

## 项目结构

```
项目2/
├── src/agent/
│   ├── main.py                  # FastAPI 应用入口 + 生命周期管理
│   ├── agent/
│   │   └── fitness_agent.py     # LangGraph ReAct Agent 构建
│   ├── api/                     # REST API 路由
│   │   ├── chat.py              # POST /chat — 对话接口
│   │   ├── workout.py           # POST /workout/add, GET /workout/history
│   │   └── documents.py         # POST /documents/upload — 文档上传
│   ├── tools/                   # Agent 可调用的工具
│   │   ├── add_workout.py       # 记录训练
│   │   ├── query_workout.py     # 查询训练历史
│   │   ├── analyze_progress.py  # 分析训练进展
│   │   └── rag_search.py        # 搜索健身知识库
│   ├── db/                      # MySQL 相关
│   │   ├── models.py            # SQLAlchemy ORM 模型
│   │   └── mysql.py             # 异步引擎和会话管理
│   ├── rag/                     # RAG 管线
│   │   ├── ingest.py            # 文档摄入
│   │   └── retriever.py         # 检索 + 重排序
│   ├── parser/
│   │   └── workout_parser.py    # 自然语言训练文本解析
│   ├── utils/
│   │   ├── vdb.py               # Qdrant 向量数据库连接
│   │   ├── config.py            # Pydantic 配置管理
│   │   ├── retriever.py         # 检索器工厂函数
│   │   └── reranker.py          # 重排序器
│   └── routes/                  # 旧版 API 路由（向后兼容）
│       ├── collection.py
│       ├── embeddings.py
│       ├── search.py
│       ├── rag.py
│       └── delete.py
├── frontend/
│   ├── assistant.py             # Streamlit 聊天界面
│   └── client.py                # 后端 API 客户端
├── tests/                       # 测试
├── docker-compose.yml           # 服务编排
├── Dockerfile                   # 后端镜像
├── Dockerfile.frontend          # 前端镜像
├── Makefile                     # 开发命令快捷方式
├── pyproject.toml               # 项目依赖和配置
└── template.env                 # 环境变量模板
```

---

## API 接口

### 对话

```
POST /chat
```

向 AI 教练发送消息，Agent 自动判断意图并调用对应工具。

```json
{
  "user_id": "uuid",
  "messages": [
    {"role": "user", "content": "今天深蹲80kg 5组5次，卧推60kg 3组8次"}
  ],
  "collection_name": "fitness_kb"
}
```

### 训练记录

```
POST /workout/add     — 自然语言记录训练
GET  /workout/history — 查询训练历史
```

### 知识库

```
POST /documents/upload — 上传健身文档（PDF/TXT/MD）
```

### 旧版接口（向后兼容）

```
POST /embeddings/embed_documents   — 嵌入文档
POST /semantic/semantic_search     — 语义搜索
POST /rag/qa                       — 基于文档的问答
DELETE /embeddings/delete          — 删除嵌入
```

---

## 配置说明

主要通过 `.env` 文件配置，由 [config.py](src/agent/utils/config.py) 中的 Pydantic Settings 加载：

### 模型配置

```bash
# 对话模型
DEEPSEEK_API_KEY=sk-xxx         # DeepSeek API 密钥
DEEPSEEK_MODEL_NAME=deepseek-chat
GEMINI_API_KEY=xxx              # Google Gemini（需通过 LiteLLM 使用）

# 嵌入模型
EMBEDDING_PROVIDER=qwen         # google / qwen / openai / dashscope
EMBEDDING_MODEL_NAME=text-embedding-v4
EMBEDDING_SIZE=768
```

### 重排序配置

```bash
RERANK_PROVIDER=cohere           # cohere / flashrank / none
RERANK_TOP_K=5                   # 重排序后保留的文档数
COHERE_API_KEY=xxx
```

### 检索配置

```bash
RETRIEVAL_K=40                    # 初始检索文档数
RETRIEVAL_K_RETRY=100             # 重试时的检索数
```

### 数据库配置

```bash
# MySQL
DATABASE_URL=mysql+aiomysql://root:root@localhost:3306/fitness_agent

# Qdrant
QDRANT_URL=http://localhost
QDRANT_API_KEY=test
QDRANT_PORT=6333

# Phoenix
PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:4318/v1/traces
```

---

## 本地开发

### 安装依赖

```bash
uv sync
```

### 单独启动 Qdrant

本地开发时不需要启动全部 Docker 服务，只需 Qdrant 和 MySQL：

```bash
docker compose up -d qdrant mysql
```

### 启动后端

```bash
uv run uvicorn src.agent.main:app --reload --port 8001
```

### 启动前端

```bash
cd frontend && uv run streamlit run assistant.py --theme.base="dark"
```

### Makefile 快捷命令

| 命令 | 说明 |
|------|------|
| `make setup` | 安装依赖和 git hooks |
| `make style` | 运行代码格式化/检查 |
| `make test` | 运行单元测试和集成测试 |
| `make start_backend` | 启动 FastAPI 后端 |
| `make start_frontend` | 启动 Streamlit 前端 |
| `make start_docker` | 启动全部 Docker 服务 |
| `make restart` | 重建并重启所有服务 |
| `make clean` | 清理构建产物 |

---

## 测试

```bash
# 单元测试和集成测试（并行执行）
uv run pytest -n auto -m "not vcr and not e2e" -v tests/

# VCR 录制测试
uv run pytest -m "vcr" -v tests/

# E2E 测试（需要完整服务运行）
RUN_LIVE_E2E=1 uv run pytest -m "e2e" -v tests/

# 覆盖率报告
uv run coverage run -m pytest tests/ && uv run coverage html
```

### 测试标记

| 标记 | 说明 |
|------|------|
| `unit` | 纯单元测试 |
| `integration` | 集成测试（进程内依赖） |
| `contract` | 接口契约测试 |
| `e2e` | 端到端测试（需要外部服务） |
| `vcr` | 基于 HTTP 录播的测试 |
| `slow` | 耗时较长的测试 |

---

## 工具链

| 工具 | 用途 |
|------|------|
| [ruff](https://github.com/astral-sh/ruff) | 代码格式化和 linting |
| [prek](https://prek.j178.dev/) | Rust 原生 git hooks |
| [commitizen](https://commitizen-tools.github.io/commitizen/) | 规范化 commit 信息 |
| [mypy](https://www.mypy-lang.org/) | 静态类型检查 |
| [pytest](https://pytest.org/) | 测试框架 |
| [Bruno](https://www.usebruno.com/) | API 测试客户端 |

---

## 什么是 RAG？

检索增强生成（Retrieval-Augmented Generation）是一种增强大语言模型（LLM）的技术。它不依赖模型预训练时记住的知识，而是**实时从外部知识库检索相关文档**，将这些文档作为上下文提供给 LLM，从而：

- 减少幻觉（hallucination）
- 回答私有/专业领域数据的问题
- 保证信息时效性

本项目中，用户上传的健身文档被分段、嵌入后存入 Qdrant。提问时，系统先检索最相关的段落，再交给 LLM 合成回答。

### 混合检索

本系统同时使用两种检索方式：

1. **稠密向量检索** — 基于语义理解，找到"意思相近"的内容
2. **稀疏向量检索 (BM25)** — 基于关键词匹配，确保精确命中

两者结果融合，兼顾语义广度和关键词精度。

---

## Star History

<a href="https://star-history.com/#mfmezger/conversational-agent-langchain&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mfmezger/conversational-agent-langchain&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mfmezger/conversational-agent-langchain&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mfmezger/conversational-agent-langchain&type=Date" />
  </picture>
</a>
