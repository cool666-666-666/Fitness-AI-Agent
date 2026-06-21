FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Enable bytecode compilation (faster startup)
ENV UV_COMPILE_BYTECODE=1

# Copy from cache instead of linking (required for Docker layer caching)
ENV UV_LINK_MODE=copy

# Use Tsinghua PyPI mirror for faster downloads in China
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_HTTP_TIMEOUT=300

# copy python installation files.
COPY ./pyproject.toml ./pyproject.toml
COPY ./README.md ./README.md
COPY ./uv.lock ./uv.lock

# installing python dependencies
RUN uv sync --frozen --no-install-project

COPY ./src /src

RUN uv sync --frozen

CMD ["uv", "run", "uvicorn", "src.agent.main:app", "--host", "0.0.0.0", "--port", "8001"]
