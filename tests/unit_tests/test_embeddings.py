"""嵌入模型工厂函数测试 —— 验证不同配置下 get_embedding_model 的行为。

测试焦点：
- get_embedding_model 是 agent.utils.embeddings 中的工厂函数，
  根据 Config 配置创建 OpenAIEmbeddings 实例。
- 不发起真实网络请求（用 mock 替换 OpenAIEmbeddings 类）。
- 不读取真实 .env 文件（用 SimpleNamespace 模拟配置对象）。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.utils.config import Config
from agent.utils.embeddings import get_embedding_model


def test_get_embedding_model_qwen() -> None:
    """使用 qwen_api_key 创建 Qwen 嵌入模型。

    验证点：
    - OpenAIEmbeddings 被调用时传入正确的 model / api_key / base_url / dimensions
    - 返回值是 mock 的 OpenAIEmbeddings 实例
    """
    # SimpleNamespace 模拟 Config 对象，无需真实 .env 文件
    cfg = SimpleNamespace(
        embedding_provider="qwen",
        embedding_model_name="text-embedding-v4",
        qwen_api_key="test-qwen-key",       # 主 API Key
        dashscope_api_key="test-dashscope-key",  # 备用 Key（本测试中不应被使用）
        qwen_embedding_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dashscope_embedding_api_base="",
        embedding_size=768,
    )

    # patch: 用 Mock 替换真实的 OpenAIEmbeddings 类，避免发起网络请求
    with patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
        mock_embeddings.return_value = MagicMock()

        result = get_embedding_model(cfg)

        # 验证：OpenAIEmbeddings 被调用了一次，参数为 qwen_api_key
        mock_embeddings.assert_called_once_with(
            model="text-embedding-v4",
            api_key="test-qwen-key",        # ← 应取 qwen_api_key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            dimensions=768,
            check_embedding_ctx_length=False,
        )
        # 函数返回值就是 mock 实例
        assert result == mock_embeddings.return_value


def test_get_embedding_model_qwen_with_dashscope_key() -> None:
    """qwen_api_key 为空时，回退使用 dashscope_api_key。

    这是 get_embedding_model 的 fallback 逻辑：
    qwen_api_key 优先 → 若为空则用 dashscope_api_key。
    """
    cfg = SimpleNamespace(
        embedding_provider="qwen",
        embedding_model_name="text-embedding-v4",
        qwen_api_key="",                    # ← 设为空，触发回退
        dashscope_api_key="test-dashscope-key",
        qwen_embedding_api_base="",
        dashscope_embedding_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        embedding_size=768,
    )

    with patch("langchain_openai.OpenAIEmbeddings") as mock_embeddings:
        mock_embeddings.return_value = MagicMock()

        result = get_embedding_model(cfg)

        # 验证：使用 dashscope_api_key，base_url 也对应切换到 dashscope
        mock_embeddings.assert_called_once_with(
            model="text-embedding-v4",
            api_key="test-dashscope-key",   # ← 因 qwen_api_key 为空，回退到此
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            dimensions=768,
            check_embedding_ctx_length=False,
        )
        assert result == mock_embeddings.return_value


def test_config_accepts_dashscope_api_key(monkeypatch) -> None:
    """验证 Config 类能从环境变量 DASHSCOPE_API_KEY 读取配置。

    monkeypatch 是 pytest 内置 fixture，用于临时修改环境变量。
    测试结束后自动恢复原值，不影响其他测试。
    """
    # 临时设置环境变量（仅在本测试内有效）
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-dashscope-key")

    cfg = Config()

    # 验证 Config 正确读取了环境变量
    assert cfg.dashscope_api_key == "env-dashscope-key"
