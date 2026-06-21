"""端到端（E2E）API 测试 —— 用 TestClient 对 FastAPI 应用发起模拟 HTTP 请求。

与 unit_tests 的区别：
- unit_tests: 直接 import 函数，传入参数，断言返回值（不经过 HTTP 层）
- e2e_tests: 通过 TestClient 模拟 HTTP 请求，走完整的路由→处理→响应链路

TestClient 的工作原理：
1. conftest.py 的 app fixture 启动 FastAPI 应用（patch 掉真实的 Qdrant/Phoenix 连接）
2. client fixture 用 app 创建 TestClient 实例
3. TestClient 不绑端口，直接通过 ASGI 内存协议与应用通信
4. 无需启动真实服务器，但行为等价于 HTTP 请求
"""

from __future__ import annotations

from http import HTTPStatus

import pytest

# 将本文件内的所有测试标记为 integration 类型
# 运行 pytest -m integration 时只执行这些测试
# 运行 pytest -m "not integration" 时跳过它们
pytestmark = pytest.mark.integration


def test_read_root(client) -> None:
    """测试根路径健康检查。

    验证点：
    - GET / 返回 200 OK
    - 应用能正常启动并响应（TestClient 不走真实网络，通过 ASGI 内存协议通信）

    client 参数由 conftest.py 的 client fixture 自动注入，
    是一个已启动的 TestClient 实例。
    """
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
