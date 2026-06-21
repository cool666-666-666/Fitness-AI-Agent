"""该脚本包含 REST API 响应的 Pydantic 数据模型。

本模块定义了 FastAPI 各接口返回的响应体结构。
路由处理函数返回这些模型的实例后，FastAPI 会自动将其序列化为 JSON 返回给前端。
"""

from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    """操作状态的枚举类。

    继承自 str + Enum，既可以用作字符串比较，又具备枚举的类型安全性。
    """

    # 操作成功完成。
    SUCCESS = "success"
    # 操作执行失败。
    FAILURE = "failure"


class EmbeddingResponse(BaseModel):
    """文本嵌入端点的响应模型。

    返回嵌入操作的结果状态以及成功处理的文件列表。
    """

    # 嵌入操作的整体状态。默认值为 SUCCESS，仅在出现异常时返回 FAILURE。
    status: Status = Field(Status.SUCCESS, title="Status", description="The status of the request.")

    # 本次成功完成嵌入的文件名列表，用于告知前端哪些文件已入库。
    files: list[str] = Field([], title="Files", description="The list of files that were embedded.")


