"""Fitness AI Coach —— 基于 Streamlit 的前端界面。

页面结构：
- 主区域：对话历史 + 聊天输入框
- 侧边栏：用户设置、快速记录训练、知识库文档上传

数据流：
1. 用户输入问题 → 发送给 agent 后端 → 流式接收 NDJSON → 逐字渲染
2. 用户填写训练表单 → 直接调用 /workout/add（绕过 Agent 推理链路）
3. 用户上传文档 → 调用 /documents/upload → 后端嵌入 Qdrant
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import streamlit as st
from client import AgentClient
from loguru import logger

# Constants
PDF_FILE_TYPE = "pdf"  # 文档上传中 PDF 文件类型的标识

# 模块级客户端单例 —— 整个 Streamlit 会话共享同一个 HTTP 客户端实例
# base_url 通过环境变量 BACKEND_HOST 自动解析（Docker 内用服务名，本地用 localhost）
client = AgentClient()

logger.info("Starting Fitness AI Coach.")

# ---- 页面全局配置 ----
st.set_page_config(page_title="Fitness AI Coach", page_icon=":weightlifter:", layout="wide")

# ---- 页面标题 ----
st.title("weightlifter: Fitness AI Coach")
st.caption("你的智能健身教练 — 记录训练 · 分析进展 · 解答健身问题")


def init_session_state() -> None:
    """初始化 Streamlit 会话状态 —— 在用户首次访问时设置默认值。

    st.session_state 是 Streamlit 的"持久化字典"：
    - 同一个浏览器标签页内，多次 rerun 之间数据保持
    - 刷新页面后保留（除非重启 Streamlit 服务）
    - 不同用户的会话相互隔离

    初始化的字段：
    - messages: 聊天历史列表，格式 [{"role":"user|assistant","content":"..."}]
    - user_id: 用户标识符，默认 "default-user"，可在侧边栏修改
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default-user"


def display_chat_history() -> None:
    """从 st.session_state.messages 中读取历史消息并渲染到聊天界面。

    每次页面 rerun 时调用，Streamlit 的 st.chat_message 自动
    根据 role 渲染不同的气泡样式：
    - "user": 靠右，蓝色气泡
    - "assistant": 靠左，灰色气泡
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


async def handle_chat_async(prompt: str) -> None:
    """以流式方式发送消息，实时显示 AI 思考过程和回答内容。

    这是前端最核心的函数，负责整个对话交互：
    1. 创建回答占位区（message_placeholder）和状态面板（status_container）
    2. 调用 client.chat_stream() 发起流式请求
    3. 根据后端推送的 NDJSON 事件类型分别处理：
       - "status"  → 更新状态提示文字
       - "tool"    → 展示工具调用的开始/完成
       - "content" → 逐 token 追加到回答文字
    4. 流结束后，将完整回答存入 session_state 实现跨 rerun 持久化

    Streamlit 流式渲染原理：
    - st.empty() 创建的容器可反复 .markdown() 覆盖内容
    - 每次覆盖后 Streamlit 自动 diff 并只更新变化的部分
    - "▌" 字符模拟打字时的闪烁光标效果
    """
    with st.chat_message("assistant"):
        # st.empty() 创建一个空白容器，后续反复 .markdown() 实现逐字更新
        message_placeholder = st.empty()
        # 可展开的状态面板，展示工具调用和 AI 思考进度
        status_container = st.status("思考中...", expanded=True)

        try:
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            # 累积流式返回的回答文本
            full_answer = ""

            async for line in client.chat_stream(
                messages=messages,
                user_id=st.session_state.user_id,
                collection_name=st.session_state.get("collection_name", "fitness_kb"),
            ):
                event = json.loads(line)
                event_type = event.get("type")

                if event_type == "status":
                    # 更新状态栏文字
                    status_container.write(f"🔄 {event['data']}")

                elif event_type == "tool":
                    # 展示工具调用详情
                    tool_info = event["data"]
                    if tool_info.get("status") == "start":
                        status_container.write(f"🔧 调用工具: **{tool_info['tool']}**")
                    elif tool_info.get("status") == "end":
                        status_container.write(f"✅ 工具 **{tool_info['tool']}** 完成")

                elif event_type == "content":
                    # 逐 token 追加到回答
                    full_answer += event["data"]
                    message_placeholder.markdown(full_answer + "▌")

            # 流结束，显示最终回答（去掉光标）
            if full_answer:
                message_placeholder.markdown(full_answer)
            else:
                message_placeholder.markdown("抱歉，我没有生成回复。")

            status_container.update(label="完成", state="complete", expanded=False)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})

        except httpx.HTTPStatusError as err:
            status_container.update(label="错误", state="error")
            st.error(f"服务端错误: {err}")
            logger.error(f"HTTP error: {err}")
        except Exception as e:
            status_container.update(label="错误", state="error")
            st.error(f"发生错误: {e}")
            logger.error(f"Unexpected error: {e}")


def handle_chat(prompt: str) -> None:
    """同步包装器 —— Streamlit 的事件回调是同步上下文，用 asyncio.run 桥接异步。

    Streamlit 的按钮、输入框等回调运行在同步上下文中，不能直接 await。
    asyncio.run() 创建临时事件循环来执行异步函数，执行完毕后自动清理。
    """
    asyncio.run(handle_chat_async(prompt))


async def handle_workout_quick_add(exercises: list[dict]) -> None:
    """将侧边栏填写的训练动作列表提交到后端 /workout/add。

    此路径绕过 Agent 推理链路，由前端直接调用后端 API。
    后端会解析动作名称、提取肌肉部位类别、写入 WorkoutSession。

    Args:
        exercises: 训练动作列表，每项包含 name/sets/reps/weight_kg/set_details/notes
    """
    if not exercises:
        st.warning("请先添加至少一个训练动作。")
        return
    try:
        result = await client.add_workout(
            user_id=st.session_state.user_id,
            exercises=exercises,
        )
        if result.get("status") == "success":
            # parsed 字段包含后端成功解析的动作数量
            st.success(f"已记录 {len(result.get('parsed', []))} 项训练！💪")
        else:
            st.warning(f"记录失败，请重试。")
    except Exception as e:
        st.error(f"记录失败: {e}")


def sidebar() -> None:
    """侧边栏 —— 三个功能区域：设置、快速记录训练、知识库上传。

    整个侧边栏在每次页面 rerun 时重新执行。Streamlit 通过 st.session_state
    在 rerun 之间保持状态，实现"看似有记忆"的交互体验。

    布局：
    - 区域一：用户设置（ID、知识库集合名称）
    - 区域二：快速记录训练（结构化表单 → 发送到 /workout/add）
    - 区域三：知识库上传（文件选择 → 发送到 /documents/upload）
    """
    with st.sidebar:
        # =====================
        #  区域一：用户设置
        # =====================
        st.header(":gear: 设置")

        # 用户 ID —— 用于后端数据隔离和鉴权
        st.session_state.user_id = st.text_input(
            "用户 ID",
            value=st.session_state.user_id,
            help="用于区分不同用户的训练数据",
        )

        # RAG 检索使用的 Qdrant 集合名称
        st.session_state.collection_name = st.text_input(
            "知识库",
            value="fitness_kb",
            help="Qdrant 知识库集合名称",
        )

        st.divider()

        # =====================
        #  区域二：快速记录训练
        # =====================
        st.header("pencil: 快速记录训练")

        # pending_exercises: 用户在当前会话中已添加但尚未提交的训练动作列表
        # 每次添加到列表后 st.rerun() 刷新页面，列表内容靠 session_state 保持
        if "pending_exercises" not in st.session_state:
            st.session_state.pending_exercises = []

        # 动作名称 + 组数（横向排列）
        col1, col2 = st.columns(2)
        with col1:
            ex_name = st.text_input("动作名称", placeholder="深蹲", key="ex_name")
        with col2:
            ex_sets = st.number_input("组数", min_value=1, max_value=20, value=3, key="ex_sets")

        # ---- 动态输入：按组数生成次数和重量输入框 ----
        # Streamlit 没有"动态 input 数量"的 API，所以用 st.columns 模拟。
        # 根据组数分为两种模式：
        #   ≤5 组：每列一组，横向展开
        #   >5 组：每行 5 列，换行排列
        st.caption("每组次数与重量：")
        reps_per_set: list[int] = []
        weights_per_set: list[float] = []
        if ex_sets <= 5:
            # ≤5 组：直接按组数创建等宽列，每组一列横向展开
            set_cols = st.columns(ex_sets)
            for i in range(ex_sets):
                with set_cols[i]:
                    st.caption(f"第{i+1}组")
                    r = st.number_input(
                        "次数", min_value=0, max_value=100, value=10,
                        key=f"rep_{i}"
                    )
                    w = st.number_input(
                        "重量kg", min_value=0.0, max_value=999.0, value=0.0, step=2.5,
                        key=f"wt_{i}"
                    )
                    reps_per_set.append(r)
                    weights_per_set.append(w)
        else:
            # >5 组：每行固定 5 列，超出部分换行
            # (ex_sets + 4) // 5 计算总行数（向上取整）
            rows = (ex_sets + 4) // 5
            for row in range(rows):
                cols = st.columns(5)
                for col_idx in range(5):
                    i = row * 5 + col_idx
                    if i < ex_sets:  # 最后一行可能不满 5 列
                        with cols[col_idx]:
                            st.caption(f"第{i+1}组")
                            r = st.number_input(
                                "次数", min_value=0, max_value=100, value=10,
                                key=f"rep_{i}"
                            )
                            w = st.number_input(
                                "重量kg", min_value=0.0, max_value=999.0, value=0.0, step=2.5,
                                key=f"wt_{i}"
                            )
                            reps_per_set.append(r)
                            weights_per_set.append(w)

        # ---- 汇总计算 ----
        # 总次数（所有组次数之和）
        total_reps = sum(reps_per_set)
        unique_reps = set(reps_per_set)
        unique_weights = set(weights_per_set)
        # 平均重量：仅统计非零项（0 表示自重训练，不参与均值计算）
        nonzero_weights = [w for w in weights_per_set if w > 0]
        avg_weight = round(sum(nonzero_weights) / len(nonzero_weights), 1) if nonzero_weights else 0.0

        # ---- 汇总文字显示 ----
        # 根据"每组次数是否相同"和"每组重量是否相同"分四种情况处理显示文案
        if len(unique_reps) == 1 and len(unique_weights) == 1:
            # 每组次数重量都一样，简化为"共 X 次，Ykg"
            w_str = f"{weights_per_set[0]}kg" if weights_per_set[0] > 0 else "自重"
            st.caption(f"共 {total_reps} 次，{w_str}")
        elif len(unique_reps) == 1 and len(unique_weights) > 1:
            # 次数相同但重量逐组不同
            st.caption(f"共 {total_reps} 次，重量 {', '.join(f'{w}kg' for w in weights_per_set)}")
        elif len(unique_reps) > 1 and len(unique_weights) == 1:
            # 重量相同但次数逐组不同
            w_str = f"{weights_per_set[0]}kg" if weights_per_set[0] > 0 else "自重"
            st.caption(f"次数 {', '.join(str(r) for r in reps_per_set)}，共 {total_reps} 次，{w_str}")
        else:
            # 次数和重量都不同，分两行显示
            st.caption(f"次数 {', '.join(str(r) for r in reps_per_set)}，共 {total_reps} 次")
            st.caption(f"重量 {', '.join(f'{w}kg' for w in weights_per_set)}")

        # ---- 构建备注文本 ----
        # 仅当每组次数或重量不一致时才记录明细（相同的话不需要备注，汇总就够了）
        notes_parts = []
        if len(unique_reps) > 1:
            notes_parts.append(f"每组次数:{','.join(str(r) for r in reps_per_set)}")
        if len(unique_weights) > 1:
            notes_parts.append(f"每组重量:{','.join(str(w) for w in weights_per_set)}")
        notes = "; ".join(notes_parts) if notes_parts else None

        # ---- 构建每组明细 ----
        # 次数与重量按组一一对应，存入 set_details 字段
        set_details = [
            {"reps": r, "weight_kg": w}
            for r, w in zip(reps_per_set, weights_per_set)
        ]

        # ---- "添加到列表"按钮 ----
        # 点击后将当前动作存入 pending_exercises，然后 rerun 刷新列表显示
        if st.button("➕ 添加到列表", use_container_width=True):
            if ex_name.strip():
                st.session_state.pending_exercises.append({
                    "name": ex_name.strip(),
                    "sets": ex_sets,
                    "reps": total_reps,
                    "weight_kg": avg_weight,
                    "set_details": set_details,
                    "notes": notes,
                })
                st.rerun()  # 触发页面刷新，清空输入框 + 显示新添加的动作
            else:
                st.warning("请输入动作名称")

        # ---- 已添加动作列表 ----
        # 显示所有待提交的动作，每行：名称 | 组数+详情 | 重量 | 删除按钮
        if st.session_state.pending_exercises:
            st.caption(f"已添加 {len(st.session_state.pending_exercises)} 个动作：")
            for i, ex in enumerate(st.session_state.pending_exercises):
                cols = st.columns([4, 2, 2, 1])  # 四列布局：名称占宽，删除按钮占窄
                with cols[0]:
                    st.text(ex["name"])
                with cols[1]:
                    # 有备注则显示备注（差异化组详情），否则只显示总次数
                    detail = ex["notes"] if ex["notes"] else f"{ex['reps']}次"
                    st.text(f"{ex['sets']}组 {detail}")
                with cols[2]:
                    st.text(f"{ex['weight_kg']}kg" if ex['weight_kg'] > 0 else "自重")
                with cols[3]:
                    # 删除按钮：从列表中移除该动作并刷新
                    if st.button("🗑", key=f"del_{i}"):
                        st.session_state.pending_exercises.pop(i)
                        st.rerun()

            # ---- "记录训练"按钮 ----
            # 将整个 pending_exercises 列表提交到后端，完成后清空并刷新
            if st.button("💾 记录训练", use_container_width=True):
                # .copy() 防止异步过程中列表被修改
                asyncio.run(handle_workout_quick_add(
                    st.session_state.pending_exercises.copy()
                ))
                st.session_state.pending_exercises = []
                st.rerun()

        st.divider()

        # =====================
        #  区域三：知识库文档上传
        # =====================
        st.header(":books: 知识库上传")

        # 目标 Qdrant 集合名称
        collection_name = st.text_input("集合名称", value="fitness_kb", key="upload_collection")

        # 文档类型选择 —— 影响文件选择器的过滤范围和 UI 提示
        file_ending = st.selectbox("文档类型", options=[".pdf", ".txt", ".md"])

        # 文件选择器：type 参数限制用户可见的文件类型
        uploaded_files = st.file_uploader(
            "选择文件",
            type=["pdf"] if file_ending == ".pdf" else ["txt", "md"],
            accept_multiple_files=True,  # 支持批量选择
        )

        if st.button("上传文档", use_container_width=True):
            if uploaded_files:
                # st.spinner 显示加载动画，提示用户正在处理中
                with st.spinner("上传并嵌入文档..."):
                    try:
                        # httpx 需要的文件格式: [("files", (filename, file_bytes, mime_type)), ...]
                        files = [
                            ("files", (file.name, file, file.type))
                            for file in uploaded_files
                        ]
                        # 调用后端 /documents/upload，同步等待完成
                        asyncio.run(
                            client.upload_documents(
                                files=files,
                                collection_name=collection_name,
                                category="general",
                            )
                        )
                        st.success(f"成功上传 {len(uploaded_files)} 个文件！")
                    except httpx.HTTPStatusError as err:
                        st.error(f"上传失败: {err}")
                    except Exception as e:
                        st.error(f"发生错误: {e}")
            else:
                st.warning("请选择至少一个文件。")


def initialize() -> None:
    """应用主入口 —— 初始化会话 → 渲染侧边栏 → 渲染聊天历史 → 监听聊天输入。

    Streamlit 的执行模型：整个脚本从上到下每次 rerun 都重新执行一次。
    initialize() 在每次 rerun 时被调用，但 session_state 中的数据得以保留，
    所以用户看到的聊天历史和侧边栏表单不会丢失。

    执行顺序：
    1. init_session_state()   → 确保新用户有默认的 session 状态
    2. sidebar()              → 渲染侧边栏的 UI 和逻辑
    3. display_chat_history() → 渲染主区域的对话气泡
    4. st.chat_input()        → 主区域底部聊天输入框（阻塞等待用户输入）
    """
    init_session_state()
    sidebar()
    display_chat_history()

    # st.chat_input 在当前没有输入时返回 None，页面渲染完后"安静等待"
    # 用户按回车后，触发 rerun，这里的 prompt 就有值了，进入 if 分支
    if prompt := st.chat_input("你想了解或记录什么？"):
        # 将用户消息立即渲染（assistant 侧气泡）
        st.chat_message("user").markdown(prompt)
        # 存入会话历史，供下一轮对话传给后端
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 发起 Agent 对话
        handle_chat(prompt)


if __name__ == "__main__":
    initialize()
