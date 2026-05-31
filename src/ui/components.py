"""Streamlit UI component renderers for each HITL stage.

Each render function corresponds to one graph stage:
  - render_sidebar            → progress bar + stage indicator in sidebar
  - render_chat_interface     → NeedsAlignment multi-turn chat
  - render_profile_approval   → LearnerProfile confirmation
  - render_stage_indicator    → 5-step pipeline progress bar
  - render_knowledge_tree     → KnowledgeTree approval
  - render_chapter_plan       → ChapterPlan approval
  - render_review_issues      → Reviewer feedback + retry/accept buttons

All render functions accept a state dict + callbacks (on_approve, on_retry, etc.).
"""
import streamlit as st
import json


def render_sidebar(session_state: dict):
    """Render sidebar with overall progress and navigation."""
    st.sidebar.title("🎨 BlogGen")

    tree = session_state.get("knowledge_tree", {})
    posts = session_state.get("posts", [])
    current_idx = session_state.get("current_post_index", 0)
    completed = session_state.get("completed_posts", [])
    stage = session_state.get("stage", "needs_alignment")

    st.sidebar.markdown("---")

    # Post count from split plan, fall back to knowledge tree topic count
    post_count = len(posts) if posts else len(tree.get("topics", []))
    done_count = len(completed)
    if post_count > 0:
        st.sidebar.markdown(f"### 📋 进度：{done_count}/{post_count}")
        for i in range(post_count):
            if i < done_count:
                emoji = "✅"
            elif i == current_idx and stage not in ("done",):
                emoji = "🔄"
            else:
                emoji = "⬜"
            if posts:
                label = posts[i].get("title", f"第{i+1}篇")
            else:
                kt_topics = tree.get("topics", []) if tree else []
                label = kt_topics[i] if i < len(kt_topics) else f"第{i+1}篇"
            st.sidebar.markdown(f"{emoji} {label[:25]}")
    else:
        st.sidebar.markdown("*等待知识树生成...*")

    st.sidebar.markdown("---")
    stage_names = {
        "needs_alignment": "需求对齐",
        "needs_done": "需求确认",
        "knowledge_tree_done": "知识树",
        "chapter_plan_done": "章节规划",
        "writer_done": "写作中",
        "review_reject": "审查修改",
        "review_pass": "审查通过",
        "done": "已完成",
    }
    st.sidebar.caption(f"阶段：{stage_names.get(stage, stage)}")


def render_chat_interface(session_state: dict, on_user_message):
    """Render the chat interface for needs alignment phase."""
    st.markdown("## 📝 需求对齐")

    # Display chat history (messages may be dicts or LangChain objects)
    messages = session_state.get("messages", [])
    # Show greeting if no messages yet
    if not messages:
        greeting = (
            "你好！我是你的学习博客生成助手 👋\n\n"
            "为了给你量身定制最适合的学习博客，请告诉我以下信息：\n\n"
            "📚 **学习领域**：你想学什么？（如：AI应用开发、系统设计...）\n"
            "🎯 **当前水平**：初学者 / 有一定基础 / 进阶高手\n"
            "🏁 **学习目标**：学到什么程度？为了什么目的？（如：通过面试、落地项目...）\n"
            "⏰ **时间约束**：（可选）每天能投入多少学习时间？\n"
            "🎨 **风格偏好**：（可选）偏实战 / 偏理论 / 两者平衡\n\n"
            "你可以直接回复，也可以多写几句说明你的具体情况。"
        )
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(greeting)

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", "") or getattr(msg, "type", "")
            content = getattr(msg, "content", "")
        if role == "assistant" or role == "ai":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
        elif role == "user" or role == "human":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)

    # Chat input
    user_input = st.chat_input("输入你的回答...")
    if user_input:
        on_user_message(user_input)


def render_profile_approval(profile: dict, on_approve, on_edit):
    """Render the learner profile for approval."""
    st.markdown("## ✅ 学习者画像确认")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📚 **学习领域**")
        st.write(profile.get("domain", ""))
        st.markdown("🎯 **当前水平**")
        st.write(profile.get("level", ""))
    with col2:
        st.markdown("🏁 **学习目标**")
        st.write(profile.get("goal", ""))
        st.markdown("🎨 **风格偏好**")
        st.write(profile.get("style", "balanced"))

    col_a, col_b = st.columns(2)
    with col_a:
        st.button("✅ 确认，继续规划", on_click=on_approve, use_container_width=True)
    with col_b:
        st.button("✏️ 需要修改", on_click=on_edit, use_container_width=True)


def render_stage_indicator(stage: str):
    """Render the 5-agent pipeline stage indicator."""
    stages = [
        ("needs_alignment", "需求对齐"),
        ("knowledge_tree", "知识树"),
        ("chapter_plan", "章节规划"),
        ("writing", "写作审校"),
        ("done", "完成"),
    ]

    stage_to_idx = {
        "needs_alignment": 0, "needs_alignment_done": 0, "needs_done": 0,
        "knowledge_tree_done": 1,
        "chapter_plan_done": 2,
        "writer_done": 3, "review_reject": 3, "review_pass": 3,
        "done": 4,
    }
    current_idx = stage_to_idx.get(stage, 0)

    cols = st.columns(len(stages))
    for i, (s, label) in enumerate(stages):
        with cols[i]:
            if i < current_idx:
                st.markdown("✅")
            elif i == current_idx:
                st.markdown("🔄")
            else:
                st.markdown("⬜")
            st.caption(label)

    # Add assembler_done to the writing stage
    stage_to_idx["assembler_done"] = 3


def render_knowledge_tree(st_data: dict, on_approve, on_regenerate):
    """Render the learning path (flat topic list) and approval UI."""
    tree = st_data.get("knowledge_tree", {})
    topics = tree.get("topics", [])
    total = len(topics)

    st.markdown("## 📋 学习路径")
    st.caption(f"领域：{tree.get('domain', '')} | 共 {total} 个知识点")

    if topics:
        for i, topic in enumerate(topics):
            st.markdown(f"{i+1}. {topic}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.button("✅ 确认路径，开始写作", on_click=on_approve, use_container_width=True)
    with col_b:
        st.button("✏️ 重新生成", on_click=on_regenerate, use_container_width=True)


def render_chapter_plan(st_data: dict, on_approve, on_regenerate):
    """Render the chapter plan, split info, and approval UI."""
    plan = st_data.get("chapter_plan", {})
    chapters = plan.get("chapters", [])
    posts = st_data.get("posts", [])
    current_idx = st_data.get("current_post_index", 0)

    st.markdown(f"## 📝 章节规划")

    # Show split info if multi-post
    if len(posts) > 1:
        st.info(f"共 {len(chapters)} 章，自动分为 **{len(posts)} 篇** 博客")
        for i, p in enumerate(posts):
            marker = "🔄" if i == current_idx else "⬜"
            indices = p.get("chapter_indices", [])
            p_chapters = [chapters[j] for j in indices if j < len(chapters)]
            est = p.get("word_estimate", 0)
            with st.expander(f"{marker} 第{i+1}篇：{p.get('title', '')}（{len(p_chapters)}章，约{est}字）", expanded=(i==current_idx)):
                for ch in p_chapters:
                    st.markdown(f"- **{ch.get('title', '')}**")
                    for pt in ch.get("key_points", []):
                        st.markdown(f"  - {pt}")

    # Show all chapters
    for i, ch in enumerate(chapters):
        post_label = ""
        if len(posts) > 1:
            for pi, p in enumerate(posts):
                if i in p.get("chapter_indices", []):
                    post_label = f" [第{pi+1}篇]"
                    break
        with st.container():
            st.markdown(f"### 第{i+1}章：{ch.get('title', '')}{post_label}")
            for pt in ch.get("key_points", []):
                st.markdown(f"- {pt}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.button("✅ 确认，开始写作", on_click=on_approve, use_container_width=True)
    with col_b:
        st.button("✏️ 调整章节", on_click=on_regenerate, use_container_width=True)


def render_review_issues(st_data: dict, on_retry, on_accept):
    """Render reviewer issues and retry/accept buttons."""
    retries = st_data.get("writer_retry_count", 0)
    feedback = st_data.get("review_feedback", {})
    issues = feedback.get("issues", [])

    st.warning(f"⚠️ 审查未通过（第 {retries} 次修改）")

    if issues:
        st.markdown("### 审查意见")
        for issue in issues:
            if isinstance(issue, dict):
                severity_icon = "🔴" if issue.get("severity") == "critical" else "🟡"
                st.markdown(f"{severity_icon} **{issue.get('type', '')}** — {issue.get('paragraph', '')}")
                st.caption(issue.get("description", ""))
                st.caption(f"建议：{issue.get('suggestion', '')}")

    col_a, col_b = st.columns(2)
    with col_a:
        if retries < 2:  # MAX_REVIEW_RETRIES
            st.button("🔄 根据意见修改", on_click=on_retry, use_container_width=True)
        else:
            st.button("✅ 接受当前版本", on_click=on_accept, use_container_width=True)
    with col_b:
        st.button("👀 手动审阅后决定", on_click=on_accept, use_container_width=True)
