"""BlogGen — Multi-Agent Collaborative Blog Generation System.

5-agent LangGraph pipeline:
  1. NeedsAlignment  — chat-based requirements collection → LearnerProfile
  2. KnowledgeTree    — research + topic listing → ordered knowledge tree
  3. ChapterPlanner   — groups topics into chapters → ChapterPlan
  4. Writer           — writes each chapter (fan-out), then assembles
  5. Reviewer         — checklist review per chapter + cross-chapter structure check

HITL (human-in-the-loop) checkpoints after each agent except Writer.
Writer→Reviewer loop with MAX_REVIEW_RETRIES for quality gate.

Entry point: streamlit run main.py
"""
import streamlit as st
from src.graph.session import BlogGenSession
from src.ui.components import (
    render_sidebar,
    render_chat_interface,
    render_profile_approval,
    render_stage_indicator,
    render_knowledge_tree,
    render_chapter_plan,
    render_review_issues,
)
from src.monitor import render_monitor_sidebar


# ============================================================
# Session helpers
# ============================================================

def init_session():
    if "bloggen" not in st.session_state:
        st.session_state.bloggen = BlogGenSession()
        st.session_state.bloggen.create()
        _seed_knowledge_base()
    if "app_stage" not in st.session_state:
        snap = st.session_state.bloggen.get_state()
        st.session_state.app_stage = snap.get("stage", "needs_alignment")


def S() -> BlogGenSession:
    return st.session_state.bloggen


def state() -> dict:
    return S().get_state()


def sync_stage():
    st.session_state.app_stage = state().get("stage", "needs_alignment")


# ============================================================
# Graph execution helpers
# ============================================================

def run_graph():
    """Run graph until next interrupt or completion."""
    with st.spinner("Agent 工作中..."):
        try:
            result = S().invoke()
            sync_stage()
        except Exception as e:
            st.error(f"Graph execution error: {e}")
            sync_stage()


def resume_graph():
    """Resume from interrupt_after pause. Calls invoke(None) to continue."""
    run_graph()


def _seed_knowledge_base():
    """Import local markdown knowledge files into vector store (one-time per session)."""
    from pathlib import Path
    import logging
    logger = logging.getLogger("BlogGen")
    project_root = Path(__file__).parent
    seed_files = list(project_root.glob("*.md"))
    seed_files = [f for f in seed_files if f.name not in ("CLAUDE.md", "README.md")]
    for fp in seed_files:
        try:
            from src.rag import seed_from_markdown
            n = seed_from_markdown(str(fp))
            if n > 0:
                logger.info(f"Seeded {n} chunks from {fp.name}")
        except Exception as e:
            logger.warning(f"Failed to seed from {fp.name}: {e}")


# ============================================================
# HITL handlers
# ============================================================

def handle_user_message(user_input: str):
    """NeedsAlignment multi-turn chat. Each message drives one graph invocation."""
    s = state()
    messages = list(s.get("messages", []))
    messages.append({"role": "user", "content": user_input})
    S().update_state({"messages": messages})
    run_graph()
    st.rerun()


def handle_approve_profile():
    resume_graph()


def handle_approve_tree():
    resume_graph()


def handle_approve_chapter_plan():
    resume_graph()


def handle_retry_writer():
    resume_graph()


def handle_accept_as_is():
    s = state()
    draft = s.get("draft", s.get("assembled_draft", ""))
    S().update_state({
        "final": draft,
        "review_result": {"action": "accept", "overall_assessment": "用户强制接受"},
        "stage": "review_pass",
    })
    sync_stage()
    resume_graph()  # Let graph route to next_post or END


def handle_approve_final():
    s = state()
    posts_to_save = [{
        "title": s.get("current_post_title", ""),
        "content": s.get("final", s.get("draft", "")),
    }]
    completed = list(s.get("completed_posts", []))
    completed.extend(posts_to_save)
    S().update_state({"completed_posts": completed})
    sync_stage()
    # Keep stage as review_pass — let the graph route to next_post or END
    resume_graph()


def handle_regenerate_tree():
    S().update_state({"stage": "needs_alignment_done", "knowledge_tree": {}})
    run_graph()


def handle_regenerate_chapter_plan():
    S().update_state({"stage": "knowledge_tree_done", "chapter_plan": {}})
    run_graph()


# ============================================================
# Auto-run logic
# ============================================================

def maybe_auto_run():
    """Auto-advance through non-HITL internal stages.

    HITL stages (needs_alignment_done, knowledge_tree_done, chapter_plan_done,
    review_reject, review_pass) are rendered by the UI and wait for user input.
    This only auto-advances through internal stages like writer_done and
    assembler_done — these are invisible relay points, not user-facing screens.

    Called from the main script body (not a callback), so st.rerun() is OK here.
    """
    s = state()
    stage = s.get("stage", "")

    auto_stages = {"writer_done", "assembler_done"}

    if stage in auto_stages:
        run_graph()
        st.rerun()


# ============================================================
# Regenerate / edit handlers
# ============================================================

def handle_edit_profile():
    S().update_state({"stage": "needs_alignment", "user_needs": {}})
    sync_stage()


# ============================================================
# Main App
# ============================================================

st.set_page_config(
    page_title="BlogGen — 多Agent协作博客生成",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 BlogGen — 多 Agent 协作博客生成器")
init_session()

s_data = state()
app_stage = st.session_state.app_stage

render_sidebar(s_data)
render_monitor_sidebar()
render_stage_indicator(app_stage)
st.markdown("---")

# Auto-run non-HITL transitions
maybe_auto_run()

# ================================================================
# Stage routing
# ================================================================

if app_stage == "needs_alignment":
    render_chat_interface(s_data, handle_user_message)

elif app_stage in ("needs_alignment_done", "needs_done"):
    profile = s_data.get("user_needs", {})
    if profile:
        render_profile_approval(profile, on_approve=handle_approve_profile, on_edit=handle_edit_profile)
    else:
        st.warning("等待需求分析...")

elif app_stage == "knowledge_tree_done":
    render_knowledge_tree(s_data, on_approve=handle_approve_tree, on_regenerate=handle_regenerate_tree)

elif app_stage == "chapter_plan_done":
    render_chapter_plan(s_data, on_approve=handle_approve_chapter_plan, on_regenerate=handle_regenerate_chapter_plan)

elif app_stage == "review_reject":
    render_review_issues(s_data, on_retry=handle_retry_writer, on_accept=handle_accept_as_is)

elif app_stage == "review_pass":
    final = s_data.get("final", s_data.get("draft", ""))
    post_title = s_data.get("current_post_title", "博客")
    word_count = len(final)
    review = s_data.get("review_result", {})

    st.success(f"✅ 审查通过（{word_count} 字）")
    if review.get("overall_assessment"):
        st.caption(review["overall_assessment"])

    with st.container(height=500):
        st.markdown(final)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.button("✅ 确认发布", on_click=handle_approve_final, use_container_width=True)
    with col_b:
        if "show_edit_area" not in st.session_state:
            st.session_state.show_edit_area = False
        if st.button("✏️ 手动修改" if not st.session_state.show_edit_area else "🔽 收起编辑",
                     use_container_width=True):
            st.session_state.show_edit_area = not st.session_state.show_edit_area
    with col_c:
        st.download_button("💾 下载", data=final, file_name=f"{post_title}.md",
                          mime="text/markdown", use_container_width=True)

    if st.session_state.get("show_edit_area"):
        edited = st.text_area("编辑内容", value=final, height=400, key="edit_draft_area")
        if st.button("💾 保存修改"):
            S().update_state({"final": edited, "draft": edited})
            st.success("已保存")
            st.session_state.show_edit_area = False
            st.rerun()

elif app_stage == "done":
    st.balloons()
    st.success("🎉 全部博客已生成！")
    completed = s_data.get("completed_posts", [])
    for i, post in enumerate(completed):
        with st.expander(f"✅ {post.get('title', f'第{i+1}篇')}"):
            st.markdown(post.get("content", "")[:500] + "...")
            st.download_button(
                f"下载第{i+1}篇",
                data=post.get("content", ""),
                file_name=f"{post.get('title', f'post_{i+1}')}.md",
                mime="text/markdown",
            )

elif app_stage == "split_error" or app_stage.endswith("_error") or s_data.get("_error"):
    error_msg = s_data.get("_error", "未知错误")
    st.error(f"❌ 节点执行失败：{app_stage}")
    st.caption(f"错误信息：{error_msg}")
    if st.button("🔄 重新开始", use_container_width=True):
        S().create()
        sync_stage()
        st.rerun()

else:
    # writer_done, assembler_done, or unknown — graph should auto-advance
    st.info(f"当前阶段：{app_stage}")
    if st.button("🔄 继续", use_container_width=True):
        run_graph()
        st.rerun()
