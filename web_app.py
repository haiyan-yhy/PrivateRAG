"""Streamlit web UI for the RAG knowledge base."""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

import streamlit as st

# Allow imports from the same directory
sys.path.insert(0, str(Path(__file__).parent))

import config
from llm import build_prompt, get_llm_and_embedding
from llama_index.core import Settings

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG 知识库",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG 知识库问答")
st.caption(f"Provider: `{config.LLM_PROVIDER}` · Top-K: `{config.TOP_K}`")


# ── Cached resource: retriever ─────────────────────────────────────────────
@st.cache_resource(show_spinner="加载向量库...")
def load_retriever():
    from retriever import Retriever
    return Retriever()


# ── Sidebar: document management ──────────────────────────────────────────
with st.sidebar:
    st.header("文档管理")

    uploaded = st.file_uploader(
        "上传 PDF 或 Markdown 文件",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("写入 data/ 并重建索引", type="primary"):
        config.DATA_DIR.mkdir(exist_ok=True)
        for f in uploaded:
            dest = config.DATA_DIR / f.name
            dest.write_bytes(f.read())
            st.success(f"已保存：{f.name}")

        with st.spinner("正在建立索引，请稍候..."):
            # Clear old vectorstore so ingest rebuilds from scratch
            if config.VECTOR_DIR.exists():
                shutil.rmtree(config.VECTOR_DIR)
            from ingest import ingest
            ingest()
            # Invalidate cached retriever
            load_retriever.clear()

        st.success("索引构建完成！")
        st.rerun()

    st.divider()

    # List files currently in data/
    st.subheader("当前文档")
    if config.DATA_DIR.exists():
        files = list(config.DATA_DIR.glob("*.pdf")) + list(config.DATA_DIR.glob("*.md"))
        if files:
            for f in files:
                st.text(f"• {f.name}")
        else:
            st.caption("data/ 目录为空")
    else:
        st.caption("data/ 目录不存在")

    st.divider()
    debug_mode = st.toggle("Debug 模式（显示检索 chunks）", value=config.DEBUG)


# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chunks") and debug_mode:
            with st.expander(f"检索到 {len(msg['chunks'])} 个 chunks"):
                for i, c in enumerate(msg["chunks"], 1):
                    st.markdown(f"**[{i}] {c['source']}** · score: `{c['score']:.4f}`")
                    st.code(c["text"], language=None)


# ── Chat input ─────────────────────────────────────────────────────────────
query = st.chat_input("输入你的问题...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("思考中...")

        try:
            retriever = load_retriever()
            chunks = retriever.retrieve(query)

            if not chunks:
                reply = "我无法从当前知识库中找到答案。"
                chunk_data = []
            else:
                context = "\n\n".join(
                    f"[{i}] {c.text}" for i, c in enumerate(chunks, 1)
                )
                prompt = build_prompt(query=query, context=context)
                response = Settings.llm.complete(prompt)
                reply = str(response).strip()

                sources = list(dict.fromkeys(c.source for c in chunks))
                source_line = "【来源】" + "、".join(sources)
                if source_line not in reply:
                    reply = f"{reply}\n\n{source_line}"

                chunk_data = [
                    {"text": c.text, "source": c.source, "score": c.score}
                    for c in chunks
                ]

        except Exception as exc:
            reply = f"❌ 出错了：{exc}\n\n请确认已运行 `python ingest.py` 建立索引。"
            chunk_data = []

        placeholder.markdown(reply)

        if chunk_data and debug_mode:
            with st.expander(f"检索到 {len(chunk_data)} 个 chunks"):
                for i, c in enumerate(chunk_data, 1):
                    st.markdown(f"**[{i}] {c['source']}** · score: `{c['score']:.4f}`")
                    st.code(c["text"], language=None)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "chunks": chunk_data}
    )
