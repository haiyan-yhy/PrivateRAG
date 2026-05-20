"""CLI entry point for the RAG knowledge base."""

from __future__ import annotations

import logging
import sys

import config
from llm import build_chat_messages, get_llm_and_embedding
from retriever import Retriever
from reranker import Reranker
from router import router_query

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def answer(query: str, retriever: Retriever, reranker: Reranker, history: list[dict] | None = None) -> str:
    history = history or []
    chunks = retriever.retrieve(query)
    for c in chunks:
        logger.debug("原始检索结果：source=%s score=%.4f\n%s\n", c.source, c.score, c.text[:200])
    chunks = reranker.rerank(query, chunks, top_n=config.RERANK_TOP_N)
    for c in chunks:
        logger.debug("重新排序后：source=%s\n%s\n", c.source, c.text[:200])

    if not chunks:
        return "我无法从当前知识库中找到答案。"

    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] {chunk.text}")
        if chunk.source not in sources:
            sources.append(chunk.source)

    context = "\n\n".join(context_parts)

    from llama_index.core import Settings
    get_llm_and_embedding()
    messages = build_chat_messages(query=query, context=context, history=history)
    response = Settings.llm.chat(messages)
    reply = str(response.message.content).strip()

    source_line = "【来源】" + "、".join(sources)
    if source_line not in reply:
        reply = f"{reply}\n\n{source_line}"

    return reply


def main() -> None:
    print("=" * 60)
    print("  RAG 知识库问答系统")
    print(f"  Provider : {config.LLM_PROVIDER}")
    print(f"  Top-K    : {config.TOP_K}")
    print(f"  Debug    : {config.DEBUG}")
    print("  输入 'exit' 或 'quit' 退出")
    print("=" * 60)

    try:
        retriever = Retriever()
        reranker = Reranker(model_path=config.RERANK_MODEL)
    except Exception as exc:
        logger.error("向量库加载失败：%s", exc)
        logger.error("请先运行 python ingest.py 建立索引。")
        sys.exit(1)

    history: list[dict] = []

    while True:
        try:
            query = input("\n你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break

        print("\n思考中...\n")
        try:
            reply = answer(query, retriever, reranker, history)
            print(reply)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": reply})
        except Exception as exc:
            logger.error("回答生成失败：%s", exc)


if __name__ == "__main__":
    main()
