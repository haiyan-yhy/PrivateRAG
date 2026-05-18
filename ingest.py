"""Load documents from data/, chunk them, and persist to ChromaDB."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from llm import get_llm_and_embedding

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def ingest() -> None:
    data_dir = config.DATA_DIR
    vector_dir = config.VECTOR_DIR

    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.error("data/ 目录为空，请先放入 PDF 或 Markdown 文件。")
        sys.exit(1)

    logger.info("加载文档：%s", data_dir)
    documents = SimpleDirectoryReader(
        input_dir=str(data_dir),
        required_exts=[".pdf", ".md"],
        recursive=True,
        filename_as_id=True,
    ).load_data()
    logger.info("共加载 %d 个文档片段", len(documents))

    llm, embed_model = get_llm_and_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model

    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    vector_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(vector_dir))
    chroma_collection = chroma_client.get_or_create_collection(config.CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("开始 embedding 并写入向量库...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )
    logger.info("向量库构建完成，存储路径：%s", vector_dir)


if __name__ == "__main__":
    ingest()
