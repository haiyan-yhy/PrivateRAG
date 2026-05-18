"""Retrieve top-K relevant chunks from ChromaDB for a given query."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from llm import get_llm_and_embedding

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class Retriever:
    def __init__(self) -> None:
        llm, embed_model = get_llm_and_embedding()
        Settings.llm = llm
        Settings.embed_model = embed_model

        chroma_client = chromadb.PersistentClient(path=str(config.VECTOR_DIR))
        chroma_collection = chroma_client.get_or_create_collection(
            config.CHROMA_COLLECTION
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        self._retriever = index.as_retriever(similarity_top_k=config.TOP_K)

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        nodes = self._retriever.retrieve(query)
        chunks: List[RetrievedChunk] = []
        for node in nodes:
            source = node.metadata.get("file_name") or node.metadata.get(
                "file_path", "unknown"
            )
            chunks.append(
                RetrievedChunk(
                    text=node.get_content(),
                    source=source,
                    score=node.score if node.score is not None else 0.0,
                )
            )
        if config.DEBUG:
            for i, c in enumerate(chunks, 1):
                logger.debug(
                    "[chunk %d] source=%s score=%.4f\n%s\n", i, c.source, c.score, c.text[:200]
                )
        return chunks
