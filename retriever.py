"""Retrieve top-K relevant chunks from ChromaDB for a given query."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

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
        vector_retriever = index.as_retriever(similarity_top_k=config.TOP_K * 2)

        raw = chroma_collection.get(include=["documents", "metadatas"])
        bm25_nodes = [
            TextNode(text=doc, metadata=meta or {})
            for doc, meta in zip(raw["documents"], raw["metadatas"])
        ]
        bm25_retriever = BM25Retriever.from_defaults(
                            nodes=bm25_nodes,
                            similarity_top_k=config.TOP_K * 2,
                            )
        self._retriever = QueryFusionRetriever(
                            retrievers=[vector_retriever, bm25_retriever],
                            similarity_top_k=config.TOP_K,
                            num_queries=3,
                            mode="reciprocal_rerank",  # RRF
                            use_async=False,
                            )

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
