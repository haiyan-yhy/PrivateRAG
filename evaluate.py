"""Evaluate RAG quality using RAGAS with auto-generated synthetic test data.

Usage:
    python evaluate.py                 # 生成20个问题并评估
    python evaluate.py --test-size 30  # 自定义测试集大小
    python evaluate.py --skip-gen      # 跳过生成，复用已有的 testset.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from llama_index.core import SimpleDirectoryReader, Settings

import config
from llm import build_prompt, get_llm_and_embedding
from retriever import Retriever
from reranker import Reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TESTSET_PATH = config.BASE_DIR / "testset.json"
RESULT_PATH  = config.BASE_DIR / "eval_result.json"


def _make_ragas_llm(llm):
    import warnings
    from ragas.llms import LlamaIndexLLMWrapper
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wrapper = LlamaIndexLLMWrapper(llm)
    # Some providers (DeepSeek, Ollama, etc.) reject n>1. RAGAS occasionally
    # requests n>1 for certain metrics, so clamp it here unconditionally.
    _orig = wrapper.check_args
    wrapper.check_args = lambda n, temperature, stop, callbacks: _orig(
        1, temperature, stop, callbacks
    )
    return wrapper


def _run_rag(query: str, retriever: Retriever, reranker: Reranker) -> tuple[str, list[str]]:
    chunks = retriever.retrieve(query)
    chunks = reranker.rerank(query, chunks, top_n=config.RERANK_TOP_N)
    if not chunks:
        return "无法从知识库中找到答案。", []
    contexts = [c.text for c in chunks]
    context_text = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(contexts))
    response = Settings.llm.complete(build_prompt(query=query, context=context_text))
    return str(response).strip(), contexts


def generate_testset(documents, llm, embed_model, test_size: int):
    from langchain_core.documents import Document as LCDocument
    from ragas.testset import TestsetGenerator
    from ragas.llms import LlamaIndexLLMWrapper
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    from ragas.testset.transforms import default_transforms
    from ragas.testset.transforms.splitters.headline import HeadlineSplitter

    ragas_llm  = _make_ragas_llm(llm)
    ragas_emb  = LlamaIndexEmbeddingsWrapper(embed_model)

    # default_transforms needs LangChain docs to decide which pipeline branch to build.
    lc_docs = [LCDocument(page_content=f"""
                请基于以下内容生成中文问题与答案：
                          
                {doc.text}""") for doc in documents if doc.text and doc.text.strip()]
    transforms = default_transforms(documents=lc_docs, llm=ragas_llm, embedding_model=ragas_emb)

    # HeadlinesExtractor only processes nodes with >500 tokens, but HeadlineSplitter
    # has no filter — it crashes on nodes that never got their 'headlines' property set.
    # Restrict the splitter to nodes where the property actually exists.
    for t in transforms:
        if isinstance(t, HeadlineSplitter):
            t.filter_nodes = lambda node: node.get_property("headlines") is not None
            break

    generator = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_emb)
    testset = generator.generate_with_llamaindex_docs(documents, testset_size=test_size, transforms=transforms)
    df = testset.to_pandas()
    df.to_json(TESTSET_PATH, orient="records", force_ascii=False, indent=2)
    logger.info("测试集已保存至 %s", TESTSET_PATH)
    return df


def load_testset():
    if not TESTSET_PATH.exists():
        logger.error("找不到 %s，请先运行生成步骤（去掉 --skip-gen）", TESTSET_PATH)
        sys.exit(1)
    import pandas as pd
    df = pd.read_json(TESTSET_PATH)
    logger.info("从文件加载测试集，共 %d 条", len(df))
    return df


def main(test_size: int = 20, skip_gen: bool = False) -> None:
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    except ImportError:
        logger.error("请先安装依赖：pip install ragas")
        sys.exit(1)

    llm, embed_model = get_llm_and_embedding()
    Settings.llm = llm
    Settings.embed_model = embed_model

    if skip_gen:
        df = load_testset()
    else:
        logger.info("加载文档...")
        documents = SimpleDirectoryReader(
            input_dir=str(config.DATA_DIR),
            required_exts=[".pdf", ".md"],
            recursive=True,
            filename_as_id=True,
        ).load_data()
        logger.info("共加载 %d 个文档", len(documents))
        logger.info("生成合成测试集（%d 个问题）...", test_size)
        df = generate_testset(documents, llm, embed_model, test_size)

    logger.info("初始化 RAG pipeline...")
    retriever = Retriever()
    reranker = Reranker(model_path=config.RERANK_MODEL)

    logger.info("对每个问题运行 RAG pipeline...")
    samples = []
    for _, row in df.iterrows():
        question = row["user_input"]
        ground_truth = row.get("reference", "")
        logger.info("Q: %.80s", question)
        answer, contexts = _run_rag(question, retriever, reranker)
        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))

    logger.info("开始 RAGAS 评估...")
    ragas_llm        = _make_ragas_llm(llm)
    ragas_embeddings = LlamaIndexEmbeddingsWrapper(embed_model)

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores = result.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ].mean().to_dict()

    print("\n" + "=" * 50)
    print("  RAGAS 评估结果")
    print("=" * 50)
    labels = {
        "faithfulness":      "忠实度     （有无幻觉）",
        "answer_relevancy":  "答案相关性 （是否回答了问题）",
        "context_precision": "上下文精度 （检索是否精准）",
        "context_recall":    "上下文召回 （相关内容是否都找到）",
    }
    for key, score in scores.items():
        print(f"  {labels.get(key, key):<30} {score:.4f}")
    print("=" * 50)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    logger.info("详细结果已保存至 %s", RESULT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=int, default=20, help="合成测试集大小")
    parser.add_argument("--skip-gen",  action="store_true",   help="跳过生成，复用已有 testset.json")
    args = parser.parse_args()
    main(test_size=args.test_size, skip_gen=args.skip_gen)
