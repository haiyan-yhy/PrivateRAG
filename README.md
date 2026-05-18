# MyRAG — 本地 RAG 知识库

基于 LlamaIndex + ChromaDB 的本地 RAG 问答系统，支持 PDF 和 Markdown 文档。

## 项目结构

```
rag_project/
├── app.py          # CLI 问答入口
├── ingest.py       # 文档加载 + chunking + 入库
├── retriever.py    # 向量检索封装
├── llm.py          # LLM / Embedding 工厂
├── config.py       # 配置（读取环境变量）
├── .env.example    # 环境变量模板
├── data/           # 放你的 PDF / Markdown 文件
├── vectorstore/    # ChromaDB 持久化（自动生成）
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
cd rag_project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY
```

或直接设置环境变量：

```bash
export OPENAI_API_KEY=sk-...
```

### 3. 放入文档

把 PDF 或 Markdown 文件放入 `data/` 目录：

```bash
cp your_doc.pdf data/
cp your_notes.md data/
```

### 4. 建立索引

```bash
python ingest.py
```

### 5. 启动问答

```bash
python app.py
```

## 切换 Provider

编辑 `.env` 中的 `LLM_PROVIDER`：

| 值 | 说明 | 需要 |
|----|------|------|
| `openai`（默认） | GPT-4o-mini + text-embedding-3-small | OPENAI_API_KEY |
| `anthropic` | Claude Haiku + OpenAI embedding | ANTHROPIC_API_KEY + OPENAI_API_KEY |
| `ollama` | 完全本地，无需 Key | 本地安装 [Ollama](https://ollama.com) |

### Ollama 本地运行

```bash
# 安装 Ollama 后拉取模型
ollama pull llama3.2
ollama pull nomic-embed-text

# 取消 requirements.txt 中 Ollama 相关注释后重新安装
pip install llama-index-llms-ollama llama-index-embeddings-ollama

# 设置 provider
export LLM_PROVIDER=ollama
```

## Debug 模式

```bash
DEBUG=true python app.py
```

开启后会打印每次检索到的 top-K chunks 及相似度分数。

## 重建索引

删除 `vectorstore/` 目录后重新运行 `ingest.py`：

```bash
rm -rf vectorstore/
python ingest.py
```
