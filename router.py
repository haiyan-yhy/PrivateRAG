from llama_index.core import Settings

def router_query(query: str) -> str:
    # 这里可以添加更复杂的路由逻辑，比如基于关键词、意图识别等
    prompt = f"""
            你是一个RAG路由器。

            请判断用户问题：

            - 是否需要知识库检索
            - 或者模型可直接回答

            规则：

            以下情况返回 direct：
            - 问候语
            - 简单聊天

            以下情况返回 retrieve：
            - 用户项目
            - 私有知识
            - 文档内容
            - 公司资料

            只能返回：

            direct
            或
            retrieve

            用户问题：
            {query}
            """
    response = Settings.llm.complete(prompt)

    route = str(response).strip().lower()

    return route
