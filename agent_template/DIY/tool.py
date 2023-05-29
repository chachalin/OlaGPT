# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from agent_template.DIY.DIY_main import DIYAgent
from utils.load_query import load_query


class DIYTool(BaseTool):
    """Tool that adds the origin api."""
    name = "DIYTool"
    # description = "包含了各种人类解题思维引导，可以自由选择合适的思想完成相关问题解答。针对数学问题、推理问题等有不错的效果。"
    description = (
        "Contains a variety of human problem-solving thinking guides, and you can freely choose appropriate thoughts \
        to complete related problem answers. It has a good effect on math problems, reasoning problems, etc.")
    dataset = 'aqua'
    few_shot = 0

    def _run(self, query: str) -> str:
        """Use the tool."""

        query_json = load_query(query)

        ra = DIYAgent()
        return ra.run_sim(query_json, self.dataset, self.few_shot)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CombineTool does not support async")
