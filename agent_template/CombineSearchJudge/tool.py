# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from agent_template.CombineSearchJudge.combine_sj import CombineAgent


class CombineTool(BaseTool):
    """Tool that adds the origin api."""
    name = "CombineTool"

    description = (
        "Applicable to scenarios that need to consult search engines in advance to obtain advanced knowledge or require user feedback")

    def _run(self, query: str) -> str:
        """Use the tool."""
        ra = CombineAgent()
        return ra.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CombineTool does not support async")
