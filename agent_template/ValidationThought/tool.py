# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from agent_template.ValidationThought.validation_main import ValidAgent
from utils.configs import configs
from utils.fewshots import get_notes_few_shot
from utils.load_query import load_query
from utils.prompts import get_query_format


class ValidationThought(BaseTool):
    """Tool that adds the origin api."""
    name = "ValidationThought"

    description = (
        "Suitable for use when you are not sure whether you can answer the relevant questions perfectly")
    dataset = 'aqua'
    few_shot = 0

    def _run(self, query: str) -> str:
        """Use the tool."""

        query_json = load_query(query)
        query_json, final_query = get_query_format(self.dataset, query_json)

        ra = ValidAgent()
        if self.few_shot:
            templates_prefix = ''
            notes_few_shot = get_notes_few_shot(
                query_json, self.dataset, self.few_shot, templates_prefix)
            return ra.run(notes_few_shot + final_query)
        return ra.run(final_query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("ValidationThought does not support async")
