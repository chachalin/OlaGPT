# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from langchain.llms import OpenAI
from utils.configs import configs
from utils.fewshots import get_notes_few_shot
from utils.load_query import load_query
from utils.prompts import get_query_format


class CustomOriginTool(BaseTool):
    """Tool that adds the origin api."""
    name = "GPT"
    description = (
        "A base language model which answers the question, without using other tools.")
    dataset = 'aqua'
    few_shot = 0

    def _run(self, query: str) -> str:
        """Use the tool."""

        query_json = load_query(query)
        query_json, final_query = get_query_format(self.dataset, query_json)

        llm = OpenAI(temperature=0, model_name=configs['model_name'])
        if self.few_shot:
            templates_prefix = ""
            notes_few_shot = get_notes_few_shot(
                query_json, self.dataset, self.few_shot, templates_prefix)
            return llm(notes_few_shot + final_query)

        return llm(final_query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CustomOriginTool does not support async")
