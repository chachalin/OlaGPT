# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from langchain.llms import OpenAI
from utils.configs import configs
from utils.fewshots import get_notes_few_shot
from utils.load_query import load_query
from utils.prompts import get_query_format


class StepThought(BaseTool):
    """Tool that adds the origin api."""
    name = "StepThought"
    description = (
        "Suitable for problems that need to be completed step by step."
    )
    dataset = 'aqua'
    few_shot = 0

    def _run(self, query: str) -> str:
        """Use the tool."""

        query_json = load_query(query)

        query_json, final_query = get_query_format(self.dataset, query_json)

        step_thought = """Let's think step by step."""
        llm = OpenAI(temperature=0, model_name=configs['model_name'])
        if self.few_shot:
            templates_prefix = step_thought
            notes_few_shot = get_notes_few_shot(
                query_json, self.dataset, self.few_shot, templates_prefix)
            return llm(notes_few_shot + final_query + step_thought)
        return llm(final_query + step_thought)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("StepThought does not support async")
