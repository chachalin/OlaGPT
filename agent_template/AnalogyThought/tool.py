# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from langchain.llms import OpenAI
from utils.configs import configs
from utils.fewshots import get_notes_few_shot
from utils.load_query import load_query
from utils.prompts import get_query_format


class AnalogyThought(BaseTool):
    name = "AnalogyThought"

    description = ("Suitable for complex analogical reasoning problems"
                   )
    dataset = 'aqua'
    few_shot = 0

    def _run(self, query: str) -> str:
        """Use the tool."""

        query_json = load_query(query)

        analogy_thought = """
        analogy_thought: For the problem of analogical reasoning, it is completed in three steps.
        First conduct an inductive analysis of the given sample data, considering the similarity of the relationship between words;
        Next, judge whether the sample to be selected is satisfied;
        Finally check the validity of the mapping and explain if the mapping is correct.

        Now give you the problem, please select the best option from the choices as the answer using the analogy_thought. \n
        """

        query_json, final_query = get_query_format(self.dataset, query_json)

        llm = OpenAI(temperature=0, model_name=configs['model_name'])
        if self.few_shot:
            templates_prefix = "Let's think with analogy_thought:"
            notes_few_shot = get_notes_few_shot(
                query_json, self.dataset, self.few_shot, templates_prefix)
            return llm(notes_few_shot + analogy_thought + final_query)

        return llm(analogy_thought + final_query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("AnalogyThought does not support async")
