# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from langchain.llms import OpenAI
from utils.configs import configs
from utils.fewshots import get_notes_few_shot
from utils.load_query import load_query
from utils.prompts import get_query_format


class PlanThought(BaseTool):
    """Tool that adds the origin api."""
    name = "PlanThought"
    description = (
        "Suitable for problems that require multi-step planning before and after completion")
    dataset = 'aqua'
    few_shot = 0

    def _run(self, query: str) -> str:
        """Use the tool."""

        query_json = load_query(query)

        query_json, final_query = get_query_format(self.dataset, query_json)

        planning_thought = """
        planning_thought: Think carefully about the problem to be solved and make a detailed plan to solve it. \n
        Now give you the problem, please select the best option from the choices as the answer using the planning_thought. \n
        """

        llm = OpenAI(temperature=0, model_name=configs['model_name'])
        if self.few_shot:
            if self.dataset != 'aqua':
                templates_prefix = "Think carefully about the problem to be solved and make a detailed plan to solve it."
            else:
                templates_prefix = "Let's think with planning_thought:"
            notes_few_shot = get_notes_few_shot(
                query_json, self.dataset, self.few_shot, templates_prefix)
            return llm(notes_few_shot + planning_thought + final_query)

        return llm(planning_thought + final_query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("PlanThought does not support async")
