# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from langchain.llms import OpenAI
from utils.configs import configs
from utils.fewshots import get_notes_few_shot
from utils.load_query import load_query
from utils.prompts import get_query_format


class DisassembleThought(BaseTool):
    """Tool that adds the origin api."""
    name = "DecompositionThought"
    description = (
        "Suitable for more complex problem scenarios, such as combination problems, complex reasoning problems, etc. The main idea is to simplify the complicated and make the difficult easy."
    )
    dataset = 'aqua'
    few_shot = 0

    def _run(self, query: str) -> str:
        """Use the tool."""

        query_json = load_query(query)
        query_json, final_query = get_query_format(self.dataset, query_json)

        disassemble_thought = """disassemble_thought: Disassemble the following complex problems to solve them step by step \n
        Now give you the problem, please select the best option from the choices as the answer using the disaggregated_thought. \n
        """

        llm = OpenAI(temperature=0, model_name=configs['model_name'])
        if self.few_shot:
            templates_prefix = "Let's think with disassemble_thought:"
            notes_few_shot = get_notes_few_shot(
                query_json, self.dataset, self.few_shot, templates_prefix)
            return llm(notes_few_shot + disassemble_thought + final_query)
        return llm(disassemble_thought + final_query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("DisassembleThought does not support async")
