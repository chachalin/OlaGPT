# -*- coding: utf-8 -*-
from langchain.tools import BaseTool
from utils.parse_option import parse_option


class AnswerTool(BaseTool):

    name = "placeholder for model name"
    description = "Get predicted answer according to the given model name."

    def _run(self, answer: str):
        # parse the answer
        if self.name.startswith('reg'):
            # parse the answer for regex vote agent
            final_answer = parse_option(answer)
        else:
            # parse the answer for llm vote agent
            final_answer = ': '.join(answer.split(': ')[2:])
        return final_answer

    async def _arun(self, answer: str) -> str:
        """Use the tool asynchronously."""
        # parse the answer
        if self.name.startswith('reg'):
            # parse the answer for regex vote agent
            final_answer = parse_option(answer)
        else:
            # parse the answer for llm vote agent
            final_answer = ': '.join(answer.split(': ')[2:])
        return final_answer