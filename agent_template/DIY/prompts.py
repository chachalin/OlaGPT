# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from langchain.llms import OpenAI
from utils.configs import configs


llm = OpenAI(temperature=0, model_name=configs['model_name'])

DIY_INSTRUCTION = """Use the following format:
        Question: the input question you must answer
        Thought: you should always think about what
        Action: the action to take, should be one of [{tool_names}].
        Action Input: the input question you must answer
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        """

prefix = """Answer the following questions as best you can. """
# I want you to use as many thoughts as possible to get the answer before making a final summary.
prefix += """You can choose thoughts and think about answers based on questions and prompts.
        You can use the following thoughts:"""

suffix1 = """
Action Input is the original question, do not modify it.
Here are one example:
Question: What are Jackie Chan's masterpieces?
Thought: I need to choose the most suitable idea to solve this problem and I need to plan first.
Action: PlanThought
Action Input: What are Jackie Chan's masterpieces?
Observation: First of all, I have to find out who Jackie Chan is, and then check his TV series or movie works.
Action: Search
Action Input: What are Jackie Chan's masterpieces?
Observation: Police Story，Drunken Master ，Red Bronx，The Myth
Final Answer: Jackie Chan's masterpieces include Police Story, Drunken Master, Red Bronx, The Myth
"""

suffix2 = """
Action Input must be the question.
Start!
        Question: {input}
        {agent_scratchpad}"""


class CombinedThought(BaseTool):
    """Tool that adds the origin api."""
    name = "CombinedThought"
    description = (
        "Combination thinking, also known as 'connection thinking' or 'convergence thinking',"
        " refers to a way of thinking that connects multiple seemingly unrelated "
        "things through imagination, so as to make them into a new whole that is "
        "inseparable from each other.")

    def _run(self, query: str) -> str:
        """Use the tool."""

        combined_thought = """
        Use combinatorial thinking to solve the problem. \n
        """

        return llm(combined_thought + query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("CombinedThought does not support async")


class ValidationThought(BaseTool):
    """Tool that adds the origin api."""
    name = "ValidationThought"
    description = (
        "It is suitable for situations where you are not sure about the answers you have obtained. "
        "After completing the questions, you can do verification to confirm the accuracy of the answers again.")

    def _run(self, query: str) -> str:
        """Use the tool."""

        validation_thought = """
        In another way to solve this problem, verify and synthesize the final answer. \n
        """

        return llm(validation_thought + query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("ValidationThought does not support async")


class AnalogyThought(BaseTool):
    name = "AnalogyThought"

    description = ("Suitable for complex analogical reasoning problems"
                   )

    def _run(self, query: str) -> str:
        """Use the tool."""

        analogy_thought = """
        analogy_thought: For the problem of analogical reasoning, it is completed in three steps.
        First conduct an inductive analysis of the given sample data, considering the similarity of the relationship between words;
        Next, judge whether the sample to be selected is satisfied;
        Finally check the validity of the mapping and explain if the mapping is correct.

        Now give you the problem, please select the best option from the choices as the answer using the analogy_thought. \n
        """

        llm = OpenAI(temperature=0, model_name=configs['model_name'])

        return llm(analogy_thought + query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("AnalogyThought does not support async")


class DecompositionThought(BaseTool):
    """Tool that adds the origin api."""
    name = "DecompositionThought"
    description = (
        "Suitable for more complex problem scenarios, such as combination problems, complex reasoning problems, etc. The main idea is to simplify the complicated and make the difficult easy."
    )

    def _run(self, query: str) -> str:
        """Use the tool."""

        disaggregated_thought = """disaggregated_thought: The following questions can be disassembled into multiple sub-questions to solve,
        the steps and answers of each sub-question are given, and finally the answer to the following question is given. \n
        Now give you the problem, please select the best option from the choices as the answer using the disaggregated_thought. \n
        """

        llm = OpenAI(temperature=0, model_name=configs['model_name'])
        return llm(disaggregated_thought + query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError(
            "DisaggregatedThought does not support async")


class PlanThought(BaseTool):
    """Tool that adds the origin api."""
    name = "PlanThought"
    description = (
        "Suitable for problems that require multi-step planning before and after completion")

    def _run(self, query: str) -> str:
        """Use the tool."""
        planning_thought = """
        planning_thought: Think carefully about the problem to be solved and make a detailed plan to solve it. \n
        Now give you the problem, please select the best option from the choices as the answer using the planning_thought. \n
        """
        llm = OpenAI(temperature=0, model_name=configs['model_name'])
        return llm(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("PlanThought does not support async")
