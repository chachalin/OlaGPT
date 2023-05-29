# -*- coding: utf-8 -*-
import os
import re
from typing import List, Union, Dict, Tuple, Any, Optional
from langchain.agents import Tool, AgentExecutor, AgentOutputParser, load_tools, BaseMultiActionAgent
from langchain.tools.base import BaseTool
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, GoogleSearchAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish

# import custom module
import sys
sys.path.append('.')
sys.path.append('..')
from utils.configs import configs
from utils.parser import get_arguments
from utils.load_query import load_query
from agents.custom_base_agent import LLMMultiActionAgent
from agent_template.AnalogyThought.tool import AnalogyThought
from agent_template.CombineSearchJudge.tool import CombineTool
from agent_template.DecompositionThought.tool import DecompositionThought
from agent_template.DIY.tool import DIYTool
from agent_template.Origin.tool import CustomOriginTool
from agent_template.PlanThought.tool import PlanThought
# from agent_template.Reflect.tool import ReactTool, ReactReflectTool
from agent_template.StepThought.tool import StepThought
# from agent_template.ValidationThought.tool import ValidationThought
from utils.evaluation import evaluation
from tools.get_answer import AnswerTool

args = get_arguments()
os.environ["GOOGLE_CSE_ID"] = configs['tools']['google_cse_id']
os.environ["GOOGLE_API_KEY"] = configs['tools']['google_api_key']
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']


class RegexVoteAgent(BaseMultiActionAgent):
    """Master agent that controls all sub agents"""

    @property
    def input_keys(self):
        return ["input"]

    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        question_and_answers = kwargs['input']
        qa_json = load_query(question_and_answers)
        if len(intermediate_steps) == 0:
            # parse model_to_vote and get answer of each model
            candidate_outputs = qa_json['candidate_output']
            match = re.search(r'Candidate Outputs:\n(.*)', candidate_outputs, re.DOTALL)
            if match:
                answer_to_vote = match.group(1).split("#")
            else:
                raise ValueError(f'Could not parse {candidate_outputs}')
            actions = []
            for answer in answer_to_vote:
                model_name = answer.split(': ')[0]
                actions.append(AgentAction(tool=f'reg_{model_name}', tool_input=answer, log=""))
            return actions
        else:
            # vote for multiple answers
            votes = {}

            # append tool answers
            tools_answer = "\nTool Answer:\n"
            for action, observation in intermediate_steps:
                option = observation
                votes[option] = votes.get(option, 0) + 1
                tools_answer += f'{action.tool}: {observation}\n'

            final_answer = max(votes, key=votes.get)
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": f'{{Answer: {final_answer}}}' + tools_answer},
                log='',
            )

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[List[AgentAction], AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        question_and_answers = kwargs['input']
        qa_json = load_query(question_and_answers)
        if len(intermediate_steps) == 0:
            # parse model_to_vote and get answer of each model
            candidate_outputs = qa_json['candidate_output']
            match = re.search(r'Candidate Outputs:\n(.*)', candidate_outputs, re.DOTALL)
            if match:
                answer_to_vote = match.group(1).split("#")
            else:
                raise ValueError(f'Could not parse {candidate_outputs}')
            actions = []
            for answer in answer_to_vote:
                model_name = answer.split(': ')[0]
                actions.append(AgentAction(tool=f'reg_{model_name}', tool_input=answer, log=""))
            return actions
        else:
            # vote for multiple answers
            votes = {}

            # append tool answers
            tools_answer = "\nTool Answer:\n"
            for action, observation in intermediate_steps:
                option = observation
                votes[option] = votes.get(option, 0) + 1
                tools_answer += f'{action.tool}: {observation}\n'

            final_answer = max(votes, key=votes.get)
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": f'{{Answer: {final_answer}}}' + tools_answer},
                log='',
            )


if __name__ == '__main__':
    # Define custom LLM
    model_name = configs['model_name']
    # llm = OpenAI(model_name=model_name, temperature=0)
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # Define which tools the agent can use to answer user queries
    model_to_vote = eval(args.model_to_vote)
    tools = [
        AnswerTool(name=f'reg_{model_name}')
        for model_name in model_to_vote
    ]
    tool_names = [tool.name for tool in tools]

    # Define custom LLMMultiActionAgent
    agent = RegexVoteAgent(
        stop=['\nFinal Answer: '],
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    # get args
    question = args.question
    is_eval = args.is_eval

    if is_eval:
        result = evaluation(agent_executor, llm, args)
        for k, v in result.items():
            print(f'{k}: {v}')
    else:
        ans = agent_executor.run(question)
        print(ans)
