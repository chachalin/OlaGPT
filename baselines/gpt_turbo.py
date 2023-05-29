# -*- coding: utf-8 -*-
import os
import re
from typing import List, Union, Dict, Tuple, Any
from langchain.agents import Tool, AgentExecutor, AgentOutputParser, load_tools, BaseSingleActionAgent
from langchain.tools.base import BaseTool
from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain import OpenAI, GoogleSearchAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish

# import custom module
import sys
sys.path.append('.')
sys.path.append('..')
from utils.configs import configs
from utils.parser import get_arguments
from agent_template.Origin.tool import CustomOriginTool
from utils.evaluation import evaluation

args = get_arguments()
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']


class TurboAgent(BaseSingleActionAgent):
    """Agent wrapper for gpt-3.5-turbo"""

    @property
    def input_keys(self):
        return ["input"]

    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        input = kwargs['input']
        if len(intermediate_steps) == 0:
            return AgentAction(tool="GPT", tool_input=input, log="")
        else:
            output = ""
            for action, observation in intermediate_steps:
                output += observation
            return AgentFinish(return_values={"output": output}, log="")

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        input = kwargs['input']
        if len(intermediate_steps) == 0:
            return AgentAction(tool="GPT", tool_input=input, log="")
        else:
            output = ""
            for action, observation in intermediate_steps:
                output += observation
            return AgentFinish(return_values={"output": output}, log="")


if __name__ == '__main__':
    # Define custom LLM
    model_name = configs['model_name']
    # llm = OpenAI(model_name=model_name, temperature=0)
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # Define which tools the agent can use to answer user queries
    tools = [
        CustomOriginTool(dataset=args.dataset, few_shot=args.few_shot),
    ]
    tool_names = [tool.name for tool in tools]

    # Define custom BaseSingleActionAgent
    agent = TurboAgent()

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