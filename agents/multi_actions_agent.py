# -*- coding: utf-8 -*-
import os
import re
from typing import List, Union, Dict, Tuple, Any, Optional
from langchain.agents import Tool, AgentExecutor, AgentOutputParser, load_tools
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
from agents.custom_base_agent import LLMMultiActionAgent
from agent_template.AnalogyThought.tool import AnalogyThought
from agent_template.CombineSearchJudge.tool import CombineTool
from agent_template.DecompositionThought.tool import DecompositionThought
from agent_template.DIY.tool import DIYTool
from agent_template.Origin.tool import CustomOriginTool
from agent_template.PlanThought.tool import PlanThought
# from agent_template.Reflect.tool import ReactTool, ReactReflectTool
from agent_template.StepThought.tool import StepThought
from agent_template.ValidationThought.tool import ValidationThought
from agent_template.DisassembleThought.tool import DisassembleThought
from utils.evaluation import evaluation
from utils.parse_option import parse_option

args = get_arguments()
os.environ["GOOGLE_CSE_ID"] = configs['tools']['google_cse_id']
os.environ["GOOGLE_API_KEY"] = configs['tools']['google_api_key']
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']

# Set up the base template
template = """Answer the following questions as best you can. 
You have been given the question and the following possible answers by different tools, please select the most consistent one as the final answer:

{answers}

Use the following format:

Question: the input question you must answer
Final Answer: the final answer to the input question. The answer's format must end with json format: {{Answer: one of options[A,B,C,D,E]}}

Begin!

Question: {input}
{agent_scratchpad}"""

simple_template = """You have been given the following possible answers by different tools, please select the most consistent one as the final answer:

{answers}

Use the following format:
Final Answer: the final answer to the original input question. The answer's format must end with json format: {{Answer: one of options[A,B,C,D,E]}}

{agent_scratchpad}"""


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        # print(intermediate_steps)
        answers = ""
        for action, observation in intermediate_steps:
            answers += f"\nObservation[{action.tool}]: {observation}"
        # Set the agent_scratchpad variable to empty value
        kwargs["agent_scratchpad"] = ""
        # Set the answers variable to the observation of actions
        kwargs["answers"] = answers
        print(self.template.format(**kwargs))
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[List[AgentAction], AgentFinish]:
        # Check if agent should finish
        # print(llm_output)
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        else:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )


class MasterAgent(LLMMultiActionAgent):
    """Master agent that controls all sub agents"""

    allowed_tools: Optional[List[str]] = None
    vote_mode: str = 'regex'  # can be llm or regex

    @property
    def input_keys(self):
        return ["input"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        return self.allowed_tools

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
        question = kwargs['input']
        if len(intermediate_steps) == 0:
            allowed_tools = self.get_allowed_tools()
            # TODO tools retrieval
            return [
                AgentAction(tool=tool_name, tool_input=question, log="")
                for tool_name in allowed_tools
            ]
        else:
            if self.vote_mode == 'llm':
                # vote for multiple answers by llm
                output = self.llm_chain.run(
                    intermediate_steps=intermediate_steps, stop=self.stop, **kwargs
                )
                # append tool answers
                tools_answer = "\nTool Answer:\n"
                for action, observation in intermediate_steps:
                    tools_answer += f'{action.tool}: {observation}\n'
                return self.output_parser.parse(output+tools_answer)
            else:
                # vote for multiple answers by
                votes = {}

                # append tool answers
                tools_answer = "\nTool Answer:\n"
                for action, observation in intermediate_steps:
                    option = parse_option(observation)
                    votes[option] = votes.get(option, 0) + 1
                    tools_answer += f'{action.tool}: {observation}\n'

                final_answer = max(votes, key=votes.get)
                return self.output_parser.parse(f'{{Answer: {final_answer}}}' + tools_answer)

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
        question = kwargs['input']
        if len(intermediate_steps) == 0:
            allowed_tools = self.get_allowed_tools()
            # TODO tools retrieval
            return [
                AgentAction(tool=tool_name, tool_input=question, log="")
                for tool_name in allowed_tools
            ]
        else:
            # vote for multiple answers
            output = self.llm_chain.run(
                intermediate_steps=intermediate_steps, stop=self.stop, **kwargs
            )
            # append tool answers
            tools_answer = "\nTool Answer:\n"
            for action, observation in intermediate_steps:
                tools_answer += f'{action.tool}: {observation}\n'
            return self.output_parser.parse(output + tools_answer)


if __name__ == '__main__':
    # Define custom LLM
    model_name = configs['model_name']
    vote_mode = configs['vote_mode']
    # llm = OpenAI(model_name=model_name, temperature=0)
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # Define which tools the agent can use to answer user queries
    tools = [
        # CombineTool(),
        CustomOriginTool(dataset=args.dataset, few_shot=args.few_shot),
        # ReactTool(),
        # ReactReflectTool(),
        # DIYTool(dataset=args.dataset, few_shot=args.few_shot),
        AnalogyThought(dataset=args.dataset, few_shot=args.few_shot),
        DecompositionThought(dataset=args.dataset, few_shot=args.few_shot),
        PlanThought(dataset=args.dataset, few_shot=args.few_shot),
        StepThought(dataset=args.dataset, few_shot=args.few_shot),
        # ValidationThought(dataset=args.dataset, few_shot=args.few_shot),
        DisassembleThought(dataset=args.dataset, few_shot=args.few_shot)
    ]
    tool_names = [tool.name for tool in tools]

    # Define custom prompt template
    prompt = CustomPromptTemplate(
        template=simple_template,
        tools=tools,
        # This omits the `agent_scratchpad`, `answers` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define custom output parser
    output_parser = CustomOutputParser()

    # Define custom LLMMultiActionAgent
    agent = MasterAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=['\nFinal Answer: '],
        allowed_tools=tool_names,
        vote_mode=vote_mode
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
