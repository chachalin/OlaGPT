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
        query_json = load_query(kwargs['input'])
        query_json['instruct'] = f"Now give you the {query_json['llm_task_type']} question and choices:"
        final_query = '\n'.join([
            query_json['instruct'],
            query_json['context'],
            query_json['question'],
            query_json['options'],
        ])
        kwargs["input"] = final_query
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


class LLMVoteAgent(LLMMultiActionAgent):
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
                actions.append(AgentAction(tool=model_name, tool_input=answer, log=""))
            return actions
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
                actions.append(AgentAction(tool=model_name, tool_input=answer, log=""))
            return actions
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
    # llm = OpenAI(model_name=model_name, temperature=0)
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # Define which tools the agent can use to answer user queries
    model_to_vote = eval(args.model_to_vote)
    tools = [
        AnswerTool(name=model_name)
        for model_name in model_to_vote
    ]
    tool_names = [tool.name for tool in tools]

    # Define custom prompt template
    prompt = CustomPromptTemplate(
        # template=simple_template,
        template=template,
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
    agent = LLMVoteAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
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
