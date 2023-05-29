# -*- coding: utf-8 -*-
import os
import re
from typing import List, Union, Dict
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
from agent_template.DIY.tool import DIYTool
from utils.evaluation import evaluation

args = get_arguments()
os.environ["GOOGLE_CSE_ID"] = configs['tools']['google_cse_id']
os.environ["GOOGLE_API_KEY"] = configs['tools']['google_api_key']
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']

# Set up the base template
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Actions: the tools to take (seperated by comma), tool name should be one of [{tool_names}]
Actions Input: the question and options
Observation: the result of action 1
Observation: the result of action 2
...(this Observations can repeat N time, where N is the number of possible actions)
Thought: Each observation gives an possible answer. I should get the majority vote of those answers as the final answer.
Final Answer: the final answer to the original input question

Begin!

Question: {input}
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
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation[{action.tool}]: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        # print(self.template.format(**kwargs))
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
        # Parse out the actions and actions input
        regex = r"Thought: (.*?)[\n]*Actions: (.*?)[\n]*Actions Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        thought = match.group(1).strip()
        actions = match.group(2).strip()
        actions = actions.split(', ')
        action_input = match.group(3) + "\nThe answer must end with json format: {Answer: one of options[A,B,C,D,E]}."
        # Return the actions and actions input
        print(actions)
        agent_actions = [
            AgentAction(
                tool=action.strip(),
                tool_input=action_input.strip(" ").strip('"'),
                log=f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}"
            )
            for action in actions
        ]
        return agent_actions


if __name__ == '__main__':
    # Define custom LLM
    model_name = configs['model_name']
    # model_name = 'text-davinci-003'
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
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )

    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define custom output parser
    output_parser = CustomOutputParser()

    # Define custom LLMMultiActionAgent
    agent = LLMMultiActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=['\nObservation: '],
        allowed_tools=tool_names
    )

    # Define agent executor
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
