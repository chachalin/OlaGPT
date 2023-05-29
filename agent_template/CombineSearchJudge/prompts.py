# -*- coding: utf-8 -*-
import os
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.tools.human.tool import HumanInputRun
from langchain.agents import ZeroShotAgent, Tool, tool, AgentExecutor
from langchain.prompts import PromptTemplate

import sys
sys.path.append("..")
sys.path.append('../..')
from agent_template.Origin.tool import CustomOriginTool
from utils.configs import configs

# from demo_agents.tool_retrieval_agent import get_tools

os.environ["GOOGLE_CSE_ID"] = configs['tools']['google_cse_id']
os.environ["GOOGLE_API_KEY"] = configs['tools']['google_api_key']
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']

COMBINE_INSTRUCTION = """Use the following format:
        Question: the input question you must answer
        Thought: Think first about the issues that require access to search engines to acquire the latest knowledge
        Search: when you find some problems you need to visit the latest knowledge base in advance, you can choose the google_search tool to ask
        Action Input: the input to the action.
        Thought: you should always think about what
        Observation: the result of the action to do
        Action: the action to take, should be one of [{tool_names}]. Don't choose human and google_search tool
        Action Input: the input to the action. Questions should be as complete as possible
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Analysis: when you are not sure about the answer, you can choose the human tool to ask
        Finish: the final answer to the original input question

        """

prefix = """Answer the following questions as best you can."""
prefix += """You can choose tools and think about answers based on questions and prompts.
        I want you to use as many tools as possible to get the answer before making a final summary.
        You can use the following tools:"""

suffix = """

Start!
        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

examples = """
Question: What are Jackie Chan's masterpieces?
Thought: First of all, I have to find out who Jackie Chan is, and then check his TV series or movie works.
Search: google_search
Search Input: Jackie Chan's masterpieces
Observation: 
Final Answer: 
"""


search = GoogleSearchAPIWrapper()
tools = [
    CustomOriginTool(),
    HumanInputRun(),
    Tool(
        name="google_search",
        func=search.run,
        description="A wrapper around Google Search. "
        "Useful for when you need to answer questions about current events. "
             "Input should be a search query."
        "After using the tool to return the answers, analyze and judge, and some answers may be wrong."
    )
]