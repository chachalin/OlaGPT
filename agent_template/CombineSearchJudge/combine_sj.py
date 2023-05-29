# -*- coding: utf-8 -*-
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.tools.human.tool import HumanInputRun
# from langchain.tools.google_search.tool import GoogleSearchRun
from langchain.agents import initialize_agent
from langchain.agents import ZeroShotAgent, Tool, tool, AgentExecutor
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain import LLMChain

import os
import sys
sys.path.append("..")
sys.path.append('../..')
from agent_template.CombineSearchJudge.prompts import COMBINE_INSTRUCTION, suffix, prefix, examples
from agent_template.Origin.tool import CustomOriginTool
from utils.configs import configs

os.environ["GOOGLE_CSE_ID"] = configs['tools']['google_cse_id']
os.environ["GOOGLE_API_KEY"] = configs['tools']['google_api_key']
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']


class CombineAgent:

    def __init__(self):
        self.llm = OpenAI(model_name='gpt-3.5-turbo-0301')
        self.main_agent()

    def main_agent(self):
        self.search = GoogleSearchAPIWrapper()
        tools = [
            Tool(
                name="Intermediate Answer",
                func=self.search.run,
                description="useful for when you need to ask with search"
            )
        ]
        self.self_ask_with_search = initialize_agent(
            tools, self.llm, agent="self-ask-with-search", verbose=True)

    def son_agent(self):
        tools = [
            CustomOriginTool(),
            HumanInputRun(),
            Tool(
                name="google_search",
                func=self.search.run,
                description="A wrapper around Google Search. "
                "Useful for when you need to answer questions about current events. "
                     "Input should be a search query."
            )

        ]

        combine_prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=COMBINE_INSTRUCTION,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )

        self.memory = ConversationSummaryBufferMemory(llm=OpenAI(
            temperature=0), max_token_limit=10, memory_key="chat_history")

        self.llm_chain = LLMChain(llm=self.llm, prompt=combine_prompt)
        self.agent = ZeroShotAgent(
            llm_chain=self.llm_chain,
            tools=tools,
            verbose=True)

        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=tools, verbose=True, memory=self.memory)

    def run(self, query):
        self.son_agent()
        return self.agent_chain.run(input=query)
