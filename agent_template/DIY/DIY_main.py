# -*- coding: utf-8 -*-
from utils.fewshots import get_notes_few_shot
from agent_template.StepThought.tool import StepThought
from langchain.tools.human.tool import HumanInputRun
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from agent_template.Origin.tool import CustomOriginTool
from agent_template.DIY.fine_tuning import CustomPromptTemplate, CustomOutputParser
from agent_template.DIY.prompts import CombinedThought, ValidationThought, AnalogyThought, DecompositionThought, PlanThought
from agent_template.DIY.prompts import DIY_INSTRUCTION, suffix1, suffix2, prefix
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain import LLMMathChain
from langchain.agents import initialize_agent
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, ZeroShotAgent
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain import LLMChain
from utils.configs import configs
import os
import sys
sys.path.append("..")
sys.path.append('../..')


os.environ["GOOGLE_CSE_ID"] = configs['tools']['google_cse_id']
os.environ["GOOGLE_API_KEY"] = configs['tools']['google_api_key']
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']


class DIYAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0, model_name=configs['model_name'])

    def get_tools(self, query):
        docs = self.retriever.get_relevant_documents(query)
        return [self.ALL_TOOLS[d.metadata["index"]] for d in docs]

    def tool_preprocessing(self):
        self.ALL_TOOLS = [
            AnalogyThought(),
            DecompositionThought(),
            # CombinedThought(),
            PlanThought(),
            ValidationThought()
        ]
        tool_lib = configs['tools']['tool_faiss_index']
        if os.path.exists(tool_lib):
            vector_store = FAISS.load_local(tool_lib, OpenAIEmbeddings())
        else:
            docs = [
                Document(
                    page_content=t.description, metadata={
                        "index": i}) for i, t in enumerate(
                    self.ALL_TOOLS)]
            vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
            vector_store.save_local(tool_lib)
            print(tool_lib)
        retriever = vector_store.as_retriever()

        return retriever

    def main_agent(self, query, dataset, is_few_shot):
        if configs['is_few_shot']:
            suffix = suffix1 + get_notes_few_shot(query, dataset, is_few_shot, '') + suffix2
        else:
            suffix = suffix1 + suffix2
        prompt = CustomPromptTemplate(
            template=prefix + DIY_INSTRUCTION + suffix,
            tools_getter=self.get_tools,
            input_variables=["input", "intermediate_steps"]
        )

        model_name = configs['model_name']
        llm = OpenAI(model_name=model_name, temperature=0)
        # LLM chain consisting of the LLM and a prompt
        self.llm_chain = LLMChain(llm=llm, prompt=prompt)

    def run(self, query, dataset):
        self.main_agent(query, dataset)
        tools = self.get_tools(query)
        print(tools)
        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser()

        agent = LLMSingleActionAgent(
            llm_chain=self.llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True)
        agent_executor.run(query)

    def run_sim(self, query_json, dataset, is_few_shot):
        self.retriever = self.tool_preprocessing()
        query = '\n'.join([
            query_json['instruct'],
            query_json['context'],
            query_json['question'],
            query_json['options'],
        ])

        if is_few_shot:
            suffix = suffix1 + \
                get_notes_few_shot(query_json, dataset, is_few_shot, '') + suffix2
        else:
            suffix = suffix1 + suffix2

        combine_prompt = ZeroShotAgent.create_prompt(
            self.ALL_TOOLS,
            prefix=prefix,
            suffix=suffix,
            format_instructions=DIY_INSTRUCTION,
            input_variables=["input", "agent_scratchpad"]
        )

        llm_chain_ = LLMChain(llm=self.llm, prompt=combine_prompt)
        self.agent = ZeroShotAgent(
            llm_chain=llm_chain_,
            tools=self.ALL_TOOLS,
            verbose=True)

        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.ALL_TOOLS, verbose=True)
        return self.agent_chain.run(input=query)

    def run_origin(self, query):
        llm = OpenAI(temperature=0, model_name=configs['model_name'])
        return llm(query)
