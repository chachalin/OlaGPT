# -*- coding: utf-8 -*-
import os
from langchain import OpenAI
from langchain.agents import Tool, load_tools
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from utils.configs import configs
from utils import configs


def fake_func(inp: str) -> str:
    return "foo"


def get_tools(query: str):
    model_name = configs['model_name']
    llm = OpenAI(model_name=model_name, temperature=0)
    builtin_tools = load_tools(['google-search', 'wikipedia'], llm=llm)
    fake_tools = [
        Tool(
            name=f"foo-{i}",
            func=fake_func,
            description=f"a silly function that you can use to get more information about the number {i}"
        )
        for i in range(99)
    ]
    all_tools = builtin_tools + fake_tools
    # tools retrieval
    tool_lib = configs['tools']['tool_faiss_index']
    if os.path.exists(tool_lib):
        vector_store = FAISS.load_local(tool_lib, OpenAIEmbeddings())
    else:
        docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(all_tools)]
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        vector_store.save_local(tool_lib)
    retriever = vector_store.as_retriever()

    docs = retriever.get_relevant_documents(query)
    return [all_tools[d.metadata["index"]] for d in docs]