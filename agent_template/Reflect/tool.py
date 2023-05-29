# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from agent_template.Reflect.react_cls import ReactAgent, ReactReflectAgent


class ReactTool(BaseTool):
    """Tool that adds the origin api."""
    name = "ReactTool"
    description = (""
                   )

    def _run(self, query: str) -> str:
        """Use the tool."""
        ra = ReactAgent(query)
        return ra.run()

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("ReactTool does not support async")


class ReactReflectTool(BaseTool):
    """Tool that adds the origin api."""
    name = "ReactReflectTool"
    # description = "带有反思策略的模板，适用于"
    description = (""
                   )

    def _run(self, query: str) -> str:
        """Use the tool."""
        ra = ReactReflectAgent(query)
        return ra.run()

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("ReflectTool does not support async")
