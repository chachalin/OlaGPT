# -*- coding: utf-8 -*-
import logging
from abc import abstractmethod
from langchain.agents import Tool, AgentExecutor, Agent, ZeroShotAgent, BaseMultiActionAgent, AgentOutputParser, \
    BaseSingleActionAgent
from langchain.agents.tools import InvalidTool
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain import OpenAI, SerpAPIWrapper, PromptTemplate, FewShotPromptTemplate, BasePromptTemplate
from langchain.chains.base import Chain
from langchain.input import get_color_mapping
from langchain.schema import AgentAction, AgentFinish, BaseLanguageModel, BaseMessage
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
from langchain.tools import BaseTool
from pydantic import BaseModel, root_validator

logger = logging.getLogger()


class LLMMultiActionAgent(BaseMultiActionAgent):

    llm_chain: LLMChain
    output_parser: AgentOutputParser
    stop: List[str]

    @property
    def input_keys(self) -> List[str]:
        return list(set(self.llm_chain.input_keys) - {"intermediate_steps"})

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
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps, stop=self.stop, **kwargs
        )
        return self.output_parser.parse(output)

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
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps, stop=self.stop, **kwargs
        )
        return self.output_parser.parse(output)

    def tool_run_logging_kwargs(self) -> Dict:
        return {
            "llm_prefix": "\n",
            "observation_prefix": "" if len(self.stop) == 0 else self.stop[0],
        }
