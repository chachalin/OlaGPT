# -*- coding: utf-8 -*-
from langchain import OpenAI
from agent_template.ValidationThought.prompts import VALID_INSTRUCTION, REPORT_INSTRUCTION
from utils.configs import configs
import json


class ValidAgent:
    # not tested
    def __init__(self):
        self.main_llm = OpenAI(temperature=0, model_name=configs['model_name'])
        self.valid_llm = OpenAI(
            temperature=0,
            model_name=configs['model_name'])

    """
    {"is_correct": "no", "suggestion": "The correct answer is D. Peanut: peanut butter. Paddy is the crop that produces rice, similarly,
    peanut is the crop that produces peanut butter. Walnut crisp is a snack made from walnuts, which is not an appropriate analogy for paddy:rice."}
    """

    def parse_correction(self, llm_output: str):
        # Check if agent should finish
        try:
            correction = json.loads(llm_output)
            if correction['is_correct'] == 'yes':
                return 'it is ok'
            else:
                return correction['suggestion']
        except BaseException:
            # todo: add some file
            raise ValueError(f'could not parse {llm_output}')
            # pass

    def run(self, query):
        self.answer = self.main_llm(query)
        for i in range(2):
            correct_text = self.is_correct_llm(query)
            if self.parse_correction(correct_text) == 'it is ok':
                return self.answer
            else:
                re_answer = REPORT_INSTRUCTION.format(
                    suggestion=correct_text, question=query)
                self.answer = self.main_llm(re_answer)
        return self.answer

    def is_correct_llm(self, query):
        is_correct_text = self.valid_llm(
            VALID_INSTRUCTION + '\n\n' + query + '\n' + self.answer)
        return is_correct_text
