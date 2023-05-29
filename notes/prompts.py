# -*- coding: utf-8 -*-
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.output_parsers import RegexParser
import json


class GetNotesPrompt(object):
    def __init__(self):
        pass

    def construct_analogy_note_type(self):
        prompt_template = """
        You are the examiner of the Chinese Civil Service Examination,
        and you need to judge the specific question types of the following analogy questions and don't give an explanation.
        Question: {question}
        Answer: The output must only be in a strict JSON format: "task_type": "question type".
        """
        return PromptTemplate(template=prompt_template, input_variables=["question"])

    def construct_math_note_type(self):
        prompt_template = """
        As a mathematics professor, you need to judge the type of the following question and don't give an explanation
        Question: {question}
        Answer: The output must only be in a strict JSON format: "task_type": "question type".
        """
        return PromptTemplate(template=prompt_template, input_variables=["question"])

    def construct_note(self, note, templates_prefix):
        return f"""
        This is a kind of {note['llm_task_type']}: {note['question']} \n
        {templates_prefix} {note['explanation']}.
        Therefore, the answer is: {{"Answer": "{note['answer']}"}}.
        """

    def construct_notes(self, notes, templates_prefix):
        text = 'Here are some similar questions, and the solving process that may assist you in answering your questions: \n'
        for note in notes:
            note_json = json.loads(note)
            text += self.construct_note(note_json, templates_prefix)
            text += '\n'
        return text
