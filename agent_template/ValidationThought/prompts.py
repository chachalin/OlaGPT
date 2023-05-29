# -*- coding: utf-8 -*-

# test paper corrector
VALID_INSTRUCTION = """
You are a test paper corrector, responsible for checking whether the questions done by others are correct. Given the question and other people's answers, you can judge whether it is right or wrong, and if you judge that the solution to the problem is wrong, you need to give optimization suggestions.
Your answer format must be in json format: {"is_correct": "choose one of [yes, no]", "suggestion": "your optimization suggestions about this answer"
"""

REPORT_INSTRUCTION = """
You got a question wrong the last time you answered it, and the expert has some suggestions for you. I hope you can give a new answer based on expert advice.
Suggestion: {suggestion}
Question: {question}
"""
