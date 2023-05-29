# -*- coding: utf-8 -*-



import re


def parse_option(answer: str):
    # extract option by regex
    # match = re.search(r'\"Answer\": \"(.*?)\"', answer)
    # match = re.search(r'\{\D*?Answer\D+?([A-Z])', answer)
    match = re.search(r'\{*?.*?[Aa]nswer[^A-Z]*?([A-F])[^a-zA-Z]+', answer)
    if match:
        option = match.group(1)
    else:
        option = answer
    final_answer = option
    return final_answer
