# -*- coding: utf-8 -*-

def get_query_format(dataset, query_json):
    # intention refinement and enhancement and
    if dataset == 'aqua':
        query_json['instruct'] = f"Now give you the {query_json['llm_task_type']} question and choices:"
    elif dataset == 'ekar_chinese':
        query_json['instruct'] = "Now give you the analogy question and choices:"

    prefix = """The answer must end with json format: {Answer: one of options[A,B,C,D,E]}."""

    final_query = '\n'.join([
        query_json['instruct'],
        query_json['context'],
        query_json['question'],
        query_json['options'],
        prefix
    ])

    return query_json, final_query
