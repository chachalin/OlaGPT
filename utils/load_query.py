# -*- coding: utf-8 -*-



import json


def load_query(query):
    """query must be a json string"""
    try:
        query_json = json.loads(query)
    except Exception:
        raise ValueError('query is not a json string')
    return query_json
