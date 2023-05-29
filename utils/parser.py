# -*- coding: utf-8 -*-
import argparse


def get_arguments():
    """the global argument parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', type=str, default='How many people live in canada as of 2023?',
                        help='Test the agent with one question.')
    parser.add_argument('--is_eval', type=bool, default=False,
                        help='Automatically evaluate the agent on the given dataset.')
    parser.add_argument('--dataset', type=str, default='agent-search-calculator',
                        help='The dataset for evaluating the agent.')
    parser.add_argument('--eval_num', type=int, default=10,
                        help='Sample a subset to test evaluation.')
    parser.add_argument('--eval_full', type=bool, default=False,
                        help='Whether to evaluate on full data.')
    parser.add_argument('--eval_mode', type=str, default='regex', choices=['llm', 'regex'],
                        help='Determine evaluation mode.')
    parser.add_argument('--vote_mode', type=str, default='regex', choices=['llm', 'regex'],
                        help='Determine voting mode.')
    parser.add_argument('--is_random', type=bool, default=False,
                        help='Whether to random sample the dataset.')
    # parser.add_argument('--explain', type=bool, default=False,
    #                     help='Whether to explain when llm generating answers.')
    parser.add_argument('--n_split', type=int, default=5,
                        help='Divide dataset to n split to speed up evaluation by parallelling.')
    parser.add_argument('--few_shot', type=int, default=0, choices=[0, 1, 2, 3],
                        help='few shot: [0: not use; 1: random; 2: retrieval, 3: combine]')
    parser.add_argument('--is_vote', type=bool, default=False,
                        help='Whether to evaluate vote agent.')
    parser.add_argument('--model_to_vote', type=str, default="['at', 'pt']",
                        help='Choose the models to vote.')
    parser.add_argument('--model_name', type=str, choices=['turbo', 'cos', 'cot_sc', 'react', 'auto_cot', 'diy',
                                                           'at', 'dt', 'pt', 'st', 'cos_llm_vote', 'cos_reg_vote', 'dst'],
                        required=True, help='Select which model to use.')
    args = parser.parse_args()
    return args
