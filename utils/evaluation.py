# -*- coding: utf-8 -*-



import os
import re
import random
import json
import threading
import time
from langchain.evaluation.qa import QAEvalChain
from langchain.agents import AgentExecutor
from typing import Dict, List
from data.loading import load_dataset
from utils.parse_option import parse_option
from utils.vote import get_candidate_predictions
from utils.configs import few_shot_suffix


def save_predictions(agent: AgentExecutor, dataset, args) -> List:
    # get args
    model_name = args.model_name
    data_path = args.dataset
    eval_num = args.eval_num
    eval_full = args.eval_full
    is_random = args.is_random
    few_shot = args.few_shot
    n_split = args.n_split
    is_vote = args.is_vote
    model_to_vote = eval(args.model_to_vote)

    if not eval_full:
        if is_random and len(dataset) >= eval_num:
            dataset = random.sample(dataset, eval_num)
        else:
            dataset = dataset[:eval_num]

    # divide each split as equal as possible
    quotient = len(dataset) // n_split
    remainder = len(dataset) % n_split
    split_size = [quotient] * n_split
    for i in range(remainder):
        split_size[i] += 1
    for i in range(1, len(split_size)):
        split_size[i] += split_size[i - 1]

    # make predictions
    pred_func = predict_for_vote if is_vote else predict
    predictions = []
    threads = []
    start = 0
    for size in split_size:
        split_dataset = dataset[start:start + size]
        t = threading.Thread(
            target=pred_func,
            args=(
                agent,
                split_dataset,
                predictions,
                start))
        threads.append(t)
        t.start()
        start += size

    for t in threads:
        t.join()

    predictions = sorted(predictions, key=lambda x: x['question_id'])

    # save predictions
    save_dir = f'data/predictions/{model_name}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if is_vote:
        model_to_vote = sorted(model_to_vote)
        vote_dir = '_'.join(model_to_vote)
        save_dir = f'{save_dir}/{vote_dir}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    save_file = f'{data_path}_predict'
    save_file = save_file + few_shot_suffix[few_shot]
    save_file += '.json'
    with open(f'{save_dir}/{save_file}', 'w', encoding='utf-8') as f:
        json.dump({'predictions': predictions},
                  f, indent=4, ensure_ascii=False)

    return predictions


def evaluation(agent: AgentExecutor, llm, args) -> Dict:
    # get args
    model_name = args.model_name
    data_path = args.dataset
    eval_mode = args.eval_mode
    few_shot = args.few_shot
    is_vote = args.is_vote
    model_to_vote = eval(args.model_to_vote)

    # initialize the dataset
    dataset = load_dataset(data_path)
    if is_vote:
        # append candidate outputs into dataset
        candidate_predictions = get_candidate_predictions(
            model_to_vote, data_path, few_shot)
        for i, data in enumerate(dataset):
            tmp_output = []
            for model, pred in candidate_predictions.items():
                answer = pred[i]['output']
                tmp_output.append(f'{model}: {answer}')
            data['candidate_outputs'] = tmp_output

    predictions = save_predictions(agent, dataset, args)

    # evaluate performance
    if eval_mode == 'llm':
        eval_chain = QAEvalChain.from_llm(llm)
        graded_outputs = eval_chain.evaluate(
            dataset,
            predictions,
            question_key="question",
            prediction_key="output")
        for i, prediction in enumerate(predictions):
            prediction['grade'] = graded_outputs[i]['text']
        save_file = f'{data_path}_llm'
    else:
        predictions = evaluation_regex(predictions)
        save_file = f'{data_path}_regex'

    # filter correct and incorrect examples
    correct = [pred for pred in predictions if pred['grade'] == "CORRECT"]
    incorrect = [pred for pred in predictions if pred['grade'] == "INCORRECT"]

    result = {
        'correct_num': len(correct),
        'incorrect_num': len(incorrect),
        'accuracy': len(correct) / (len(correct) + len(incorrect)),
        'correct': correct,
        'incorrect': incorrect
    }

    # save result
    save_dir = f'data/predictions/{model_name}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if is_vote:
        model_to_vote = sorted(model_to_vote)
        vote_dir = '_'.join(model_to_vote)
        save_dir = f'{save_dir}/{vote_dir}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    save_file = save_file + few_shot_suffix[few_shot]
    save_file += '.json'
    with open(f'{save_dir}/{save_file}', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return result


def retry(agent, data, cnt):
    count = cnt
    pred = {}
    while count:
        try:
            time.sleep(20)
            pred = agent(data)
            break
        except Exception as e:
            pass
        count -= 1
    return pred


def predict(agent, dataset, predictions, offset):
    for i, data in enumerate(dataset):
        print(f'predicting question: {offset+i}')
        format_instruct = 'The answer must end with json format: {Answer: one of options[A,B,C,D,E]}.'
        full_input_json = {
            'instruct': data['instruct'],
            'context': data['context'],
            'question': data['question'],
            'options': data['options'],
            'llm_task_type': data['llm_task_type']
        }
        full_input = json.dumps(full_input_json, ensure_ascii=False)
        print(full_input)
        new_data = {
            "question_id": offset + i,
            "input": full_input,
            "answer": data["answer"],
            "explanation": data["explanation"]
        }
        try:
            pred = agent(new_data)
            predictions.append(pred)
        except Exception as e:
            pred = retry(agent, new_data, 3)
            if len(pred) > 0:
                predictions.append(pred)
            else:
                predictions.append({
                    "question_id": offset + i,
                    "input": new_data["input"],
                    "answer": new_data["answer"],
                    "explanation": new_data["explanation"],
                    "output": "I don't know the answer"
                })
                print(e)


def predict_for_vote(agent, dataset, predictions, offset):
    for i, data in enumerate(dataset):
        format_instruct = 'The answer must end with json format: {Answer: one of options[A,B,C,D,E]}.'
        full_input_json = {
            'instruct': data['instruct'],
            'context': data['context'],
            'question': data['question'],
            'options': data['options'],
            'candidate_output': 'Candidate Outputs:\n' + '#'.join(data['candidate_outputs']),
            'llm_task_type': data['llm_task_type']
        }
        full_input = json.dumps(full_input_json, ensure_ascii=False)
        print(full_input)
        new_data = {
            "question_id": offset + i,
            "input": full_input,
            "answer": data["answer"],
            "explanation": data["explanation"],
            "candidate_outputs": data["candidate_outputs"]
        }
        try:
            pred = agent(new_data)
            predictions.append(pred)
        except Exception as e:
            pred = retry(agent, new_data, 3)
            if len(pred) > 0:
                predictions.append(pred)
            else:
                predictions.append({
                    "question_id": offset + i,
                    "input": new_data["input"],
                    "output": "I don't know the answer",
                    "answer": new_data["answer"],
                    "explanation": new_data["explanation"],
                    "candidate_outputs": data["candidate_outputs"]
                })
                print(e)


def evaluation_regex(predictions):

    for pred in predictions:
        answer = pred['answer'][0]
        output = pred['output']

        # extract option
        option = parse_option(output)

        # evaluate with the gold answer
        if answer == option:
            pred['grade'] = "CORRECT"
        else:
            pred['grade'] = "INCORRECT"

    return predictions
