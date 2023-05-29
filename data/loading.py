# -*- coding: utf-8 -*-
import json
from typing import List, Dict
from utils.configs import few_shot_suffix


def load_dataset(uri: str) -> List[Dict]:
    from datasets import load_dataset

    dataset = load_dataset(f"data/{uri}")
    key = list(dataset.keys())[0]  # key: train or test
    return [d for d in dataset[key]]


def load_predictions(model_name, dataset, few_shot):
    pred_dir = f'data/predictions/{model_name}'
    pred_file = f'{dataset}_predict'
    pred_file = pred_file + few_shot_suffix[few_shot]
    pred_file += '.json'
    with open(f'{pred_dir}/{pred_file}', 'r', encoding='utf-8') as f:
        predictions = json.load(f)['predictions']

    return predictions


def load_evaluation(model_name, dataset, few_shot):
    pred_dir = f'data/predictions/{model_name}'
    pred_file = f'{dataset}_regex'
    pred_file = pred_file + few_shot_suffix[few_shot]
    pred_file += '.json'
    with open(f'{pred_dir}/{pred_file}', 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    return predictions
