# -*- coding: utf-8 -*-
from data.loading import load_predictions


def get_candidate_predictions(model_to_vote, dataset, few_shot):

    pred_dict = {}
    for model_name in model_to_vote:
        pred_dict[model_name] = load_predictions(model_name, dataset, few_shot)

    return pred_dict
