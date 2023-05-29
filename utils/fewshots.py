# -*- coding: utf-8 -*-

import re
import sys
sys.path.append('.')
sys.path.append('..')
from notes.retriever_notes import GetNotes


def get_notes_few_shot(query, dataset, few_shot, templates_prefix):
    notes = GetNotes()
    if few_shot == 1:
        notes_text = notes.random_select(dataset, templates_prefix)
    elif few_shot == 2:
        notes_text = notes.get_notes_ret(query, dataset, templates_prefix)
    elif few_shot == 3:
        notes_text = notes.get_notes_com(query, dataset, templates_prefix)
    else:
        raise ValueError

    DIYFEWSHOTS = """"""

    DIYFEWSHOTS = DIYFEWSHOTS + notes_text + '\n'
    print("------------")
    print(DIYFEWSHOTS)
    print("------------")
    return DIYFEWSHOTS

