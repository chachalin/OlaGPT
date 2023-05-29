# -*- coding: utf-8 -*-
configs = {
    'model_name': 'gpt-3.5-turbo',
    'openai_api_key': 'your key',
    'agents': {

    },
    'data': {

    },
    'tools': {
        'google_cse_id': 'your id',
        'google_api_key': 'your key',
        'tool_faiss_index': './tools/tool_faiss_index',
    },
    'notes': {
        'file_path': './notes/note_data/',
        'file_name': 'origin_notes.json',
        'faiss_index_dir': './notes/faiss_index_dir/',
        'top_n': 3
    }
}

few_shot_suffix = {
    0: '',  # zero shot
    1: '_few_shot_random',
    2: '_few_shot_retrieval',
    3: '_few_shot_combine',
}
