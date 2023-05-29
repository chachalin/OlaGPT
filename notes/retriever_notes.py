# -*- coding: utf-8 -*-
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.chains.question_answering import load_qa_chain
from langchain import LLMChain

from utils.configs import configs
from notes.prompts import GetNotesPrompt
import os
import json
import re
import random


class GetNotes:
    """Given the document database, get search text from llm."""

    def __init__(self):
        self.pm = GetNotesPrompt()
        self.model_name = configs['model_name']
        self.llm = OpenAI(model_name=self.model_name, temperature=0)

    def llm_agent_init(self, dataset):
        if dataset == 'aqua':
            note_prompt = self.pm.construct_math_note_type()
            self.llm_chain_get_notes_type = LLMChain(
                llm=self.llm,
                prompt=note_prompt
            )
        elif dataset == 'ekar_chinese':
            note_prompt = self.pm.construct_analogy_note_type()
            self.llm_chain_get_notes_type = LLMChain(
                llm=self.llm,
                prompt=note_prompt
            )
        else:
            # todo:
            pass

    def regex_task_type(self, task_type_json):
        task_type_json.replace('"question_type":', '"task_type":')
        if task_type_json == "Analogy" or task_type_json.replace('\n', '') == 'Analogy':
            return "Analogy"
        try:
            try:
                task_type_json = re.search(r'"task_type":\s*"(.*?)"', task_type_json.lower()).group()
                if task_type_json[0] != '{':
                    task_type_json = json.loads('{' + task_type_json + '}')['task_type']
                else:
                    task_type_json = json.loads(task_type_json)['task_type']
            except:
                task_type_json = re.search(r'task_type:\s*(.*?)',  task_type_json.lower()).group()
                task_type_json = task_type_json[10:]
        except:
            return task_type_json.replace('\n', '')
        return task_type_json

    def process_test_data_type(self, dataset):
        # get question type of test data
        self.llm_agent_init(dataset)

        notes = []
        with open('./data/' + dataset + '/' + dataset + '_process_test.json', 'r', encoding='UTF-8') as f:
            for line_txt in f.readlines():
                notes.append(line_txt)

        for note in notes:
            note_json = json.loads(note)
            task_type_json = self.llm_chain_get_notes_type.run(note_json['question'] + '\n' + note_json['explanation'][0])
            task_type_json = self.regex_task_type(task_type_json)
            note_json['llm_task_type'] = task_type_json
            with open(configs['notes']['file_path'] + 'llm_test_data/' + dataset + '.json', 'a+', encoding='utf-8') as f:
                f.write(json.dumps(note_json, ensure_ascii=False) + '\n')

    def process_notes_type(self, dataset):
        # get question type of train data
        self.llm_agent_init(dataset)

        notes = []
        with open(configs['notes']['file_path'] + 'origin_data/' + dataset + '.json', 'r', encoding='UTF-8') as f:
            for line_txt in f.readlines():
                notes.append(line_txt)

        for note in notes:
            note_json = json.loads(note)
            task_type_json = self.llm_chain_get_notes_type.run(note_json['question'])
            try:
                task_type_json = self.regex_task_type(task_type_json)
                note_json['llm_task_type'] = task_type_json
                with open(configs['notes']['file_path'] + 'llm_data/' + dataset + '.json', 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(note_json, ensure_ascii=False) + '\n')
            except:
                pass

    def get_query_type(self, query, dataset):
        query_type_text = query['llm_task_type']

        embeddings = OpenAIEmbeddings()
        if not os.path.exists(configs['notes']['faiss_index_dir'] + f'/{dataset}_notes_faiss_index'):
            task_type_list = []
            with open(configs['notes']['file_path'] + 'llm_data/' + dataset + '.json', 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    try:
                        task_type_list.append(json.loads(line)['llm_task_type'])
                    except:
                        print(line)
                        exit(0)
            task_type_list = list(set(task_type_list))
            state_of_the_union = '\n'.join(task_type_list)
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=10,
                chunk_overlap=0,
                length_function=len,
            )
            texts = text_splitter.create_documents([state_of_the_union])
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(configs['notes']['faiss_index_dir'] + f'/{dataset}_notes_faiss_index')
            docs_and_scores = db.similarity_search(query_type_text, k=1)
        else:
            db = FAISS.load_local(configs['notes']['faiss_index_dir'] + f'/{dataset}_notes_faiss_index', embeddings)
            docs_and_scores = db.similarity_search(query_type_text, k=1)
        notes_type = docs_and_scores[0].page_content
        return notes_type

    def get_specific_type(self, dataset, notes_type):
        candidate_all_docs = []
        with open(configs['notes']['file_path'] + 'llm_data/' + dataset + '.json', 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                if json.loads(line)['llm_task_type'] == notes_type:
                    candidate_all_docs.append(line)
        return candidate_all_docs

    def get_notes_com(self, query, dataset, templates_prefix, file_type='txt'):
        # combine strategy (notes retrieve)
        notes_type = self.get_query_type(query, dataset)

        candidate_all_docs = self.get_specific_type(dataset, notes_type)

        random.shuffle(candidate_all_docs)
        notes_docs = candidate_all_docs[:configs['notes']['top_n']]
        return self.parse(notes_docs, templates_prefix)

    def get_notes_ret(self, query, dataset, templates_prefix, file_type='txt'):
        # judge the query type: non pre-process

        # select the same task type-top 1
        notes_type = self.get_query_type(query, dataset)
        embeddings = OpenAIEmbeddings()

        # select the dataset notes
        if not os.path.exists(configs['notes']['faiss_index_dir'] + dataset + f'/{notes_type}_faiss_index'):
            if file_type == 'txt':
                candidate_question_set = self.get_specific_type(dataset, notes_type)
                candidate_question_set = '\n'.join(candidate_question_set)
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=100,
                    chunk_overlap=0,
                    length_function=len,
                )
                if len(candidate_question_set) == 0:
                    raise ValueError
                texts = text_splitter.create_documents([candidate_question_set])

                db = FAISS.from_documents(texts, embeddings)
                db.save_local(configs['notes']['faiss_index_dir'] + dataset + f'/{notes_type}_faiss_index')
                docs_and_scores = db.similarity_search(query['question'], k=configs['notes']['top_n'])
        else:
            db = FAISS.load_local(configs['notes']['faiss_index_dir'] + dataset + f'/{notes_type}_faiss_index', embeddings)
            docs_and_scores = db.similarity_search(query['question'], k=configs['notes']['top_n'])

        notes_docs = []
        if len(docs_and_scores) == 0:
            return ''
        for d in docs_and_scores:
            notes_docs.append(d.page_content)
        return self.parse(notes_docs, templates_prefix)

    def parse(self, notes_docs, templates_prefix):
        return self.pm.construct_notes(notes_docs, templates_prefix)

    def random_select(self, dataset, templates_prefix):
        # random strategy (notes retrieve)
        with open(configs['notes']['file_path'] + 'llm_data/' + dataset + '.json', 'r', encoding='UTF-8') as f:
            all_docs = f.readlines()

        random.shuffle(all_docs)
        notes_docs = all_docs[:configs['notes']['top_n']]
        return self.parse(notes_docs, templates_prefix)


