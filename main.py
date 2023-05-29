# -*- coding: utf-8 -*-
import os
from utils.configs import configs
from utils.parser import get_arguments

args = get_arguments()
os.environ["GOOGLE_CSE_ID"] = configs['tools']['google_cse_id']
os.environ["GOOGLE_API_KEY"] = configs['tools']['google_api_key']
os.environ["OPENAI_API_KEY"] = configs['openai_api_key']


if __name__ == '__main__':
    pass
