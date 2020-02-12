#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:31:31 2020

@author: fatimamh
"""

import os

import re
import pandas as pd
import time
import resource

from lib.file_utils import read_content
from lib.data_utils import process_data
from lib.dict_utils import process_dictionaries
from lib.dataset_utils import dataset_to_tensor

config_path = '/hits/basement/nlp/fatimamh/data_processing/configuration'
	

if __name__ == '__main__':
	config   = eval(read_content(config_path))
	
	# Step 1: CONVERT DATA FROM JSON TO CSV. CLEAN AND SHORT 
	start_time = time.time()
	'''
	process_data(files = config['json_files'], folder = config['in_folder'], 
				 clean_text = True, short_text = True, 
				 text_len = config['text_len'], summary_len = config['sum_len'], 
				 ext = '.csv')
	print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
		format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))
	'''
	# Step 2: GENERATE DICTIONARIES
	start_time = time.time()
	'''
	process_dictionaries(files = config['csv_files'], in_folder = config['in_folder'], dict_folder = config['dict_folder'])
	print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
		format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))
	'''

	# Step 3: TRANSFORM DATA
	start_time = time.time()
	dataset_to_tensor(files = config['csv_files'], folder = config['in_folder'],
					  text_len = config['text_len'], summary_len = config['sum_len'],
					  w_i_f = config['w_i_file'])
	print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
		format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))