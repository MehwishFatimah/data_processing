#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 18:34:16 2019
Modified on Wed Nov 6
Modified on Wed Feb 12
@author: fatimamh

"""

'''-----------------------------------------------------------------------
Import libraries and defining command line arguments
-----------------------------------------------------------------------'''
import csv
import os
import numpy as np
import pandas as pd
import torch
import argparse
import time
import resource


from lib.file_utils import read_content
from lib.dict_utils import word2index


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")



'''-----------------------------------------------------------------------
Takes data, max_length and pad_index, to make sequences of same length.
  Args:
    data        : list
    max_length  : int
    pad_index   : int
  Returns:
    padded_data : list
'''    
def padding(data, max_length, pad_index):
    
    # padding to max_content
    len_data = min([max_length, len(data)])
    padded_data = np.pad(data, (0,max(0, max_length-len_data)), 'constant', constant_values = (pad_index))[:max_length]
    
    return padded_data

'''-----------------------------------------------------------------------
Takes data and word_index dictionary, to convert it into numeric form.
  Args:
    data            : list
    word_index      : dict
  Returns:
    indexed_words   : list
'''
def word_to_index(data, word_index):
        
    #words = ['<START>']
    #words += data.split()
    words = data.split()
    words += ['<END>']
    indexed_words = word2index(words, word_index)
    #indexed_words = list(map(float, indexed_words)) 
    
    return indexed_words

'''-----------------------------------------------------------------------
To tensors
Args:
    option                  : int (default = 0)
    configure               : dict
  Returns:
    corresponfing file name : str
'''
def df_to_tensor(folder, df): 

    for i, row in df.iterrows():
        print('i, sum, text:\t{}\t{}\t{}'.format(i, len(row['summary']), len(row['text'])))
        out_t = 'text_' + str(i) + '.pt'
        out_s = 'sum_' + str(i) + '.pt'
        
        text = torch.tensor(row['text'], dtype=torch.long)
        #print('text: {}\n'.format(text))
        
        summary = torch.tensor(row['summary'], dtype=torch.long)
        #print('summary: {}\n'.format(summary))
        
        file  = os.path.join(folder, out_t)
        print('\n-----------------------Saving tensor------------------------\n{}\n'.format(file))
        torch.save(text, file)
        file  = os.path.join(folder, out_s)
        print('\n-----------------------Saving tensor------------------------\n{}\n'.format(file))
        torch.save(summary, file)
    

'''-----------------------------------------------------------------------
Takes dataset and configurations, converts the dataset into tensor.
Saves the tensor on disk.
  Args:
    option  : int (default = 1)
    path    : str
  Returns:  
'''
def dataset_to_tensor(files, folder, text_len, summary_len, w_i_f):

    #print(files)
    word_index  = eval(read_content(w_i_f))
    #print(word_index)
    pad_index   = word_index['<pad>']

    #del configure
    to_file = ''
    for file in files:
        #print(file)
        file_name      = os.path.splitext(file)[0]
        f_name = file_name.split('_')[1]       
        out = os.path.join(folder, f_name)
        #print(out)
        
        csv.field_size_limit(1000000000)
        # 1.Read the content of file
        file = os.path.join(folder, file)
        df   = pd.read_csv(file, encoding = 'utf-8')
        #df = df.head(10)
        to_file = to_file + f_name + ': ' + str(len(df)) + '\n'
        
        #print(to_file)
        
        # 2.Tranfer the words to index
        df['text']    = df['text'].apply(lambda x: word_to_index(x, word_index))
        df['summary'] = df['summary'].apply(lambda x: word_to_index(x, word_index))
        
        #check lengths of summary and text before padding
        '''
        for i, row in df.iterrows():
            print('i, sum, text:\t{}\t{}\t{}'.format(i, len(row['summary']), len(row['text'])))
            #print(row['summary'])
        '''
        # 3.Padding the data
        df['text']    = df['text'].apply(lambda x: padding(x, text_len, pad_index))
        df['summary'] = df['summary'].apply(lambda x: padding(x, summary_len, pad_index))
        
        #check lengths of summary and text after padding
        
        #for i, row in df.iterrows():
            #print('i, sum, text:\t{}\t{}\t{}'.format(i, len(row['summary']), len(row['text'])))
            #print(row['summary'])
        
        df_to_tensor(out, df)
    
    file = os.path.join(folder, 'file_size')
    print(file)
    with open(file, 'w') as f:
        print(to_file, file = f)    

