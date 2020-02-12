#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:28:08 2019
Modified on Wed Nov 6
Modified on Wed Feb 12
@author: fatimamh
"""

'''-----------------------------------------------------------------------
Import libraries and defining command line arguments
-----------------------------------------------------------------------'''
import os
import pandas as pd
import argparse
import time
import resource

'''-----------------------------------------------------------------------
'''
class dictionary_cell():
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.index = 0
        self.insert_specials()

    def insert(self, word):
        if word not in self.word_index:
            self.word_index[word] = self.index
            self.index_word[self.index] = word
            self.index += 1

    def insert_specials(self):
        words = ['<unk>','<pad>','<START>','<END>']

        for word in words:
            self.insert(word)

'''-----------------------------------------------------------------------
Takes text and generates dictionaries.
  Args:
    text        : str
  Returns:
    word_index  : dict
    index_word  : dict
'''
def generate_dictionary(text):

    dictionary = dictionary_cell()
    #print(dictionary)
    tokens = text.split()
    #print(tokens)
    for token in tokens:
        #print(token)
        dictionary.insert(token)

    word_index = dictionary.word_index
    index_word = dictionary.index_word

    return word_index, index_word

'''-----------------------------------------------------------------------
Takes words and word_index dictionary, converts words to corresponding indices.
Returns them as list.
  Args:
    words       : list
    word_index  : dict
  Returns:
    result      : list
    index_word  : dict
'''
def word2index(words, word_index):
   
    index = []
    for word in words:
        if word in word_index:
            index.append(word_index[word])
        else: 
            index.append(word_index['<unk>'])
    return index

'''-----------------------------------------------------------------------
Takes files, in_folder and out_folder, merge the files content to make dictionaries.
Stores dictionaries in out_folder with specific names.
Returns them as list.
  Args:
    words        : list
    word_index   : dict
  Returns:   
'''
def process_dictionaries(files, in_folder, dict_folder):
    print(files)
    #combined dictionary
    dfs = []
    #print(len(dfs))
    for file in files:
        file  = os.path.join(in_folder, file)
        df = pd.read_csv(file, encoding = 'utf-8')
        #print('\n=========================================')
        #print('Training data:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))
        df['merge'] = df[['text', 'summary']].apply(lambda x: ' '.join(x), axis = 1)
        dfs.append(df)
        
    big_df = pd.concat(dfs, axis = 0, ignore_index = True)
    #print(len(big_df))  

    text = big_df['merge'].str.cat(sep = ' ')
    #print(text)
    word_index, index_word = generate_dictionary(text)
    
    file = os.path.join(dict_folder, 'word_index')
    with open(file, 'w') as f:
        print(word_index, file = f)

    file = os.path.join(dict_folder, 'index_word')
    with open(file, 'w') as f:
        print(index_word, file = f)
    
    print(len(word_index))
    file = os.path.join(dict_folder, 'dict_len')
    with open(file, 'w') as f:
        print(len(word_index), file = f) 
    
