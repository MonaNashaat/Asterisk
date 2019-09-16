#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:53:47 2019

@author: mona
"""

import pandas as pd
import numpy as np
import csv
from .doc_preprocessors import TSVDocPreprocessor

from .spacy_parser import Spacy

from .corpus_parser import CorpusParser

def read_data(input_file, y_column):
    df_test = pd.read_csv(input_file, sep=',')    
    df_test = df_test.rename(columns={y_column:'prediction'})
    df_test=df_test.reset_index()
    print("Columns: \n")
    print(df_test.columns)
    return df_test

def doc_creation(df_features):
    # write the subset to a .csv and convert it to a .tsv file
    df_features.to_csv('dataset.csv',header=False)
    #	
    csv.writer(open('dataset.tsv', 'w+'), delimiter='	').writerows(csv.reader(open("dataset.csv")))
    doc_preprocessor = TSVDocPreprocessor('dataset.tsv')
    corpus_parser = CorpusParser(parser=Spacy())
    corpus_parser.apply(doc_preprocessor)


