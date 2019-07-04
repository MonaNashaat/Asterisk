#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:53:47 2019

@author: mona
"""

import pandas as pd
import csv
from snorkel.parser import TSVDocPreprocessor
from snorkel.parser.spacy_parser import Spacy
from snorkel.parser import CorpusParser
from snorkel import SnorkelSession
from snorkel.models import Document
from snorkel.models import Sentence
from snorkel.models import candidate_subclass
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import RegexMatchEach
from util import load_external_labels
from snorkel.annotations import load_gold_labels
from snorkel.learning import GenerativeModel
import numpy as np

def read_data(input_file, y_column):
    df_test = pd.read_csv(input_file, sep=',')    
    df_test = df_test.rename(columns={y_column:'prediction'})
    df_test=df_test.reset_index()
    print("Columns: \n")
    print(df_test.columns)
    return df_test

def doc_creation(df_features, session):
    # write the subset to a .csv and convert it to a .tsv file
    df_features.to_csv('dataset.csv',header=False)
    #	
    csv.writer(open('dataset.tsv', 'w+'), delimiter='	').writerows(csv.reader(open("dataset.csv")))
    doc_preprocessor = TSVDocPreprocessor('dataset.tsv')
    corpus_parser = CorpusParser(parser=Spacy())
    corpus_parser.apply(doc_preprocessor)
    print("Documents:", session.query(Document).count())
    print("Sentences:", session.query(Sentence).count())
    
def cand_creation(df_subset):
    #Generating Candidates
    global T1 
    T1 = candidate_subclass('T1',['Features'])
    r = '^-?\d*.\d*'
    ngrams         = Ngrams(n_max=300)
    regex_matcher = RegexMatchEach(rgx = r)
    cand_extractor = CandidateExtractor(T1, [ngrams], [regex_matcher])
    return T1, cand_extractor

def splitting_sets(session, cand_extractor, T1):
    docs = session.query(Document).order_by(Document.name).all()
    train_sents = set()
    dev_sents   = set()
    test_sents  = set()
    counter = 0
    for i, doc in enumerate(docs):
        for s in doc.sentences:     
            if i % 10 == 8:
                dev_sents.add(s)
            elif i % 10 == 9:
                test_sents.add(s)
            else:
                train_sents.add(s)
    # test print: we print the sizes of the document splits
    print("Train sent : # " + str(len(train_sents)))
    print("Dev sent : # " + str(len(dev_sents)))
    print("Test sent : # " + str(len(test_sents)))
    #session.rollback()
    #Next, we'll apply the candidate extractor to the three sets of sentences. The results will be persisted 
    #in the database backend.
    for i, sents in enumerate([train_sents, dev_sents, test_sents]):
        cand_extractor.apply(sents, split=i)
        print("Number of candidates:", session.query(T1).filter(T1.split == i).count())   
    
    return train_sents, dev_sents, test_sents

def splitting_sets_with_labels(session, cand_extractor, T1, df_ground):
    docs = session.query(Document)#.order_by(Document.name).all()
    train_sents = list()
    dev_sents   = list()
    test_sents  = list()
    train_ground = []
    val_ground = []
    test_ground = []
    for i, doc in enumerate(docs):
        for s in doc.sentences:     
            if i % 10 == 8:
                dev_sents.append(s)
                #val_ground= val_ground.append(df_ground.iloc[i])
                val_ground = np.append(val_ground, df_ground.iloc[i])
                #val_ground.append(pd.Series(df_ground.iloc[i]))
                #print(df_ground.iloc[i])
            elif i % 10 == 9:
                test_sents.append(s)
                test_ground = np.append(test_ground, df_ground.iloc[i])
                #print(df_ground.iloc[i])
                #test_ground.append(pd.Series(df_ground.iloc[i]))

            else:
                train_sents.append(s)
                train_ground = np.append(train_ground, df_ground.iloc[i])
                

    # test print: we print the sizes of the document splits
    print("Train sent : # " + str(len(train_sents)))
    print("Dev sent : # " + str(len(dev_sents)))
    print("Test sent : # " + str(len(test_sents)))
    #session.rollback()
    #Next, we'll apply the candidate extractor to the three sets of sentences. The results will be persisted 
    #in the database backend.
    for i, sents in enumerate([train_sents, dev_sents, test_sents]):
        cand_extractor.apply(sents, split=i)
        print("Number of candidates:", session.query(T1).filter(T1.split == i).count())   
    
    return train_sents, dev_sents, test_sents, train_ground, val_ground, test_ground

def prepare_gold_labels(session, T1, df_test, columns_list):
    #Preparing gold labels
    index_list=[]
    feature_list=[]

    for c in session.query(T1).filter(T1.Features,T1.split == 1).all():
        index_list.append(c.Features.sentence.document_id-1)

    que = session.query(T1).filter(T1.Features).filter(T1.split==1).all()
    for item in que:
        feature_list.append(item.Features.stable_id)

    for c in session.query(T1).filter(T1.Features,T1.split == 2).all():
        index_list.append(c.Features.sentence.document_id-1)

    que = session.query(T1).filter(T1.Features).filter(T1.split==2).all()
    for item in que:
        feature_list.append(item.Features.stable_id)


    ## creating the overall gold-label dataframe
    df_gold_label = pd.DataFrame(index_list)
    df_gold_label.columns=['Index']
    df_gold_label['Features'] = feature_list

    df_subset = df_test
    df_subset.rename(columns={'index':'Index'}, inplace=True)
    df_subset = df_subset.drop(columns_list, axis = 1)

    df_gold_label = df_gold_label.merge(df_subset, on=['Index'], how='left', indicator= True)

    df_gold_label = df_gold_label.drop(['Index','_merge'], axis=1)
    df_gold_label.rename(columns={'prediction':'label'}, inplace=True)
    df_gold_label.loc[df_gold_label ['label'] == 0, 'label'] = -1
    df_gold_label = df_gold_label.set_index('Features')
    # Writing the gold labels into a .tsv file
    df_gold_label.to_csv('gold_labels.csv')
    csv.writer(open('gold_labels.tsv', 'w+'), delimiter='	').writerows(csv.reader(open("gold_labels.csv")))

    return df_gold_label


def Loading_sets(session, T1):
    # Generating and modeling noisy training labels
    # Using a labeled _development set_
    # Loading the labels from both dev set and test set
    missed = load_external_labels(session, T1, annotator_name='gold')

    L_gold_dev = load_gold_labels(session, annotator_name='gold', split=1)
    L_gold_test = load_gold_labels(session, annotator_name='gold', split=2)
    L_gold_train = load_gold_labels(session, annotator_name='gold', split=0)
    
    return L_gold_dev, L_gold_test, L_gold_train


def Fitting_Gen_Model(L_train):
    gen_model = GenerativeModel()
    gen_model.train(L_train, epochs=100, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1e-6)

    #-------------------------
    print(gen_model.weights.lf_accuracy)
    print(gen_model.weights.class_prior)
    #-------------------------
    #We now apply the generative model to the training candidates to get the noise-aware training label set. We'll refer to these as the training marginals:
    train_marginals = gen_model.marginals(L_train)
    return gen_model, train_marginals


def Calculate_labeling_accuracy(threeshold, columns_list, train_cands, train_marginals, df_test):
    doc_id_list = []
    i=0
    for cand in train_cands:
        i=i+1
        #print(cand.Features.sentence.document_id-1)
        doc_id_list.append(cand.Features.sentence.document_id-1)
    #print(i)
    cand_list=[]
    j=0
    for cand in train_cands:
        j=j+1
        for feat in cand.Features.sentence.document.sentences:
            each_cand = feat.text.strip().split('@')
            cand_list.append(each_cand)
    #print(j)    
                
    df_verify = pd.DataFrame.from_records(cand_list)
    print(df_verify.shape)
    print(len(doc_id_list))

    #print(df_verify.shape)
    #print(len(doc_id_list))
    #print(len(cand_list))
    df_verify.columns = columns_list
    df_verify ['Index'] = doc_id_list
    
    #print ("1- df_verify columns")
    #print (df_verify.columns)
    
    cols = df_verify.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_verify = df_verify[cols]

    #print ("2- df_verify columns")
    #print (cols)
    
    
    df_verify ['Label'] = train_marginals.tolist()
    df_verify.loc[df_verify ['Label'] >= 0.5, 'Label'] = 1
    df_verify.loc[df_verify ['Label'] < 0.5, 'Label'] = -1

    
    #print ("3- df_verify columns")
    #print (df_verify.columns)
    
    df_copy = df_test
    
    #print ("1- df_copy columns")
    #print (df_copy.columns)
    
    
    df_copy = df_copy.drop(columns_list, axis = 1)
    
    #print ("1- df_copy columns")
    #print (df_copy.columns)
    
    #merging
    df_verify = df_verify.merge(df_copy, on=['Index'], how='left', indicator= True)
    df_verify.loc[df_verify ['prediction'] == 0, 'prediction'] = -1
    df_verify=df_verify.drop(['_merge'], axis=1)

    #calculating the labeling accuracy
    counter = 0
    for index, row in df_verify.iterrows():
        if row['Label'] == float(row['prediction']):
            counter = counter+1

    labeling_accuracy = float(counter)/len(doc_id_list)
    return labeling_accuracy

def Calculate_labeling_accuracy_reef(threeshold, columns_list, train_cands, train_marginals, df_test):
    doc_id_list = []
    i=0
    for cand in train_cands:
        i=i+1
        #print(cand.Features.sentence.document_id-1)
        doc_id_list.append(cand.Features.sentence.document_id-1)
    #print(i)
    cand_list=[]
    j=0
    for cand in train_cands:
        j=j+1
        for feat in cand.Features.sentence.document.sentences:
            each_cand = feat.text.strip().split('@')
            cand_list.append(each_cand)
    #print(j)    
                
    df_verify = pd.DataFrame.from_records(cand_list)
    #print(df_verify.shape)
    #print(len(doc_id_list))
    #print(len(cand_list))
    df_verify.columns= columns_list
    df_verify ['Index'] = doc_id_list
    
    #print ("1- df_verify columns")
    #print (df_verify.columns)
    
    cols = df_verify.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_verify = df_verify[cols]

    #print ("2- df_verify columns")
    #print (cols)
    
    
    df_verify ['Label'] = train_marginals
    df_verify.loc[df_verify ['Label'] >= 0.5, 'Label'] = 1
    df_verify.loc[df_verify ['Label'] < 0.5, 'Label'] = -1

    
    #print ("3- df_verify columns")
    #print (df_verify.columns)
    
    df_copy = df_test
    
    #print ("1- df_copy columns")
    #print (df_copy.columns)
    
    
    df_copy = df_copy.drop(columns_list, axis = 1)
    
    #print ("1- df_copy columns")
    #print (df_copy.columns)
    
    #merging
    df_verify = df_verify.merge(df_copy, on=['Index'], how='left', indicator= True)
    df_verify.loc[df_verify ['prediction'] == 0, 'prediction'] = -1
    df_verify=df_verify.drop(['_merge'], axis=1)

    #calculating the labeling accuracy
    counter = 0
    for index, row in df_verify.iterrows():
        if row['Label'] == float(row['prediction']):
            counter = counter+1

    labeling_accuracy = float(counter)/len(doc_id_list)
    return labeling_accuracy


def Create_discrimintative_input (cands, train_marginals, columns_list):
    
    cand_list=[]
    for cand in cands:
        for feat in cand.Features.sentence.document.sentences:
            each_cand = feat.text.strip().split('@')
            cand_list.append(each_cand)

    Discrimintative_input = pd.DataFrame.from_records(cand_list)
    Discrimintative_input.columns=columns_list

    Discrimintative_input ['Label'] = train_marginals.tolist()
    Discrimintative_input.loc[Discrimintative_input ['Label'] >= 0.5, 'Label'] = 1
    Discrimintative_input.loc[Discrimintative_input ['Label'] < 0.5, 'Label'] = -1
    Discrimintative_input = Discrimintative_input.convert_objects(convert_numeric=True)
    
    return Discrimintative_input
    
