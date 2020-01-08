#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:53:47 2019

@author: mona
"""

import pandas as pd
import csv
from Asterisk.heuristics_generator.parser.doc_preprocessors import TSVDocPreprocessor
from Asterisk.heuristics_generator.parser.spacy_parser import Spacy
from Asterisk.heuristics_generator.parser.corpus_parser import CorpusParser
from Asterisk import AsteriskSession
from Asterisk.heuristics_generator.models import Document
from Asterisk.heuristics_generator.models import Sentence
from Asterisk.heuristics_generator.models import candidate_subclass
from Asterisk.heuristics_generator.candidates import Ngrams, CandidateExtractor
from Asterisk.heuristics_generator.matchers import RegexMatchEach
from Asterisk.heuristics_generator.util import load_external_labels
from Asterisk.heuristics_generator.annotations import load_gold_labels
from Asterisk.heuristics_generator.learning.gen_learning import GenerativeModel
import numpy as np
from Asterisk import AsteriskSession

def split_features(DU):
    columns_list_full=['index', 'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
           'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'prediction']
    df_features = DU
    df_features = df_features.assign(Features = df_features.fLength.astype(str)+
                                 '@'+df_features.fWidth.astype(str)+
                                 '@'+df_features.fSize.astype(str)+
                                 '@'+df_features.fConc.astype(str)+
                                 '@'+df_features.fConc1.astype(str)+
                                 '@'+df_features.fAsym.astype(str)+
                                 '@'+df_features.fM3Long.astype(str)+
                                 '@'+df_features.fM3Trans.astype(str)+
                                 '@'+df_features.fAlpha.astype(str)+
                                 '@'+df_features.fDist.astype(str))
    
    df_ground = DU['prediction']
    df_features = df_features.drop(columns_list_full, axis = 1)

    return df_features,df_ground

def split_sets(df_features, df_ground):
    # write the subset to a .csv and convert it to a .tsv file
    df_features.to_csv('dataset.csv',header=False)
    #	
    csv.writer(open('dataset.tsv', 'w+'), delimiter='	').writerows(csv.reader(open("dataset.csv")))
    doc_preprocessor = TSVDocPreprocessor('dataset.tsv')
    corpus_parser = CorpusParser(parser=Spacy())
    corpus_parser.apply(doc_preprocessor)


    global cand_extractor
    global T
    global session
    T1, cand_extractor = cand_creation(df_features)
    session = AsteriskSession()
    rows = session.query(T1).count()


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


def prepare_hg(df_test):
    #Preparing gold labels
    columns_list=['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym',
       'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
    index_list=[]
    feature_list=[]

    for c in session.query(T1).filter(T1.Features,T1.split == 1).all():
        index_list.append(c.Features.sentence.document_id-1)

    que = session.query(T1).filter(T1.Features).filter(T1.split==1).all()
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



def cand_creation(df_subset):
    #Generating Candidates
    global T1 
    T1 = candidate_subclass('T1',['Features'])
    r = '^-?\d*.\d*'
    ngrams         = Ngrams(n_max=300)
    regex_matcher = RegexMatchEach(rgx = r)
    cand_extractor = CandidateExtractor(T1, [ngrams], [regex_matcher])
    return T1, cand_extractor


def evaluate_disagreement_factor(L_train,DU):
    index_list = []
    agreements_list=[]
    unlabeled_list=[]
    rows = session.query(T1).filter(T1.split==0)
    for r in rows: 
        index_list.append(r.Features.sentence.document_id-1),
    for rows in L_train:
        if len(rows.data) == 0:
            unlabeled_list.append(True)
        else:
            unlabeled_list.append(False)
        agreements_list.append(abs(sum(rows.data)))
    
    df_with_additional_info = pd.DataFrame(index_list)
    df_with_additional_info.columns=['index']
    df_with_additional_info['disagreement_factor'] = agreements_list
    df_with_additional_info['unlabeled_flag'] = unlabeled_list

    df_with_additional_info=df_with_additional_info.rename(columns={'index': 'index'})
    original_data = DU
    original_data = DU.rename(columns={'Index': 'index'})
    df_with_additional_info = df_with_additional_info.merge(original_data, on=['index'], how='left', indicator= True)
    cond1 = df_with_additional_info['unlabeled_flag'] == True
    cond2 = df_with_additional_info['disagreement_factor'] <= 3
    df_active_learning= df_with_additional_info[cond1 | cond2]
    print("Data with additional info: disagreements and abstain")
    print(df_with_additional_info.shape)
    print("Data for Active learning: Data with additional info after applying conditions")
    print(df_active_learning.shape)
    df_active_learning = df_active_learning.drop(['unlabeled_flag', 'disagreement_factor', '_merge'], axis=1)
	
    return df_with_additional_info,df_active_learning    
    



