from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import pandas as pd
from Asterisk.heuristics_generator.models import StableLabel
from Asterisk.heuristics_generator.db_helpers import reload_annotator_labels

FPATH = 'gold_labels.tsv'

def number_of_people(sentence):
    active_sequence = False
    count = 0
    for tag in sentence.ner_tags:
        if tag == 'PERSON' and not active_sequence:
            active_sequence = True
            count += 1
        elif tag != 'PERSON' and active_sequence:
            active_sequence = False
    return count


def load_external_labels(session, candidate_class, annotator_name='gold'):
    gold_labels = pd.read_csv(FPATH, sep="\t")
    counter = 0
    #print(session.query(StableLabel).filter(StableLabel.context_stable_ids.label.im_self)[:20])
    for index, row in gold_labels.iterrows():    
    
        # We check if the label already exists, in case this cell was already executed
        context_stable_ids = row['Features']
        #print(row['Features'])
        #print(row['label'])
        '''
        session.add(StableLabel(
                idx=index,
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))
        
        '''
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))
          
        '''
        # Because it's a symmetric relation, load both directions... (it is for persons only)
        context_stable_ids = row['Features']
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))
        '''

    # Commit session
    session.commit()

    # Reload annotator labels
    reload_annotator_labels(session, candidate_class, annotator_name, split=1, filter_label_split=False)
    reload_annotator_labels(session, candidate_class, annotator_name, split=2, filter_label_split=False)
    #reload_annotator_labels(session, candidate_class, annotator_name, split=0, filter_label_split=False)
