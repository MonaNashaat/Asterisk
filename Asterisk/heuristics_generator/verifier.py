import numpy as np
from scipy import sparse

from Asterisk.heuristics_generator.label_aggregator import *
from Asterisk.heuristics_generator.label_aggregator import LabelAggregator
from Asterisk.heuristics_generator import *
def odds_to_prob(l):
  """
  This is the inverse logit function logit^{-1}:
    l       = \log\frac{p}{1-p}
    \exp(l) = \frac{p}{1-p}
    p       = \frac{\exp(l)}{1 + \exp(l)}
  """
  return np.exp(l) / (1.0 + np.exp(l))

class Verifier(object):
    """
    A class for the Asterisk Model Verifier
    """

    def __init__(self, L_train, L_val, val_ground, has_Asterisk=True):
        self.L_train = L_train.astype(int)
        self.L_val = L_val.astype(int)
        self.val_ground = val_ground

        self.has_Asterisk = has_Asterisk
        if self.has_Asterisk:
            from Asterisk.heuristics_generator.learning import GenerativeModel
            from Asterisk.heuristics_generator.learning import RandomSearch
            from Asterisk.heuristics_generator.learning.structure import DependencySelector

    def train_gen_model(self,deps=False,grid_search=False):
        """ 
        Calls appropriate generative model
        """
        if self.has_Asterisk:
            #TODO: GridSearch
            from Asterisk.heuristics_generator.learning import GenerativeModel
            from Asterisk.heuristics_generator.learning import RandomSearch
            from Asterisk.heuristics_generator.learning.structure import DependencySelector
            gen_model = GenerativeModel()
            gen_model.train(self.L_train, epochs=100, decay=0.001 ** (1.0 / 100), step_size=0.005, reg_param=1.0)
        else:
            gen_model = LabelAggregator()
            gen_model.train(self.L_train, rate=1e-3, mu=1e-6, verbose=False)
        self.gen_model = gen_model

    def assign_marginals(self):
        """ 
        Assigns probabilistic labels for train and val sets 
        """ 
        self.train_marginals = self.gen_model.marginals(sparse.csr_matrix(self.L_train))
        self.val_marginals = self.gen_model.marginals(sparse.csr_matrix(self.L_val))
        #print 'Learned Accuracies: ', odds_to_prob(self.gen_model.w)

    def find_vague_points(self,gamma=0.1,b=0.5):
        """ 
        Find val set indices where marginals are within thresh of b 
        """
        val_idx = np.where(np.abs(self.val_marginals-b) <= gamma)
        return val_idx[0]

    def find_incorrect_points(self,b=0.5):
        """ Find val set indices where marginals are incorrect """
        val_labels = 2*(self.val_marginals > b)-1
        val_idx = np.where(val_labels != self.val_ground)
        return val_idx[0]
