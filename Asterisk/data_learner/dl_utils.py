from sklearn.ensemble import RandomForestRegressor
from Asterisk.data_learner.active_learner import *
from Asterisk.data_learner.models import DDL_Dataset 
from Asterisk.data_learner.lal_model import LALmodel
from Asterisk.data_learner.experiment import Experiment
from Asterisk.data_learner.results import Results
import numpy as np

def run_dll(lalModel,labeling_budget,df_active_learning):
	nExperiments = 1
	nEstimators = 50
	nStart = 2
	nIterations = labeling_budget-2
	quality_metrics = ['accuracy']
	dtst = DDL_Dataset(df_active_learning = df_active_learning)
	dtst.setStartState(nStart)
	alLALiterative = ActiveLearnerLAL(dtst, nEstimators, 'lal-iter', lalModel)
	als = [alLALiterative]
	exp = Experiment(nIterations, nEstimators, quality_metrics, dtst, als, 'here we can put a comment about the current experiments')

	res = Results(exp, nExperiments)

	for i in range(nExperiments):
	    performance = exp.run()
	    res.addPerformance(performance)
	    exp.reset()

	'''print('final results:')
	print('-----------------')
	print(alLALiterative.labeled_indices)
	print(alLALiterative.labeled_labels)'''
	return alLALiterative.labeled_indices, alLALiterative.labeled_labels


