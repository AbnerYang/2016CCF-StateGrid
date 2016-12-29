# -- encoding:UTF-8 --
#-- Author: TNT_000 by Abner yang
import numpy as np 
import pandas as pd 
import xgboost as xgb 
from sklearn.preprocessing import OneHotEncoder
from function import *
from scipy.sparse import hstack
from matplotlib import pyplot
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
#-- map eval function

# -*- encoding:utf-8 -*- 
import numpy as np 
import pandas as pd 
import xgboost as xgb 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import time


def evalerror(predict, true):
	print average_precision_score(true, predict, average='macro', sample_weight=None)
	



def map_eval(true, predict):
	result = pd.DataFrame({'true':true, 'predict':predict})
	result = result.sort(['predict'], ascending = [0])
	#print result
	score = []
	num = 0
	total = 0
	for line in result['true'].values.T:
		total += 1
		if line == 1:
			num += 1
			score.append(float(num)/total)
	mapScore = np.mean(score)
	print mapScore
	return mapScore

#-- xgboost local train-test Model frame
def xgbLocalModel(trainFeature, testFeature, trainLabel, testLabel, params, rounds):
	params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
	print params['scale_pos_weight']

	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	dtest = xgb.DMatrix(testFeature, label = testLabel)

	watchlist  = [(dtest,'eval'), (dtrain,'train')]
	num_round = rounds
	print 'run local: ' + 'round: ' + str(rounds)
	model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval = 20)#,feval = evalerror)

	predict = model.predict(dtest)

	return predict

#-- xgboost cross-validation Model frame
def xgbCVModel(trainFeature, trainLabel, rounds, folds, params):
	
	#--Set parameter: scale_pos_weight-- 
	params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])
	print params['scale_pos_weight']


	#--Get User-define DMatrix: dtrain--
	#print trainQid[0]
	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	num_round = rounds

	#--Run CrossValidation--
	print 'run cv: ' + 'round: ' + str(rounds) + ' folds: ' + str(folds) 
	res = xgb.cv(params, dtrain, num_round, nfold = folds, verbose_eval = 20)
	return res

#-- xgboost online predict Model frame
def xgbPredictModel(trainFeature, trainLabel, testFeature, params, rounds):

	dtrain = xgb.DMatrix(trainFeature, label = trainLabel)
	dtest = xgb.DMatrix(testFeature, label = np.zeros(testFeature.shape[0]))

	watchlist  = [(dtest,'eval'), (dtrain,'train')]
	
	params['scale_pos_weight'] = (float)(len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])

	print params['scale_pos_weight']

	num_round = rounds
	
	model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval = 100)


	importance = pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort('importance', ascending=False)


	predict = model.predict(dtest)

	importance.to_csv('../importance/im.csv', index = False)
	
	return model, predict












