#-- encoding:UTF-8 --
#-- Author: TNT_000 by Abner yang
import numpy as np 
import pandas as pd 
import datetime
from function import *
from model import *
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

#--- xgboost parameters
params={
	'scale_pos_weight': 0,
	'booster':'gbtree',
	'objective': 'binary:logistic',
	#'objective': 'rank:pairwise',

    'eval_metric': 'map',
	'stratified':True,

	'max_depth':4,
	'min_child_weight':0.01,
	'gamma':0.1,
	'subsample':0.8,
	'colsample_bytree':0.6,
	#'max_delta_step':8,
	#'colsample_bylevel':0.5,
	#'rate_drop':0.3,
	
	'lambda':0.0001,   #550
	#'alpha':10,
	#'lambda_bias':0,
	
	'eta': 0.02,
	'seed':1288,

	'nthread':8,
	
	'silent':1
}

#--- the config of TNT_000(Abner)'s solution in stateGrid competition
config = {
	'xgbParams':params,
	'xgbRounds':2300,
	'stackFolds':5,
	'seed':12,
	'stackPath':'lr-l2',
	'rounds':2300, #---xgb rounds
	'folds':5, #--- cross validation folds 
	'useMatrix':True, #--- True: use matrix feature, False: no use matrix feature
	'matrixStack':False, #--- True: use teammate's stack feature, False: no use..
	'final':True,
	'myStack':False, #--- True: use my stack feature, False: no use
	'des2':False,
	'listMatrix':[2], #--- list: the value means the window of time to get feature
	'uselistMatrix1':[1,2,3], #--- use matrix feature window list
	'uselistMatrix2':[1,2,3,4,5,6,7,14,21,28,35],#--- use description feature window list
	'uselistMatrix3':[1],#--- use Trend feature window list
	'uselistMatrix4':[1],
	'uselistMatrix5':[2,3,4,5,6,7,14,21,28,35],
	'pccList':[27,28,29,30], #--- list: the value means the window of time to get trend feature
	'desList':[7,14,21,28,35],
	'biasList':[0],
	'name':['kwhU','kwhN','kwhE','kwhS'], #--- the column list to make feature
	'useId':False,#--- True: use id feature False: no use
	'base':[[5,20],[4,20],[3,25],[2,30]],#--id wondws to get id feature
	'matrixStackList':['xgb_prob1'], #--use teammate's stack result name
	'myStackList':['xgb-1','xgb-2'], #--my stack result name
	'Description':True, #--- True: use description feature False: no use
	'Trend':False,#--- True: use trend feature False: no use
	'pcc-dis':[29,27,28,30], #--- list: the value means the window of time to get trend feature
	'des-dis':[7,14,21,28,35],
	'filter':True #--- True: feature selection False: no use
	}



if __name__ == '__main__':

	translateData() #--- preprocess raw data

	getUseMatrix(config, 2016) #--- get matrix feature 
	getDescriptionFeature(config, 2016) #--- get description feature
	getTrendFeature(config, 2016) #--- get trend feature
	getDescribeFeature1(config, 2016)
	getFinalFeature(config, 2016)

	trainFeature, testFeature, trainLabel, testIndex = getFeature(config, 2016)  #-- read feature

	res = xgbCVModel(trainFeature, trainLabel, config['rounds'], config['folds'], params)  #-- xgb cross validation

	model, predict = xgbPredictModel(trainFeature, trainLabel, testFeature, params, config['rounds']) #-- xgb online predict

	storeResult(testIndex, predict, model, 'guodian_final_002') #-- store result

	

	











