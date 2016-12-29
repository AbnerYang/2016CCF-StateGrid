#-- encoding:UTF-8 --
#-- Author: TNT_000 by Abner yang
import pandas as pd 
import numpy as np 
import datetime
import math

#-- get number of days from 2014/01/01 
def getDate(date, p):
	listTime = []
	print min(date)
	i = 0
	for d in date:
		time = datetime.datetime.strptime(d,"%Y/%m/%d")
		time1 = datetime.datetime.strptime(p,"%Y/%m/%d")
		listTime.append((time-time1).days)
		if i%10000 == 1:
			print i
		i += 1
	return listTime

#-- preprocess data
def translateData():
	train = pd.read_csv('../data/train.csv', header = None)
	train.columns = ['CONS_NO','label']

	train.to_csv('../data/trainInfo.csv', index = False)

	test = pd.read_csv('../data/test.csv', header = None)
	test.columns = ['CONS_NO']

	test.to_csv('../data/testInfo.csv', index = False)
	
	useData	= pd.read_csv('../data/all_user_yongdian_data_2015.csv', header = 0)
	time = getDate(useData['DATA_DATE'].values.T,'2015/01/01')
	
	useData['Time'] = time
	print useData.shape
	useData1 = useData[(useData['Time'] >= 0) & (useData['Time'] < 365)]
	print useData1.shape
	useData1 = useData1.sort(['CONS_NO','Time', 'KWH_READING'], ascending=[1,1,0])
	useData1.to_csv('../data/useDataInfo_2016.csv', index = False)

	useData2 = useData[(useData['Time'] >= -365) & (useData['Time'] < 0)]
	useData2['Time'] = useData2['Time'].values.T+365
	print useData2.shape
	useData2 = useData2.sort(['CONS_NO','Time', 'KWH_READING'], ascending=[1,1,0])
	useData2.to_csv('../data/useDataInfo_2015.csv', index = False)

	useData3 = useData[(useData['Time'] >= -730) & (useData['Time'] < -365)]
	useData3['Time'] = useData3['Time'].values.T+730
	print useData3.shape
	useData3 = useData3.sort(['CONS_NO','Time', 'KWH_READING'], ascending=[1,1,0])
	useData3.to_csv('../data/useDataInfo_2014.csv', index = False)

	useData	= pd.read_csv('../data/user_dianliang_round3.csv', header = 0)
	time = getDate(useData['DATA_DATE'].values.T,'2016/01/01')

	useData['Time'] = time
	useData = useData.sort(['CONS_NO','Time', 'KWH_READING'], ascending=[1,1,0])
	useData.to_csv('../data/useDataInfo_finalTest_2016.csv', index = False)


#-- get matrix feature
def getUseMatrix(config, p):
	useData1 = pd.read_csv('../data/useDataInfo_'+str(p)+'.csv', header = 0)
	useData1 = useData1.fillna(-1)

	useData2 = pd.read_csv('../data/useDataInfo_finalTest_2016.csv', header = 0)
	useData2 = useData2.fillna(-1)

	useData = pd.concat([useData1, useData2], axis = 0)
	print useData1.shape, useData2.shape, useData.shape
	data = useData[['CONS_NO','Time','KWH','KWH_READING','KWH_READING1']].values
	print data.shape

	userNum = len(np.unique(useData['CONS_NO'].values))
	timeT = max(useData['Time'].values.T)+1
	print min(useData['Time'].values.T), timeT
	for l in config['listMatrix']:
		print l
		timeNum = int(math.ceil(float(timeT)/l)) 
		print timeNum
		matrix1 = np.zeros([userNum, timeNum]) - 1
		matrix2 = np.zeros([userNum, timeNum]) 
		matrix3 = np.zeros([userNum, timeNum]) - 1
		matrix4 = np.zeros([userNum, timeNum]) - 1
		uidIndex = []

		userDict = {}
		num = 0
		i = 0
		for line in data:
			if i%100000 == 1:
				print i
			i += 1
			if userDict.has_key(line[0]) == False:
				userDict[line[0]] = num
				uidIndex.append(line[0])
				num += 1

			col = line[1]/l
			
			if matrix1[userDict[line[0]], col] == -1:
				matrix1[userDict[line[0]], col] = line[2]
			else:
				matrix1[userDict[line[0]], col] += line[2]

			if matrix3[userDict[line[0]], col] == -1:
				matrix3[userDict[line[0]], col] = line[3]
			else:
				matrix3[userDict[line[0]], col] += line[3]

			if matrix4[userDict[line[0]], col] == -1:
				matrix4[userDict[line[0]], col] = line[4]
			else:
				matrix4[userDict[line[0]], col] += line[4]


			matrix2[userDict[line[0]], col] += 1

			

		matrixColName1 = getColName(timeNum, 'useDay'+str(l)+'-')
		matrixColName2 = getColName(timeNum, 'useDayNum'+str(l)+'-')
		matrixColName3 = getColName(timeNum, 'endNum'+str(l)+'-')
		matrixColName4 = getColName(timeNum, 'startNum'+str(l)+'-')
		

		matrixFeature1 = pd.DataFrame(matrix1, columns = matrixColName1)
		matrixFeature2 = pd.DataFrame(matrix2, columns = matrixColName2)
		matrixFeature3 = pd.DataFrame(matrix3, columns = matrixColName3)
		matrixFeature4 = pd.DataFrame(matrix4, columns = matrixColName4)

		matrixFeature1['CONS_NO'] = uidIndex
		matrixFeature2['CONS_NO'] = uidIndex
		matrixFeature3['CONS_NO'] = uidIndex
		matrixFeature4['CONS_NO'] = uidIndex


		#matrixFeature = pd.concat([matrixFeature1, matrixFeature2, matrixFeature3, matrixFeature4], axis = 1)
		
		matrixFeature1.to_csv('../feature/matrixFeature'+str(p)+'/kwhU_matrixFeature'+str(l)+'.csv', index = False)
		matrixFeature2.to_csv('../feature/matrixFeature'+str(p)+'/kwhN_matrixFeature'+str(l)+'.csv', index = False)
		matrixFeature3.to_csv('../feature/matrixFeature'+str(p)+'/kwhS_matrixFeature'+str(l)+'.csv', index = False)
		matrixFeature4.to_csv('../feature/matrixFeature'+str(p)+'/kwhE_matrixFeature'+str(l)+'.csv', index = False)
	
#-- numpy array to pandas DataFrame add columns's name list
def getColName(colNum, stri):
	print colNum, stri
	colName = []
	for i in range(colNum):
		colName.append(stri + str(i))
	return colName

#-- get description feature
def getDescriptionFeature(config, p):
	for l in config['listMatrix']:
		for n in config['name']:
			print n
			useMatrix = pd.read_csv('../feature/matrixFeature'+str(p)+'/'+n+'_matrixFeature'+str(l)+'.csv', header = 0)
			print useMatrix.shape
			uid = useMatrix['CONS_NO'].values.T
			feature = useMatrix.drop(['CONS_NO'], axis = 1)

			featureMatrix = np.zeros([len(uid), 8])

			feature = feature.values

			num = 0
			naNum = []
			outNum1 = []
			outNum2 = []
			outNum3 = []
			ii = 0
			for line in feature:
				if ii%1000 == 1:
					print ii
				ii+=1
				k = len(line)
				line = line[line != -1]
				if len(line) > 0:
					outNum3.append(len(line[line >= np.mean(line)+3*np.std(line)]))
					outNum2.append(len(line[line >= np.mean(line)+2*np.std(line)]))
					outNum1.append(len(line[line >= np.mean(line)+1*np.std(line)]))
				else:
					outNum3.append(-1)
					outNum2.append(-1)
					outNum1.append(-1)

				
				naNum.append(k-len(line)-21)
				lFrame = pd.DataFrame({'Sta':line})
				des = lFrame.describe()
				info = des.values.reshape(des.shape[0])
				featureMatrix[num,:] = info
				num += 1

			matrixColName = getColName(8, 'Description-'+n+str(l))	
			featureMatrix = pd.DataFrame(featureMatrix, columns = matrixColName)
			
			naName = 'naNum'+n+str(l) 
			outName1 = 'outNum1-'+n+str(l) 
			outName2 = 'outNum2-'+n+str(l) 
			outName3 = 'outNum3-'+n+str(l) 
			
			featureMatrix[naName] = naNum

			featureMatrix[outName1] = outNum1
			featureMatrix[outName2] = outNum2
			featureMatrix[outName3] = outNum3

			featureMatrix['CONS_NO'] = uid 

			featureMatrix.to_csv('../feature/describeFeature'+str(p)+'/Description_'+n+str(l)+'.csv', index = False)

def getFinalFeature(config, p):
	for l in config['listMatrix']:
		for n in config['name']:
			useMatrix = pd.read_csv('../feature/matrixFeature'+str(p)+'/'+n+'_matrixFeature'+str(l)+'.csv', header = 0)
			uid = useMatrix['CONS_NO'].values.T
			feature = useMatrix.drop(['CONS_NO'], axis = 1)
			feature = feature.values

			colNum = feature.shape[1] - 1
			featureMatrix = np.zeros([len(uid), colNum-1])
			for row in range(feature.shape[0]):
				for i in range(colNum-1):
					featureMatrix[row,i] = float(feature[row, i+1])/feature[row, i]
				if row%1000 == 1:
					print row
			matrixColName = getColName(colNum-1, 'Trend-final-'+n+str(l))	
			featureMatrix = pd.DataFrame(featureMatrix, columns = matrixColName)
			
			featureMatrix['CONS_NO'] = uid 

			featureMatrix.to_csv('../feature/finalFeature'+str(p)+'/Trend_'+n+str(l)+'final'+'.csv', index = False)


#-- get trend feature
def getTrendFeature(config, p):
	for l in config['listMatrix']:
		for n in config['name']:
			for bias in config['biasList']:
				for pcc in config['pcc-dis']:
					print n
					useMatrix = pd.read_csv('../feature/matrixFeature'+str(p)+'/'+n+'_matrixFeature'+str(l)+'.csv', header = 0)
					print useMatrix.shape
					uid = useMatrix['CONS_NO'].values.T
					feature = useMatrix.drop(['CONS_NO'], axis = 1)
					feature = feature.values

					colNum = feature.shape[1]/pcc
					featureMatrix = np.zeros([len(uid), colNum-1])
					for row in range(feature.shape[0]):
						for i in range(colNum-1):
							featureMatrix[row,i] = np.corrcoef(feature[row,(i*pcc+bias):((i+1)*pcc+bias)], feature[row,((i+1)*pcc+bias):((i+2)*pcc+bias)])[0,1]
						if row%1000 == 1:
							print row
					matrixColName = getColName(colNum-1, 'Trend-PCC-'+n+str(l)+'pcc'+str(pcc))	
					featureMatrix = pd.DataFrame(featureMatrix, columns = matrixColName)
					
					featureMatrix['CONS_NO'] = uid 

					featureMatrix.to_csv('../feature/trendFeature'+str(p)+'/Trend_'+n+str(l)+'pcc'+str(pcc)+'-bias-'+str(bias)+'.csv', index = False)

def getDescribeFeature1(config, p):
	for l in config['listMatrix']:
		for n in config['name']:
			for bias in config['biasList']:
				for pcc in config['des-dis']:
					print n
					useMatrix = pd.read_csv('../feature/matrixFeature'+str(p)+'/'+n+'_matrixFeature'+str(l)+'.csv', header = 0)
					print useMatrix.shape
					uid = useMatrix['CONS_NO'].values.T
					feature = useMatrix.drop(['CONS_NO'], axis = 1)
					feature = feature.values

					colNum = (feature.shape[1]-bias)/pcc
					featureMatrix = np.zeros([len(uid), colNum*5])
					for row in range(feature.shape[0]):
						for i in range(colNum):
							kk = feature[row,(i*pcc+bias):((i+1)*pcc+bias)]
							ss = [np.mean(kk),np.std(kk),np.median(kk),np.max(kk), np.min(kk)]
							featureMatrix[row,(i*5):(i+1)*5] = ss
						if row%1000 == 1:
							print row
					matrixColName = getColName(colNum*5, 'Des2_'+n+str(l)+'pcc'+str(pcc))	
					featureMatrix = pd.DataFrame(featureMatrix, columns = matrixColName)
					
					featureMatrix['CONS_NO'] = uid 

					featureMatrix.to_csv('../feature/des2Feature'+str(p)+'/Des2_'+n+str(l)+'static'+str(pcc)+'-bias-'+str(bias)+'.csv', index = False)

#-- feature selection
def filter(data):
	col = data.columns
	delName = []
	for i in col:
		value = data[i].values.T
		if len(np.unique(value)) == 1:
			delName.append(i)
	return delName
	#data = data.drop(delName, axis = 1)

	#return data

#-- get upper id from raw		
def getupper(data):
	k = []
	for d in data:
		k.append(d.upper())
	return k

#-- read feature and return
def getFeature(config, p):
	train = pd.read_csv('../data/trainInfo.csv', header = 0)
	test = pd.read_csv('../data/finalTest.csv', header = 0)

	print train.shape, test.shape

	if config['useMatrix'] == True:
		for l in config['uselistMatrix1']:
			for n in config['name']:
				name = '../feature/matrixFeature'+str(p)+'/'+n+'_matrixFeature'+str(l)+'.csv'
				useMatrix = pd.read_csv(name, header = 0)
				train = pd.merge(train, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
				test = pd.merge(test, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
				print train.shape, test.shape
	if config['Description'] == True:
		for l in config['uselistMatrix2']:
			for n in config['name']:
				name = '../feature/describeFeature'+str(p)+'/Description_'+n+str(l)+'.csv'
				useMatrix = pd.read_csv(name, header = 0)
				train = pd.merge(train, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
				test = pd.merge(test, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
				print train.shape, test.shape
	if config['final'] == True:
		for l in config['uselistMatrix5']:
			for n in config['name']:
				name = '../feature/finalFeature'+str(p)+'/Trend_'+n+str(l)+'final'+'.csv'
				useMatrix = pd.read_csv(name, header = 0)
				train = pd.merge(train, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
				test = pd.merge(test, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
				print train.shape, test.shape
	if config['Trend'] == True:
		for l in config['uselistMatrix3']:
			for n in config['name']:
				for b in config['biasList']:
					for pcc in config['pccList']:
						name = '../feature/trendFeature'+str(p)+'/Trend_'+n+str(l)+'pcc'+str(pcc)+'-bias-'+str(b)+'.csv'
						useMatrix = pd.read_csv(name, header = 0)
						train = pd.merge(train, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
						test = pd.merge(test, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
						print train.shape, test.shape
	if config['des2'] == True:
		for l in config['uselistMatrix4']:
			for n in config['name']:
				for b in config['biasList']:
					for pcc in config['desList']:
						name = '../feature/des2Feature'+str(p)+'/Des2_'+n+str(l)+'static'+str(pcc)+'-bias-'+str(b)+'.csv'
						useMatrix = pd.read_csv(name, header = 0)
						train = pd.merge(train, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
						test = pd.merge(test, useMatrix, on = 'CONS_NO', how = 'left').fillna(-1)
						print train.shape, test.shape
	if config['myStack'] == True:
		for l in config['myStackList']:
			data1 = pd.read_csv('../feature/stack/'+l+'_train.csv', header = 0)
			data2 = pd.read_csv('../feature/stack/'+l+'_test.csv', header = 0)
			
			train = pd.concat([train, data1], axis = 1).fillna(-1)
			test = pd.concat([test, data2], axis = 1).fillna(-1)
			print train.shape, test.shape

	if config['matrixStack'] == True:
		for l in config['matrixStackList']:
			data = pd.read_csv('../feature/stackFeature/'+l+'.csv',header = 0)
			data['CONS_NO'] = np.append(train['CONS_NO'].values.T, test['CONS_NO'].values.T)
			train = pd.merge(train, data, on = 'CONS_NO', how = 'left').fillna(-1)
			test = pd.merge(test, data, on = 'CONS_NO', how = 'left').fillna(-1)
			print train.shape, test.shape



	trainUid = train['CONS_NO'].values.T
	testUid = test['CONS_Index'].values.T
	
	trainFeature = train.drop(['CONS_NO','label'], axis = 1)
	testFeature = test.drop(['CONS_NO','CONS_Index'], axis = 1)

	trainLabel = train['label'].values.T


	print trainFeature.shape, testFeature.shape, trainLabel.shape

	# print trainFeature
	# print testFeature

	if config['filter'] == True:
		print 'filter...'
		delName = filter(trainFeature)
		trainFeature = trainFeature.drop(delName, axis = 1)
		testFeature = testFeature.drop(delName, axis = 1)


	print trainFeature.shape, testFeature.shape, trainLabel.shape

	return trainFeature.fillna(-1).values, testFeature.fillna(-1).values, trainLabel, testUid

#-- store the online result
def storeResult(testIndex, predict, model, day):
	result = pd.DataFrame({'CONS_NO':testIndex, 'label':predict})
	#print result
	rpath = '../result/'+ day + '.csv'
	rpath1 = '../result/'+ day + '_prob.csv'
	
	mpath = '../model/'+ day + '.m'
	result = result.sort('label', ascending = False)
	#print result
	result.to_csv(rpath1, index = False)
	result = result['CONS_NO']
	result.to_csv(rpath, index = False, header = False)
	if model != False:
		model.save_model(mpath)	






	
















