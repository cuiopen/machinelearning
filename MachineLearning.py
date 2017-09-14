#coding:utf-8
from numpy import *
#加载显示图形
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import operator
import traceback

def Load_File_Data(strFileName):
	try:
		fr = open(strFileName)
		arrayData = fr.readlines()
		#获得行数
		nDataRowCount = len(arrayData)
		#获得列数
		strData = arrayData[0].strip()
		listSingleLine = strData.split('\t')
		nDataColCount = len(listSingleLine)
		print "readLens=" + str(nDataRowCount) + ",listSingleLine=" + str(nDataColCount)
		
		Lablevector = []
		
		returnMat = zeros((nDataRowCount, nDataColCount - 1))
		#填充数据
		nindex = 0
		for strLineData in arrayData:
			strData = arrayData[nindex].strip()
			listSingleLine = strData.split('\t')
			returnMat[nindex, :] = listSingleLine[0 : nDataColCount - 1]
			Lablevector.append(int(listSingleLine[-1]))
			nindex = nindex + 1
			
		return returnMat,Lablevector
		
	except Exception,e:
		print traceback.format_exc()
		return None,None

#获得等效的权值		
def Get_Weight(returnMat):
	minVals = returnMat.min(0)
	maxVals = returnMat.max(0)
	ranges = maxVals - minVals
	#print "minVals=" + str(minVals) + ",maxVals=" + str(maxVals)
	weightDataMat = zeros(shape(returnMat))
	m = returnMat.shape[0]
	weightDataMat = returnMat - tile(minVals, (m, 1))
	weightDataMat = weightDataMat/tile(ranges, (m, 1))
	return weightDataMat
		
def Show_picture(returnMat, Lablevector):
	print returnMat
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(returnMat[:,1], returnMat[:,2], 15.0*array(Lablevector), 15.0*array(Lablevector))
	plt.savefig('plot.png', format='png')
	#plt.show()
	
def K_neighbor(nIndex, DataSet, Labels, k):
	DataSetSize = DataSet.shape[0]
	diffMat  = tile(nIndex, (DataSetSize, 1)) - DataSet
	sqdiffMat = diffMat ** 2
	sqDistance = sqdiffMat.sum(axis=1)
	distances = sqDistance ** 0.5
	sortedDistance = distances.argsort()
	classCount = {}
	for i in range(k):
		Votelabel = Labels[sortedDistance[i]]
		classCount[Votelabel] = classCount.get(Votelabel, 0) + 1
		print Labels[sortedDistance[i]]
	SortedclassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return SortedclassCount[0][0]

#检查分类器	
def Check_Class_Ifier(returnMat, Lablevector):
	hoRatio = 0.20
	m = returnMat.shape[0]
	numTestVes = int(m * hoRatio)
	#print numTestVes
	errorcount= 0.0
	for i in range(numTestVes):
		print "i=" + str(i)
		#获得除去i行的矩阵，保证计算距离的运算量
		NeighborMat = returnMat
		NeighborMat = delete(NeighborMat, i, 0)
		#print NeighborMat
		classifierResult = K_neighbor(returnMat[i,:], NeighborMat, Lablevector[numTestVes:m], 3)
		if(classifierResult != Lablevector[i]):
			errorcount = errorcount + 1.0
	print "errorcount=" + str(errorcount)
		
if __name__ == "__main__":
	returnMat, Lablevector = Load_File_Data("Test.txt")
	if (returnMat is None) or (Lablevector is None):
		print "(Load_File_Data) fail"
	else:
		weightDataMat = Get_Weight(returnMat)
		#Show_picture(weightDataMat, Lablevector)
		Check_Class_Ifier(weightDataMat, Lablevector)