#coding:utf-8
from numpy import *
#加载显示图形
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import operator
import traceback
from math import log

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leftNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#决策树生成算法，获得香浓熵
def calcShannonEnt(objDataSet):
	nDatasetCount = len(objDataSet)
	#制作关键词字典
	objLabels = {}
	for objDataRow in objDataSet:
		objCurrLabel = objDataRow[-1]
		if objCurrLabel not in objLabels.keys():
			objLabels[objCurrLabel] = 0
		objLabels[objCurrLabel] += 1
	
	nShannonEnt = 0.0
	#计算字典中的香浓,根据熵的公式
	for key in objLabels:
		#print str(float(objLabels[key])), nDatasetCount
		prop = float(objLabels[key])/nDatasetCount
		nShannonEnt -= prop * log(prop, 2)
	return nShannonEnt

#按照给定的特定数据集划分数据集	
def SplitDataSet(objDataSet, aixs, value):
	#print objDataSet
	objRetDataset = []
	for featVec in objDataSet:
		if featVec[aixs] == value:
			objReducefeatVec = featVec[:aixs]
			objReducefeatVec.extend(featVec[aixs + 1:])
			objRetDataset.append(objReducefeatVec)
	return objRetDataset
	
#获取数据集
def GetDataSet():
	objDataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
	objLabels = ['eat', 'film', 'girlOK']
	return objDataSet, objLabels
	
#获得最好的数据集划分
def ChooseBastFeatureToSplit(objDataSet):
	numFeature = len(objDataSet[0]) - 1
	fBestShannonEnt = calcShannonEnt(objDataSet)
	#print fBestShannonEnt
	fBestInfoGain = 0.0
	nBeseFeature = -1
	for i in range(numFeature):
		#获取一列的数据样本
		featList = [example[i] for example in objDataSet]
		#print featList
		#print "========================="
		#获得不重复的唯一数据值集合
		unigueVals = set(featList)
		#print unigueVals
		newEntropy = 0.0
		for value  in unigueVals:
			subDataSet = SplitDataSet(objDataSet, i, value)
			#print subDataSet
			prob = len(subDataSet)/float(len(objDataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
			#print newEntropy
		fInfoGain = fBestShannonEnt - newEntropy
		if(fInfoGain > fBestInfoGain):
			fBestInfoGain = fInfoGain
			nBeseFeature = i
	#print nBeseFeature
	return nBeseFeature
	
#取得排序最大结果
def majorityCnt(objClassList):
	objClassCount = {}
	for vote in objClassList:
		if vote not in objClassCount.keys():
			objClassCount[vote] = 0
		objClassCount[vote] += 1
	sortedClassCount = sorted(objClassCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

#创建决策树	
def CreateTree(objDataSet, objLabels):
	objClassList = [example[-1] for example in objDataSet]
	if objClassList.count(objClassList[0]) == len(objClassList):
		return objClassList[0]
	if len(objDataSet[0]) == 1:
		return majorityCnt(objClassList)
		
	nBestFeat = ChooseBastFeatureToSplit(objDataSet)
	#print nBestFeat
	strBestLabls = objLabels[nBestFeat]
	DecisionTree = {strBestLabls:{}}
	del objLabels[nBestFeat]
	factValues = [example[nBestFeat] for example in objDataSet]
	unigueVals = set(factValues)
	for value in unigueVals:
		subLables = objLabels[:]
		DecisionTree[strBestLabls][value] = CreateTree(SplitDataSet(objDataSet, nBestFeat, value), subLables)
	return DecisionTree
	
#获得叶子节点数
def GetNumleaf(objDecisionTree):
	nLeafs = 0
	strFirstName = objDecisionTree.keys()[0]
	objSecondDict = objDecisionTree[strFirstName]
	for key in objSecondDict.keys():
		if type(objSecondDict[key]).__name__=='dict':
			#如果有子节点，递归调用叶子节点
			nLeafs += GetNumleaf(objSecondDict[key])
		else:
			nLeafs += 1
	return nLeafs
	
#获得树深度
def GetTreeDeep(objDecisionTree):
	nMaxDeeps = 0
	strFirstName = objDecisionTree.keys()[0]
	objSecondDict = objDecisionTree[strFirstName]
	for key in objSecondDict.keys():
		if type(objSecondDict[key]).__name__=='dict':
			#如果有子节点，递归调用叶子节点
			nThisDeeps += 1 + GetTreeDeep(objSecondDict[key])
		else:
			nThisDeeps = 1			
	if nThisDeeps > nMaxDeeps:
		nMaxDeeps = nThisDeeps
	return nMaxDeeps
	
def plotMidText(objCntrpt, objParentpt, strNodeTxt):
	XMid = (objParentpt[0] - objCntrpt[0])/2.0 + objCntrpt[0]
	YMid = (objParentpt[1] - objCntrpt[1])/2.0 + objCntrpt[1]
	CreatePlot.ax1.text(XMid, YMid, strNodeTxt)
	
def plotNode(strNodeText, objCntrpt, objParentpt, nNodeType):
	CreatePlot.ax1.annotate(strNodeText, xy=objParentpt, xycoords='axes fraction', \
													xytext = objCntrpt, textcoords='axes fraction', va="center", \
													bbox=nNodeType, arrowprops=arrow_args)
	
def plotTree(DecisionTree, objParentpt, strNodeTxt):
	nLeafs = GetNumleaf(DecisionTree)
	nDeeps = GetTreeDeep(DecisionTree)
	#获得首节点的名字
	strFirstName = DecisionTree.keys()[0]
	objCntrpt = (plotTree.xOff + (1.0 + float(nLeafs))/2.0/plotTree.totalW, plotTree.yOff)
	plotMidText(objCntrpt, objParentpt, strNodeTxt)
	plotNode(strFirstName, objCntrpt, objParentpt, decisionNode)
	objSecondDict = DecisionTree[strFirstName]
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	for key in objSecondDict.keys():
		if type(objSecondDict[key]).__name__=='dict':
			plotTree(objSecondDict[key], objCntrpt, str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(objSecondDict[key], (plotTree.xOff, plotTree.yOff), objCntrpt, leftNode)
			plotMidText((plotTree.xOff, plotTree.yOff), objCntrpt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
	
def CreatePlot(DecisionTree):
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	CreatePlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalW = float(GetNumleaf(DecisionTree))
	plotTree.totalD = float(GetTreeDeep(DecisionTree))
	plotTree.xOff = -0.5/plotTree.totalW;
	plotTree.yOff = 1.0
	plotTree(DecisionTree, (0.5, 1.0), '')
	plt.savefig('DecisionTree.png', format='png')
	
if __name__ == "__main__":
	objDataSet, objLabels = GetDataSet()
	#nBeseFeature = ChooseBastFeatureToSplit(objDataSet)
	DecisionTree = CreateTree(objDataSet, objLabels)
	#print DecisionTree
	#print GetTreeDeep(DecisionTree)
	CreatePlot(DecisionTree)