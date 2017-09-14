#coding:utf-8
from numpy import *
#������ʾͼ��
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import operator
import traceback
from math import log

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leftNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

#�����������㷨�������Ũ��
def calcShannonEnt(objDataSet):
	nDatasetCount = len(objDataSet)
	#�����ؼ����ֵ�
	objLabels = {}
	for objDataRow in objDataSet:
		objCurrLabel = objDataRow[-1]
		if objCurrLabel not in objLabels.keys():
			objLabels[objCurrLabel] = 0
		objLabels[objCurrLabel] += 1
	
	nShannonEnt = 0.0
	#�����ֵ��е���Ũ,�����صĹ�ʽ
	for key in objLabels:
		#print str(float(objLabels[key])), nDatasetCount
		prop = float(objLabels[key])/nDatasetCount
		nShannonEnt -= prop * log(prop, 2)
	return nShannonEnt

#���ո������ض����ݼ��������ݼ�	
def SplitDataSet(objDataSet, aixs, value):
	#print objDataSet
	objRetDataset = []
	for featVec in objDataSet:
		if featVec[aixs] == value:
			objReducefeatVec = featVec[:aixs]
			objReducefeatVec.extend(featVec[aixs + 1:])
			objRetDataset.append(objReducefeatVec)
	return objRetDataset
	
#��ȡ���ݼ�
def GetDataSet():
	objDataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
	objLabels = ['eat', 'film', 'girlOK']
	return objDataSet, objLabels
	
#�����õ����ݼ�����
def ChooseBastFeatureToSplit(objDataSet):
	numFeature = len(objDataSet[0]) - 1
	fBestShannonEnt = calcShannonEnt(objDataSet)
	#print fBestShannonEnt
	fBestInfoGain = 0.0
	nBeseFeature = -1
	for i in range(numFeature):
		#��ȡһ�е���������
		featList = [example[i] for example in objDataSet]
		#print featList
		#print "========================="
		#��ò��ظ���Ψһ����ֵ����
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
	
#ȡ�����������
def majorityCnt(objClassList):
	objClassCount = {}
	for vote in objClassList:
		if vote not in objClassCount.keys():
			objClassCount[vote] = 0
		objClassCount[vote] += 1
	sortedClassCount = sorted(objClassCount.items(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

#����������	
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
	
#���Ҷ�ӽڵ���
def GetNumleaf(objDecisionTree):
	nLeafs = 0
	strFirstName = objDecisionTree.keys()[0]
	objSecondDict = objDecisionTree[strFirstName]
	for key in objSecondDict.keys():
		if type(objSecondDict[key]).__name__=='dict':
			#������ӽڵ㣬�ݹ����Ҷ�ӽڵ�
			nLeafs += GetNumleaf(objSecondDict[key])
		else:
			nLeafs += 1
	return nLeafs
	
#��������
def GetTreeDeep(objDecisionTree):
	nMaxDeeps = 0
	strFirstName = objDecisionTree.keys()[0]
	objSecondDict = objDecisionTree[strFirstName]
	for key in objSecondDict.keys():
		if type(objSecondDict[key]).__name__=='dict':
			#������ӽڵ㣬�ݹ����Ҷ�ӽڵ�
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
	#����׽ڵ������
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