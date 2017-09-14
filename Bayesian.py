#coding:utf-8
from numpy import *
#������ʾͼ��
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import operator
import traceback

#���ò����־�
def CreateTxtList():
	objInputTxtList = [['I', 'am', 'a', 'little', 'bird'], 
										['I', 'like', 'drink', 'water'], 
										['I','working', 'at', 'company']]
	
	objClassVec = [0, 1, 0]
	return objInputTxtList, objClassVec
	
#������г��ֹ��ĵ��ʵĲ���
def CreateVocabList(objDataSet):
	objVocalSet = set([])
	for doc in objDataSet:
		objVocalSet = objVocalSet | set(doc)
	return list(objVocalSet)
	
#����һ����������
def CreateWord2Vec(objVocalSet, objLine):
	objRetVec = [0] * len(objVocalSet)
	for word in objLine:
		objRetVec[objVocalSet.index(word)] = 1
	return objRetVec
	
#bayesianѵ��������
def TrainBayes(objVocalList, objRetVec):
	nCount = len(objVocalList)
	nWordCount = len(objVocalList[0])
	pAbusive = sum(objRetVec)/float(nWordCount)
	#δ����
	p0Num = ones(nWordCount)
	#�Ѿ�����
	p1Num = ones(nWordCount)
	p0Denom = 2.0
	p1Denom = 2.0
	for i in range(nCount):
		if objRetVec[i] == 1:
			p1Num += objVocalList[i]
			p1Denom += sum(objVocalList[i])
		else:
			p0Num += objVocalList[i]
			p0Denom += sum(objVocalList[i])			
	p1Vect = log(p1Num/p1Denom)
	p0Vect = log(p0Num/p0Denom)
	return p0Vect, p0Vect, pAbusive
	
if __name__ == '__main__':
	objInputTxtList, objClassVec = CreateTxtList()
	objVocalList =  CreateVocabList(objInputTxtList)
	objVecList = []
	for objLine in objInputTxtList:
		#print objLine
		objRetVec = CreateWord2Vec(objVocalList, objLine)
		objVecList.append(objRetVec)
	p0Vect, p0Vect, pAbusive = TrainBayes(objVecList, objClassVec)	
	print p0Vect, p0Vect, pAbusive