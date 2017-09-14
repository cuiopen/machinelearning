#coding:utf-8
from numpy import *
#加载显示图形
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import operator
import traceback

if __name__ == "__main__":
	fr = open("check.txt")
	arrayData = fr.readlines()
	#获得行数
	nDataRowCount = len(arrayData)
	nindex = 0
	WriteData = []
	for strLineData in arrayData:
		#strLineData = strLineData.replace("\n", ", \n") 
		#WriteData.append(strLineData)
		strData = arrayData[nindex].strip()
		listSingleLine = strData.split(',')	
		if(len(listSingleLine) != 7):
			print "error:" + str(nindex)
		nindex = nindex + 1
		
	#f=file("check.txt","w+")	
	#f.writelines(WriteData)
	#f.close()	
	print "check ok"