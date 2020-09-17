# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 08:39:58 2020

@author: YJ001
"""

from sklearn import datasets
import numpy as np
import operator

iris = datasets.load_iris()
data = iris.data
feature_names = iris.feature_names
target = iris.target
target_names = iris.target_names



def classify0(inX, dataSet, labels, k):
    """Distance calculation"""
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  
    sortedDistIndicies = distances.argsort()
    distances = sqDistances**0.5
    classCount={}
    """Voting with lowest k distances"""
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
"""
The function classify0() takes four inputs: the input vector to classify called inX,
our full matrix of training examples called dataSet, a vector of labels called labels,
and, finally, k, the number of nearest neighbors to use in the voting. The labels vector
should have as many elements in it as there are rows in the dataSet matrix. You calculate
the distances B using the Euclidian distance where the distance between two vectors
"""

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = data, target
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],\
        datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d"\
        % (classifierResult, datingLabels[i]))
    if (classifierResult != datingLabels[i]):
        errorCount += 1.0
    print ("the total error rate is: %f" % (errorCount/float(numTestVecs)))

if __name__ =='__main__':
    datingClassTest()