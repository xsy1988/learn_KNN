# Create by MrZhang on 2019-11-18

import numpy as np
import operator

def normData(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = (dataSet - np.tile(minVals, (dataSet.shape[0], 1))) / np.tile(ranges, (dataSet[0], 1))
    return normDataSet, ranges, minVals

def computeDistance(testSet, dataSet):
    return np.sqrt(np.sum(np.square(np.tile(testSet, (dataSet.shape[0], 1)) - dataSet), axis=1))

def KNN(testSet, dataSet, k, labels):
    normDataSet, ranges, minVals = normData(dataSet)
    normTestSet = (testSet - minVals) / ranges
    distances = computeDistance(normTestSet, normDataSet)
    sortedDisIndex = distances.argsort()

    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDisIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
