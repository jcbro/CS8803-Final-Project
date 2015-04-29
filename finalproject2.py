import numpy
#import cv2
import json
from math import *
#import matplotlib.pyplot as pyplot
import scipy.cluster
import time
#import pandas
import sys
import re

def calcStates(dataIn,movAvg):
    # dataOut[row][0]  = # x position
    # dataOut[row][1]  = # y position
    # dataOut[row][2]  = # heading
    # dataOut[row][3]  = # velocity
    # dataOut[row][4]  = # turn rate
    # dataOut[row][5]  = # cumulative velocity average
    # dataOut[row][6]  = # cumulative turn rate average
    # dataOut[row][7]  = # movAvg moving average of velocity
    # dataOut[row][8]  = # movAvg moving average of turn rate
    # dataOut[row][9]  = # cumulative velocity std dev
    # dataOut[row][10] = # cumulative turn rate std dev

    # Calculate states.
    dataOut = numpy.zeros((dataIn.shape[0],9))
    dataOut[0:,0] = dataIn[0:,0]
    dataOut[0:,1] = dataIn[0:,1]
    dataOut[1:,2] = numpy.arctan2(dataOut[1:,1] - dataOut[0:-1,1],dataOut[1:,0] - dataOut[0:-1,0])
    dataOut[1:,3] = numpy.sqrt((dataOut[1:,0] - dataOut[1 - 1:-1,0])**2 + (dataOut[1:,1] - dataOut[1 - 1:-1,1])**2)
    dataOut[2:,4] = dataOut[2:,2] - dataOut[1:-1,2]

    # Eliminate turn rates of greater than pi/2 by setting to 0. Eliminates weighting wall bounces in other calculations.
    dataOut[dataOut[:,4] > pi/2,4] = 0
    dataOut[dataOut[:,4] < -pi/2,4] = 0

    # More calculate states.
    dataOut[1:,5] = numpy.cumsum(dataOut[1:,3])/numpy.linspace(1,dataOut.shape[0] - 1,dataOut.shape[0] - 1)
    dataOut[2:,6] = numpy.cumsum(dataOut[2:,4])/numpy.linspace(1,dataOut.shape[0] - 2,dataOut.shape[0] - 2)
    dataOut[0 + movAvg:,7] = (numpy.cumsum(dataOut[0:,3])[movAvg:] - numpy.cumsum(dataOut[0:,3])[:-movAvg])/(movAvg*1.0)
    dataOut[1 + movAvg:,8] = (numpy.cumsum(dataOut[1:,4])[movAvg:] - numpy.cumsum(dataOut[1:,4])[:-movAvg])/(movAvg*1.0)

    return dataOut

def kMeansCluster(dataInStates,kmeansCentroids = 10,kmeansIter = 50):
    # Whiten the state values for k means clustering.
    dataInStatesWhiten = scipy.cluster.vq.whiten(dataInStates)

    # Cluster the states via k means. Remove X and Y for clustering, start at 2.
    dataInClusters,_ = scipy.cluster.vq.kmeans(dataInStatesWhiten[:,2:],kmeansCentroids,kmeansIter)

    # Label the data set based upon the clusters. Remove X and Y for labeling, start at 2.
    dataInLabels,_ = scipy.cluster.vq.vq(dataInStatesWhiten[:,2:],dataInClusters)

    # Loop through each unique label and store the means and stds for that cluster.
    dataInLabelsUnique = numpy.unique(dataInLabels)

    # Initialize the means and stds arrays to return.
    dataInStatesClusterMeans = numpy.zeros((dataInLabelsUnique.shape[0],dataInStates.shape[1]))
    dataInStatesClusterStds = numpy.zeros((dataInLabelsUnique.shape[0],dataInStates.shape[1]))
    for dataInLabel in dataInLabelsUnique:
        dataInStatesCluster = dataInStates[dataInLabels == dataInLabel,:]
        dataInStatesClusterMeans[dataInLabel,:] = numpy.mean(dataInStatesCluster,axis=0)
        dataInStatesClusterStds[dataInLabel,:] = numpy.std(dataInStatesCluster,axis=0)

    return dataInClusters,dataInLabelsUnique,dataInStatesClusterMeans,dataInStatesClusterStds

def kMeansLabel(dataClusters,dataLabelsUnique,dataStatesClusterMeans,dataStatesClusterStds,dataIn):
    # Whiten the state values for k means labeling.
    dataInWhiten = scipy.cluster.vq.whiten(dataIn)

    # Label the data set based upon the clusters. Remove X and Y for labeling, start at 2.
    dataInLabels,_ = scipy.cluster.vq.vq(dataInWhiten[:,2:],dataClusters)

    # Initialize the means and stds arrays to return.
    dataInMeans = numpy.zeros((dataInLabels.max() + 1,dataIn.shape[1]))
    dataInStds = numpy.zeros((dataInLabels.max() + 1,dataIn.shape[1]))

    # Loop through each unique label and store the means and stds for that cluster.
    dataInLabelsUnique = numpy.unique(dataInLabels)
    for dataInLabel in dataInLabelsUnique:
        dataInMeans[dataInLabel,:] = dataStatesClusterMeans[dataLabelsUnique == dataInLabel,:]
        dataInStds[dataInLabel,:] = dataStatesClusterStds[dataLabelsUnique == dataInLabel,:]

    # # Loop through each unique label and calculate the mean and std for that cluster.
    # dataInLabelsUnique = numpy.unique(dataInLabels)
    # for dataInLabel in dataInLabelsUnique:
    #     # Find indexes that match the label.
    #     dataIndexes = dataLabels == dataInLabel
    #     dataInIndexes = dataInLabels == dataInLabel
    #
    #     # Store the means.
    #     dataTemp = dataClustersMeans[dataIndexes]
    #     dataTemp = numpy.reshape(dataTemp,(1,dataTemp.shape[0]))
    #     dataInStatesMeans[dataInIndexes,:] = dataTemp
    #
    #     # Store the stds.
    #     dataTemp = dataClustersStds[dataIndexes]
    #     dataTemp = numpy.reshape(dataTemp,(1,dataTemp.shape[0]))
    #     dataInStatesStds[dataInIndexes,:] = dataTemp

    return dataInLabels,dataInMeans,dataInStds

def kalmanRandInitStates(dataInStates,samplesTotal):
    dataRandStates = numpy.zeros((samplesTotal,dataInStates.shape[1]))
    for stateCnt in range(0,dataInStates.shape[1]):
        dataRandStates[:,stateCnt:stateCnt + 1] = numpy.random.uniform(min(dataInStates[:,stateCnt]),max(dataInStates[:,stateCnt]),(samplesTotal,1))
    return dataRandStates

def kalmanSense(dataInStates,dataSenseState):
    sigma = 20
    diff = dataInStates - numpy.tile(dataSenseState,(dataInStates.shape[0],1))
    #diffPercent = diff/numpy.tile(dataSenseState,(dataInStates.shape[0],1))
    diffSq = diff**2
    MSE = numpy.sum(diffSq,axis=1)/dataSenseState.shape[1]
    MSE = numpy.reshape(MSE,(dataInStates.shape[0],1))
    dataOutProb = numpy.exp(-1*MSE/sigma**2)
    dataOutProb = numpy.reshape(dataOutProb,(dataInStates.shape[0],1))
    return dataOutProb

def kmeansMove(dataRandStates,dataRandLabels,dataRandCurrentMeans,dataRandCurrentStds,dataXMin,dataXMax,dataYMin,dataYMax):
    # Initialize the means and stds array sizes.
    dataRandStatesMeans = numpy.zeros(dataRandStates.shape)
    dataRandStatesStds = numpy.zeros(dataRandStates.shape)

    # Loop through each unique label and build the means and stds for that cluster.
    for randLabel in dataRandLabels:
        randIndexes = dataRandLabels == randLabel
        dataRandStatesMeans[randIndexes,:] = dataRandCurrentMeans[randLabel,:]
        dataRandStatesStds[randIndexes,:] = dataRandCurrentStds[randLabel,:]

    # Assign each state value based upon a random normal generated with those means and stds.
    dataRandStates[:,8:9] = dataRandStatesStds[:,8:9]*numpy.random.randn(dataRandStatesMeans.shape[0],1) + dataRandStatesMeans[:,8:9]
    dataRandStates[:,7:8] = dataRandStatesStds[:,7:8]*numpy.random.randn(dataRandStatesMeans.shape[0],1) + dataRandStatesMeans[:,7:8]
    dataRandStates[:,6:7] = dataRandStatesStds[:,6:7]*numpy.random.randn(dataRandStatesMeans.shape[0],1) + dataRandStatesMeans[:,6:7]
    dataRandStates[:,5:6] = dataRandStatesStds[:,5:6]*numpy.random.randn(dataRandStatesMeans.shape[0],1) + dataRandStatesMeans[:,5:6]
    dataRandStates[:,4:5] = dataRandStatesStds[:,4:5]*numpy.random.randn(dataRandStatesMeans.shape[0],1) + dataRandStatesMeans[:,4:5]
    dataRandStates[:,3:4] = dataRandStatesStds[:,3:4]*numpy.random.randn(dataRandStatesMeans.shape[0],1) + dataRandStatesMeans[:,3:4]

    # Increment the heading based upon the state turn rate value.
    dataRandStates[:,2:3] = dataRandStates[:,2:3] + dataRandStates[:,8:9]

    # Keep the angle within [-pi,+pi]
    dataRandStates[:,2:3] = dataRandStates[:,2:3] - 2*pi*(dataRandStates[:,2:3] > pi)
    dataRandStates[:,2:3] = dataRandStates[:,2:3] + 2*pi*(dataRandStates[:,2:3] < -pi)

    # Move the state based upon the state velocity value and heading value.
    dataRandStates[:,1:2] = dataRandStates[:,1:2] + dataRandStates[:,7:8]*numpy.sin(dataRandStates[:,2:3])
    dataRandStates[:,0:1] = dataRandStates[:,0:1] + dataRandStates[:,7:8]*numpy.cos(dataRandStates[:,2:3])

    # Wall bounce symmetry for X limits.
    indexes = dataRandStates[:,0:1] <= dataXMin
    dataRandStates[indexes[:,0],2:3] = pi - dataRandStates[indexes[:,0],2:3]
    indexes = dataRandStates[:,0:1] > dataXMax
    dataRandStates[indexes[:,0],2:3] = -pi - dataRandStates[indexes[:,0],2:3]

    # Wall bounce symmetry for Y limits.
    indexes = dataRandStates[:,1:2] <= dataYMin
    dataRandStates[indexes[:,0],2:3] = -1*dataRandStates[indexes[:,0],2:3]
    indexes = dataRandStates[:,1:2] > dataYMax
    dataRandStates[indexes[:,0],2:3] = -1*dataRandStates[indexes[:,0],2:3]

    # Keep the angle within [-pi,+pi]
    dataRandStates[:,2:3] = dataRandStates[:,2:3] - 2*pi*(dataRandStates[:,2:3] > pi)
    dataRandStates[:,2:3] = dataRandStates[:,2:3] + 2*pi*(dataRandStates[:,2:3] < -pi)

    # # X
    # if dataOutStates[frameCurrent,0] <= xMin or dataOutStates[frameCurrent,0] >= xMax:
    #     if dataOutStates[frameCurrent,2] >= 0:
    #         dataOutStates[frameCurrent,2] =  pi - dataOutStates[frameCurrent,2]
    #     if dataOutStates[frameCurrent,2] < 0:
    #         dataOutStates[frameCurrent,2] = -pi - dataOutStates[frameCurrent,2]
    #
    # # Y
    # if dataOutStates[frameCurrent,1] <= yMin or dataOutStates[frameCurrent,1] >= yMax:
    #     dataOutStates[frameCurrent,2] = -1*dataOutStates[frameCurrent,2]

    return dataRandStates

def kalmanResample(states,prob):
    # Normalize the probabilities.
    probNorm = prob/numpy.sum(prob)

    # Randomly sample states states with replacement given probabilities prob.
    dataTemp1 = numpy.arange(states.shape[0])
    dataTemp1 = numpy.reshape(dataTemp1,(prob.shape[0],1))
    indexes = numpy.random.choice(dataTemp1.flatten(),size=dataTemp1.shape[0],replace=True,p=probNorm.flatten())
    returnStates = states[indexes,:]

    return returnStates











# Main prediction code.
def estimate(data):

    dataIn = numpy.array(data)
    #dataIn = numpy.array(json.loads(open('hexbug-training_video-centroid_data').read()))

    #pyplot.ion()

    # Calculate all of the state variables.
    movAvg = 10
    dataInStates = calcStates(dataIn,movAvg)
    dataXMin = numpy.amin(dataInStates[:,0])
    dataXMax = numpy.amax(dataInStates[:,0])
    dataYMin = numpy.amin(dataInStates[:,1])
    dataYMax = numpy.amax(dataInStates[:,1])

    # Get latest data point from the In set.
    dataInCurrent = dataInStates[-1,:]
    dataInCurrent = numpy.reshape(dataInCurrent,(1,dataInCurrent.shape[0]))

    # Initialize a uniform random set over all the data input state space.
    samplesTotal = 300
    dataRandStates = kalmanRandInitStates(dataInStates,samplesTotal)

    # Calculate kalman sense probability.
    dataInCurrentProb = kalmanSense(dataInStates,dataInCurrent)

    # Randomly down sample the data set for clustering speed.
    # sampleSize = 4000
    # sampleStates = dataInStates[numpy.random.randint(0,dataInStates.shape[0],sampleSize)]
    sampleStates = dataInStates

    # Perform clustering to get labels.
    dataInClusters,dataInLabelsUnique,dataInStatesClusterMeans,dataInStatesClusterStds = kMeansCluster(sampleStates,kmeansCentroids = 10,kmeansIter = 50)

    predictState = numpy.zeros((60,dataInStates.shape[1]))

    for frameCount in range(0,60):
        # Label the data of interest and return the mean and std.
        currentLabels,currentMeans,currentStds = kMeansLabel(dataInClusters,dataInLabelsUnique,dataInStatesClusterMeans,dataInStatesClusterStds,dataInCurrent)
        dataRandLabels,dataRandCurrentMeans,dataRandCurrentStds = kMeansLabel(dataInClusters,dataInLabelsUnique,dataInStatesClusterMeans,dataInStatesClusterStds,dataRandStates)

        # Move the rand data set based upon kmeans classification.
        dataInCurrent = kmeansMove(dataInCurrent,currentLabels,currentMeans,currentStds,dataXMin,dataXMax,dataYMin,dataYMax)
        dataRandStates = kmeansMove(dataRandStates,dataRandLabels,dataRandCurrentMeans,dataRandCurrentStds,dataXMin,dataXMax,dataYMin,dataYMax)

        # Calculate kalman sense probability.
        dataRandProb = kalmanSense(dataRandStates,dataInCurrent)
        dataRandStates = kalmanResample(dataRandStates,dataRandProb)

        predictState[frameCount,:] = numpy.mean(dataRandStates,axis=0)

        #pyplot.figure(1)
        #pyplot.clf()
        #pyplot.show()
        #pyplot.plot(predictState[:frameCount,0:1],predictState[:frameCount,1:2],'-r')
        #pyplot.scatter(dataRandStates[:,0],dataRandStates[:,1],color=numpy.random.rand(3,1))
        #pyplot.axis([dataXMin,dataXMax,dataYMin,dataYMax])
        #pyplot.draw()

    #print "done"
    return predictState[:,0:2].astype(int)







def readFile(filename):
    measurements = []
    f = open(filename)
    data = f.readlines()
    for line in data:
        match = re.search("(-?\d+), (-?\d+)", line)
        measurement = (int(match.group(1)), int(match.group(2)))
        measurements.append(measurement)
    return measurements

def printResults(filename, prediction):
    prediction = [list(i) for i in prediction]
    with open(filename, 'w') as file:
        file.write('[')
        for i in range(len(prediction)-1):
            m = prediction[i]
            file.write(str(m) + ',\n')
        file.write(str(prediction[-1]) + ']')

if __name__ == "__main__":
    total = len(sys.argv)
    if len(sys.argv) != 2:
        print ("Invalid Argument")
        sys.exit(1)

    measurements = readFile(str(sys.argv[1]))
    prediction = estimate(measurements)
    printResults("prediction.txt", prediction)





# Run through entire data set iteratively.
# for dataInCnt in range(1000,dataIn.shape[0]):
#     # Calculate all of the state variables.
#     movAvg = 10
#     dataInStates = calcStates(dataIn,movAvg)
#     dataXMin = numpy.amin(dataInStates[:,0])
#     dataXMax = numpy.amax(dataInStates[:,0])
#     dataYMin = numpy.amin(dataInStates[:,1])
#     dataYMax = numpy.amax(dataInStates[:,1])
#
#     # Get latest data point from the In set.
#     dataInCurrent = dataInStates[dataInCnt]
#     dataInCurrent = numpy.reshape(dataInCurrent,(1,dataInCurrent.shape[0]))
#
#     # Initialize a uniform random set over all the data input state space.
#     samplesTotal = 300
#     dataRandStates = kalmanRandInitStates(dataInStates,samplesTotal)
#
#     # Calculate kalman sense probability.
#     dataInCurrentProb = kalmanSense(dataInStates,dataInCurrent)
#
#     # Randomly down sample the data set for clustering speed.
#     # sampleSize = 4000
#     # sampleStates = dataInStates[numpy.random.randint(0,dataInStates.shape[0],sampleSize)]
#     sampleStates = dataInStates
#
#     # Perform clustering to get labels.
#     dataInClusters,dataInLabelsUnique,dataInStatesClusterMeans,dataInStatesClusterStds = kMeansCluster(sampleStates,kmeansCentroids = 10,kmeansIter = 50)
#
#     # Label the data of interest and return the mean and std.
#     dataInCurrentLabels,dataInCurrentMeans,dataInCurrentStds = kMeansLabel(dataInClusters,dataInLabelsUnique,dataInStatesClusterMeans,dataInStatesClusterStds,dataInCurrent)
#
#     for frameCount in range(0,60):
#         # Label the data of interest and return the mean and std.
#         currentLabels,currentMeans,currentStds = kMeansLabel(dataInClusters,dataInLabelsUnique,dataInStatesClusterMeans,dataInStatesClusterStds,dataInCurrent)
#         dataRandLabels,dataRandCurrentMeans,dataRandCurrentStds = kMeansLabel(dataInClusters,dataInLabelsUnique,dataInStatesClusterMeans,dataInStatesClusterStds,dataRandStates)
#
#         # Move the rand data set based upon kmeans classification.
#         dataInCurrent = kmeansMove(dataInCurrent,currentLabels,currentMeans,currentStds,dataXMin,dataXMax,dataYMin,dataYMax)
#         dataRandStates = kmeansMove(dataRandStates,dataRandLabels,dataRandCurrentMeans,dataRandCurrentStds,dataXMin,dataXMax,dataYMin,dataYMax)
#
#         # Calculate kalman sense probability.
#         dataRandProb = kalmanSense(dataRandStates,dataInCurrent)
#         dataRandStates = kalmanResample(dataRandStates,dataRandProb)
#
#         pyplot.figure(1)
#         pyplot.clf()
#         pyplot.show()
#         pyplot.plot(dataInStates[:dataInCnt,0:1],dataInStates[:dataInCnt,1:2],'-b')
#         pyplot.plot(dataInStates[dataInCnt:dataInCnt + frameCount,0:1],dataInStates[dataInCnt:dataInCnt + frameCount,1:2],'-r')
#         pyplot.scatter(dataRandStates[:,0],dataRandStates[:,1],color=numpy.random.rand(3,1))
#         pyplot.axis([dataXMin,dataXMax,dataYMin,dataYMax])
#         pyplot.draw()
#
#     print "done"
