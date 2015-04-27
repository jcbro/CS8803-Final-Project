import numpy
import cv2
import json
from math import *
import matplotlib.pyplot as pyplot
import scipy.cluster
import time
import pandas

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
    dataOut = numpy.zeros((dataIn.shape[0],9))
    dataOut[0:,0] = dataIn[0:,0]
    dataOut[0:,1] = dataIn[0:,1]
    dataOut[1:,2] = numpy.arctan2(dataOut[1:,1] - dataOut[0:-1,1],dataOut[1:,0] - dataOut[0:-1,0])
    dataOut[1:,3] = numpy.sqrt((dataOut[1:,0] - dataOut[1 - 1:-1,0])**2 + (dataOut[1:,1] - dataOut[1 - 1:-1,1])**2)
    dataOut[2:,4] = dataOut[2:,2] - dataOut[1:-1,2]

    # Eliminate turn rates of greater than pi/2 by setting to 0. Eliminates weighting wall bounces in other calculations.
    dataOut[dataOut[:,4] > pi/2,4] = 0
    dataOut[dataOut[:,4] < -pi/2,4] = 0

    dataOut[1:,5] = numpy.cumsum(dataOut[1:,3])/numpy.linspace(1,dataOut.shape[0] - 1,dataOut.shape[0] - 1)
    dataOut[2:,6] = numpy.cumsum(dataOut[2:,4])/numpy.linspace(1,dataOut.shape[0] - 2,dataOut.shape[0] - 2)
    dataOut[0 + movAvg:,7] = (numpy.cumsum(dataOut[0:,3])[movAvg:] - numpy.cumsum(dataOut[0:,3])[:-movAvg])/(movAvg*1.0)
    dataOut[1 + movAvg:,8] = (numpy.cumsum(dataOut[1:,4])[movAvg:] - numpy.cumsum(dataOut[1:,4])[:-movAvg])/(movAvg*1.0)

    return dataOut

def predictStates(dataInStates,frameTotal,xMin,xMax,yMin,yMax):
    dataOutStates = numpy.zeros([frameTotal,dataInStates.shape[1]])

    for frameCurrent in range(0,frameTotal):
        if frameCurrent == 0:
            dataOutStates[frameCurrent,8] = dataInStates[-1,8]
            dataOutStates[frameCurrent,7] = dataInStates[-1,7]
            dataOutStates[frameCurrent,6] = dataInStates[-1,6]
            dataOutStates[frameCurrent,5] = dataInStates[-1,5]
            dataOutStates[frameCurrent,4] = dataInStates[-1,8]
            dataOutStates[frameCurrent,3] = dataInStates[-1,7]
            dataOutStates[frameCurrent,2] = dataInStates[-1,2] + dataOutStates[frameCurrent,8]

            # Keep the angle within [-pi,+pi]
            if dataOutStates[frameCurrent,2] > pi:
                dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] - 2*pi
            if dataOutStates[frameCurrent,2] < -pi:
                dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] + 2*pi

            dataOutStates[frameCurrent,1] = dataInStates[-1,1] + dataOutStates[frameCurrent,7]*sin(dataOutStates[frameCurrent,2])
            dataOutStates[frameCurrent,0] = dataInStates[-1,0] + dataOutStates[frameCurrent,7]*cos(dataOutStates[frameCurrent,2])

        if frameCurrent >= 1:
            dataOutStates[frameCurrent,8] = dataInStates[-1,8]
            dataOutStates[frameCurrent,7] = dataInStates[-1,7]
            dataOutStates[frameCurrent,6] = dataInStates[-1,6]
            dataOutStates[frameCurrent,5] = dataInStates[-1,5]
            dataOutStates[frameCurrent,4] = dataInStates[-1,8]
            dataOutStates[frameCurrent,3] = dataInStates[-1,7]
            dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent - 1,2] + dataOutStates[frameCurrent,8]

            # Keep the angle within [-pi,+pi]
            if dataOutStates[frameCurrent,2] > pi:
                dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] - 2*pi
            if dataOutStates[frameCurrent,2] < -pi:
                dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] + 2*pi

            dataOutStates[frameCurrent,1] = dataOutStates[frameCurrent - 1,1] + dataOutStates[frameCurrent,7]*sin(dataOutStates[frameCurrent,2])
            dataOutStates[frameCurrent,0] = dataOutStates[frameCurrent - 1,0] + dataOutStates[frameCurrent,7]*cos(dataOutStates[frameCurrent,2])

        # Wall bounce symmetry.
        if dataOutStates[frameCurrent,0] <= xMin or dataOutStates[frameCurrent,0] >= xMax:
            if dataOutStates[frameCurrent,2] >= 0:
                dataOutStates[frameCurrent,2] =  pi - dataOutStates[frameCurrent,2]
            if dataOutStates[frameCurrent,2] < 0:
                dataOutStates[frameCurrent,2] = -pi - dataOutStates[frameCurrent,2]

        # Wall bounce symmetry.
        if dataOutStates[frameCurrent,1] <= yMin or dataOutStates[frameCurrent,1] >= yMax:
            dataOutStates[frameCurrent,2] = -1*dataOutStates[frameCurrent,2]

        # Keep the angle within [-pi,+pi]
        if dataOutStates[frameCurrent,2] > pi:
            dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] - 2*pi
        if dataOutStates[frameCurrent,2] < -pi:
            dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] + 2*pi

    return dataOutStates

def kalmanRandInitStates(dataInputStates,samplesTotal):
    dataRandStates = numpy.zeros((samplesTotal,dataInputStates.shape[1]))
    for stateCnt in range(dataInputStates.shape[1]):
        dataRandStates[:,stateCnt] = numpy.random.uniform(min(dataInputStates[:,stateCnt]),max(dataInputStates[:,stateCnt]),samplesTotal)
    return dataRandStates

def kalmanRandInitProb(dataRandStates):
    dataRandProb = numpy.ones((dataRandStates.shape[0],1))/numpy.prod([dataRandStates.shape[0],1])
    return dataRandProb

def stateMove(stateIn):
    stateOut[0] = stateIn[0] + stateIn[5]*cos(stateIn[2] + stateIn[6])
    stateOut[1] = stateIn[1] + stateIn[5]*sin(stateIn[2] + stateIn[6])
    stateOut[2] = stateIn[2]
    stateOut[3] = stateIn[3]
    stateOut[4] = stateIn[4]
    stateOut[5] = stateIn[5]
    stateOut[6] = stateIn[6]
    return stateOut

def stateSense(stateIn):
    prob = 1
    return prob

def noviceKalmanPrediction(dataInput):
    # Calculate other state values from the base x,y.
    movAvg = 100
    dataInputStates = calcStates(dataInput,movAvg)
    pyplot.ion()
    pyplot.figure(1)
    pyplot.clf()
    pyplot.show()
    pyplot.hist(dataInputStates[:,4], bins=20, normed=True, cumulative=False)
    pyplot.draw()
    time.sleep(5)

    # Loop through all of the data, perform predictions, and plot.
    dataXMin = numpy.amin(dataInputStates[:,0])
    dataXMax = numpy.amax(dataInputStates[:,0])
    dataYMin = numpy.amin(dataInputStates[:,1])
    dataYMax = numpy.amax(dataInputStates[:,1])
    frameTotal = 60
    for frameCurrent in range(frameTotal,dataInputStates.shape[0]):
        dataPredict = predictStates(dataInputStates[0:frameCurrent,:],frameTotal,dataXMin,dataXMax,dataYMin,dataYMax)
        pyplot.ion()
        pyplot.figure(2)
        pyplot.clf()
        pyplot.show()
        pyplot.plot(dataInputStates[0:frameCurrent,0],dataInputStates[0:frameCurrent,1],'-b')
        pyplot.plot(dataInputStates[frameCurrent:frameCurrent + frameTotal,0],dataInputStates[frameCurrent:frameCurrent + frameTotal,1],'-r')
        pyplot.plot(dataPredict[:,0],dataPredict[:,1],'or')
        pyplot.axis([dataXMin,dataXMax,dataYMin,dataYMax])
        pyplot.draw()

    return

def kMeansPrediction(dataInput):
    # Randomly down sample the data set for speed.
    sampleSize = 4000
    dataInputSample = dataInput[numpy.random.randint(0,dataInput.shape[0],sampleSize)]

    # Calculate other state values from the base x,y.
    movAvg = 50
    dataInputStates = calcStates(dataInputSample,movAvg)

    # Whiten the state values for k means clustering.
    dataInputStatesWhiten = scipy.cluster.vq.whiten(dataInputStates)
    kmeansCentroids = 2
    kmeansIter = 20
    dataClusters,_ = scipy.cluster.vq.kmeans(dataInputStatesWhiten[:,2:9],kmeansCentroids,kmeansIter)

    # Label the data set based upon the clusters.
    dataInputLabels,_ = scipy.cluster.vq.vq(dataInputStatesWhiten[:,2:9],dataClusters)

    # Concatenate the labels onto the original training data set.
    dataInputLabels.shape = (dataInputLabels.shape[0],1)
    dataInputLabeled = numpy.concatenate((dataInputStates,dataInputLabels),axis=1)

    # Loop through each unique label and scatter plot it with a random color to see where the clusters are located.
    dataInputLabelsUnique = numpy.unique(dataInputLabels)
    pyplot.ion()
    pyplot.figure(1)
    pyplot.clf()
    dataXMin = numpy.amin(dataInputStates[:,0])
    dataXMax = numpy.amax(dataInputStates[:,0])
    dataYMin = numpy.amin(dataInputStates[:,1])
    dataYMax = numpy.amax(dataInputStates[:,1])
    pyplot.axis([dataXMin,dataXMax,dataYMin,dataYMax])
    for label in dataInputLabelsUnique:
        dataFiltered = dataInputLabeled[dataInputLabeled[:,9] == label,:]
        pyplot.show()
        pyplot.scatter(dataFiltered[:,0],dataFiltered[:,1],color=numpy.random.rand(3,1))
        pyplot.draw()

    return

def kalmanPrediction(dataInput):
    # Calculate other state values from the base x,y.
    movAvg = 10
    dataInputStates = calcStates(dataInput,movAvg)

    # Initialize samples uniformly random over the entire state space.
    samplesTotal = 300
    dataRandStates = kalmanRandInitStates(dataInputStates,samplesTotal)
    dataRandProb = kalmanRandInitProb(dataRandStates)
    pyplot.ion()
    pyplot.figure(3)
    pyplot.clf()
    dataXMin = numpy.amin(dataInputStates[:,0])
    dataXMax = numpy.amax(dataInputStates[:,0])
    dataYMin = numpy.amin(dataInputStates[:,1])
    dataYMax = numpy.amax(dataInputStates[:,1])
    pyplot.axis([dataXMin,dataXMax,dataYMin,dataYMax])
    pyplot.show()
    pyplot.scatter(dataRandStates[:,0],dataRandStates[:,1])
    pyplot.draw()

    frameTotal = 60
    for frameCurrent in range(0,dataInputStates.shape[0]):
        # Resample the existing states based upon the existing probabilities.
        dataRandStates,dataRandProb = kalmanResample(dataRandStates,dataRandProb)
        # Move the states based upon the model.
        dataRandStates = kalmanMove(dataRandStates,frameTotal,dataXMin,dataXMax,dataYMin,dataYMax)
        # Given the current set of states, apply the sensor, and return new set of probabilities.
        dataRandProbSense = kalmanSense(dataRandStates,dataInputStates[frameCurrent,:])
        dataRandProb = dataRandProbSense*dataRandProb
        dataRandProb = dataRandProb/numpy.sum(dataRandProb,axis=0)


        pyplot.figure(3)
        pyplot.clf()
        pyplot.show()
        pyplot.title("frameCurrent = %s" % frameCurrent)
        pyplot.plot(dataInputStates[0:frameCurrent,0],dataInputStates[0:frameCurrent,1],'-b')
        #pyplot.plot(dataInputStates[frameCurrent:frameCurrent + frameTotal,0],dataInputStates[frameCurrent:frameCurrent + frameTotal,1],'-r')
        pyplot.scatter(dataRandStates[:,0],dataRandStates[:,1],color=numpy.random.rand(3,1))
        pyplot.axis([dataXMin,dataXMax,dataYMin,dataYMax])
        pyplot.draw()

    return

def kalmanResample(dataRandStates,dataRandProb):
    # Randomly sample states dataRandStates with replacement given probabilities dataRandProb.
    dataTemp1 = numpy.array(range(0,dataRandStates.shape[0]))
    dataTemp2 = len(dataRandStates)
    indexesRandSample = numpy.random.choice(dataTemp1.flatten(),size=dataTemp2,replace=True,p=dataRandProb.flatten())
    dataTemp1 = dataRandStates[indexesRandSample,:]
    dataTemp2 = dataRandProb[indexesRandSample,:]
    dataTemp1 = numpy.reshape(dataTemp1,dataRandStates.shape)
    dataTemp2 = numpy.reshape(dataTemp2,dataRandProb.shape)
    return dataTemp1,dataTemp2

def kalmanReseed(dataRandStates):
    reseedFraction = 0.2
    return 1

def kalmanSense(dataRandStates,dataInputState):
    sigma = 20
    diff = dataRandStates - numpy.tile(dataInputState,(dataRandStates.shape[0],1))
    diffPercent = diff/numpy.tile(dataInputState,(dataRandStates.shape[0],1))
    diffSq = diffPercent**2
    MSE = numpy.sum(diffSq,axis=1)/dataInputState.size
    dataRandProbOut = numpy.exp(-1*MSE/sigma**2)
    dataRandProbOut = numpy.reshape(dataRandProbOut,(dataRandStates.shape[0],1))
    return dataRandProbOut

def kalmanMove(dataRandStates,frameTotal,xMin,xMax,yMin,yMax):
    frameTotal = 1
    for stateCnt in range(0,dataRandStates.shape[1]):
        dataTemp1 = dataRandStates[stateCnt,:]
        dataTemp1 = numpy.reshape(dataTemp1,(1,dataRandStates.shape[1]))
        dataRandStates[stateCnt,:] = predictStates(dataTemp1,frameTotal,xMin,xMax,yMin,yMax)
    return dataRandStates

dataInput = numpy.array(json.loads(open('hexbug-training_video-centroid_data').read()))
#kMeansPrediction(dataInput)
#time.sleep(5)
noviceKalmanPrediction(dataInput)
#kalmanPrediction(dataInput)
