import numpy
import cv2
import json
from math import *
import matplotlib.pyplot as pyplot

def calcStates(dataIn,movAvg):
    # dataOut[row][0] = # x position
    # dataOut[row][1] = # y position
    # dataOut[row][2] = # heading
    # dataOut[row][3] = # velocity
    # dataOut[row][4] = # turn rate
    # dataOut[row][5] = # velocity average
    # dataOut[row][6] = # turn rate average
    # dataOut[row][7] = # movAvg moving average of velocity
    # dataOut[row][8] = # movAvg moving average of turn rate
    dataOut = numpy.zeros((dataIn.shape[0],9))
    dataOut[0:,0] = dataIn[0:,0]
    dataOut[0:,1] = dataIn[0:,1]
    dataOut[1:,2] = numpy.arctan2(dataOut[1:,1] - dataOut[0:-1,1],dataOut[1:,0] - dataOut[0:-1,0])
    dataOut[1:,3] = numpy.sqrt((dataOut[1:,0] - dataOut[1 - 1:-1,0])**2 + (dataOut[1:,1] - dataOut[1 - 1:-1,1])**2)
    dataOut[2:,4] = dataOut[2:,2] - dataOut[1:-1,2]
    dataOut[2:,5] = numpy.cumsum(dataOut[2:,3])/numpy.linspace(1,dataOut.shape[0] - 2,dataOut.shape[0] - 2)
    dataOut[3:,6] = numpy.cumsum(dataOut[3:,4])/numpy.linspace(1,dataOut.shape[0] - 3,dataOut.shape[0] - 3)
    dataOut[1 + movAvg:,7] = (numpy.cumsum(dataOut[1:,3])[movAvg:] - numpy.cumsum(dataOut[1:,3])[:-movAvg])/(movAvg*1.0)
    dataOut[2 + movAvg:,8] = (numpy.cumsum(dataOut[2:,3])[movAvg:] - numpy.cumsum(dataOut[2:,3])[:-movAvg])/(movAvg*1.0)
    return dataOut

def predictStates(dataInStates,frameTotal,xMin,xMax,yMin,yMax):
    dataOutStates = numpy.zeros([frameTotal,dataInStates.shape[1]])

    for frameCurrent in range(0,frameTotal):
        if frameCurrent == 0:
            dataOutStates[frameCurrent,8] = dataInStates[-1,8]
            dataOutStates[frameCurrent,7] = dataInStates[-1,7]
            dataOutStates[frameCurrent,6] = dataInStates[-1,6]
            dataOutStates[frameCurrent,5] = dataInStates[-1,5]
            dataOutStates[frameCurrent,4] = dataInStates[-1,6]
            dataOutStates[frameCurrent,3] = dataInStates[-1,5]
            dataOutStates[frameCurrent,2] = dataInStates[-1,2] + dataOutStates[frameCurrent,6]

            # Keep the angle [-pi,+pi]
            if dataOutStates[frameCurrent,2] > pi:
                dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] - 2*pi
            if dataOutStates[frameCurrent,2] < -pi:
                dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] + 2*pi

            dataOutStates[frameCurrent,1] = dataInStates[-1,1] + dataOutStates[frameCurrent,5]*sin(dataOutStates[frameCurrent,2])
            dataOutStates[frameCurrent,0] = dataInStates[-1,0] + dataOutStates[frameCurrent,5]*cos(dataOutStates[frameCurrent,2])

        if frameCurrent >= 1:
            dataOutStates[frameCurrent,8] = dataInStates[-1,8]
            dataOutStates[frameCurrent,7] = dataInStates[-1,7]
            dataOutStates[frameCurrent,6] = dataInStates[-1,6]
            dataOutStates[frameCurrent,5] = dataInStates[-1,5]
            dataOutStates[frameCurrent,4] = dataInStates[-1,6]
            dataOutStates[frameCurrent,3] = dataInStates[-1,5]
            dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent - 1,2] + dataOutStates[frameCurrent,6]

            # Keep the angle [-pi,+pi]
            if dataOutStates[frameCurrent,2] > pi:
                dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] - 2*pi
            if dataOutStates[frameCurrent,2] < -pi:
                dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] + 2*pi

            dataOutStates[frameCurrent,1] = dataOutStates[frameCurrent - 1,1] + dataOutStates[frameCurrent,5]*sin(dataOutStates[frameCurrent,2])
            dataOutStates[frameCurrent,0] = dataOutStates[frameCurrent - 1,0] + dataOutStates[frameCurrent,5]*cos(dataOutStates[frameCurrent,2])

        # Wall bounce symmetry.
        if dataOutStates[frameCurrent,0] <= xMin or dataOutStates[frameCurrent,0] >= xMax:
            if dataOutStates[frameCurrent,2] >= 0:
                dataOutStates[frameCurrent,2] =  pi - dataOutStates[frameCurrent,2]
            if dataOutStates[frameCurrent,2] < 0:
                dataOutStates[frameCurrent,2] = -pi - dataOutStates[frameCurrent,2]

        # Wall bounce symmetry.
        if dataOutStates[frameCurrent,1] <= yMin or dataOutStates[frameCurrent,1] >= yMax:
            dataOutStates[frameCurrent,2] = -1*dataOutStates[frameCurrent,2]

        # Keep the angle [-pi,+pi]
        if dataOutStates[frameCurrent,2] > pi:
            dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] - 2*pi
        if dataOutStates[frameCurrent,2] < -pi:
            dataOutStates[frameCurrent,2] = dataOutStates[frameCurrent,2] + 2*pi

    return dataOutStates

def stateInitialize(dataInput,samplesTotal):
    state = numpy.zeros((samplesTotal,2))
    state[:,0] = numpy.random.uniform(min(dataInput[:,0]),max(dataInput[:,0]),(samplesTotal))
    state[:,1] = numpy.random.uniform(min(dataInput[:,1]),max(dataInput[:,1]),(samplesTotal))
    return state

def probInitialize(state):
    prob = numpy.ones(state.shape)/numpy.prod(state.size)
    return prob

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



# Loop through all of the data, perform predictions, and plot.
frameTotal = 60
movAvg = 50
dataInput = numpy.array(json.loads(open('hexbug-training_video-centroid_data').read()))
dataInputStates = calcStates(dataInput,movAvg)
dataXMin = numpy.amin(dataInputStates[:,0])
dataXMax = numpy.amax(dataInputStates[:,0])
dataYMin = numpy.amin(dataInputStates[:,1])
dataYMax = numpy.amax(dataInputStates[:,1])

for frameCurrent in range(frameTotal,dataInputStates.shape[0]):
    dataPredict = predictStates(dataInputStates[0:frameCurrent,:],frameTotal,dataXMin,dataXMax,dataYMin,dataYMax)
    pyplot.ion()
    pyplot.figure(1)
    pyplot.clf()
    pyplot.show()
    pyplot.plot(dataInputStates[0:frameCurrent,0],dataInputStates[0:frameCurrent,1],'-b')
    pyplot.plot(dataInputStates[frameCurrent:frameCurrent + frameTotal,0],dataInputStates[frameCurrent:frameCurrent + frameTotal,1],'-r')
    pyplot.plot(dataPredict[:,0],dataPredict[:,1],'or')
    pyplot.axis([dataXMin,dataXMax,dataYMin,dataYMax])
    pyplot.draw()
