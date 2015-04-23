import json
import sys

class point:

    def __init__(self):
        self.frameNum = 0
        self.position = [0, 0]


class dataset:

    def __init__(self):
        self.points = []

    def readFile(self, fileName):
        self.fileName = fileName
        f = open(fileName)
        pos = json.load(f)
        for i in range(len(pos)):
            p = point()
            p.pos = pos[i]
            p.frame = i
            self.points.append(p)

def main(argv):

    testfile = "C:\prediction.txt"

    testData = dataset()
    #testData.readFile(argv[1])
    testData.readFile(testfile)

testfile = "C:\prediction.txt"
testData = dataset()
testData.readFile(testfile)
