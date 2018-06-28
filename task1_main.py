from sklearn.neural_network import MLPClassifier
import sys
import numpy as np

class RawData:

    def __init__(self,fileName):
        self._file = fileName
        self._rawData = 0


    def parseFile(self):
        file_open = open(self._file,'r')
        self._rawData = np.loadtxt(file_open)
        print(min(self._rawData,key=lambda x:x.size))
        print(max(self._rawData,key=lambda x:len(x)))
    def createTrainingData(self):
        # 70 percents
        pass

    def createTestData(self):
        # 20 percents
        pass

    def createValData(self):
        # 10 percents
        pass



if __name__ == '__main__':
    class_data = RawData(sys.argv[1])
    class_data.parseFile()