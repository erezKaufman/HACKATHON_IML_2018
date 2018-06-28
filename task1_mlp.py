import numpy as np
import sys
from sklearn.neural_network import MLPClassifier

class RawData:

    def __init__(self,fileName,fixed_size,trainSize,testSize,valSize):
        self._file = fileName
        self.parseFile(fixed_size)
        self.divideToSets(trainSize,testSize,valSize)

    def parseFile(self,fixed_size):
        file_open= open(self._file,'r')
        returned_list = []
        for line in file_open:
            while len(line) > fixed_size:
                returned_list.append((line[0:fixed_size]))
                line = line[fixed_size + 1:]
        self._rawData = returned_list

    def divideToSets(self,trainSize,testSize,valSize):
        origin_length = len(self._rawData)
        number_for_train = round(origin_length*(trainSize/100))
        number_for_test = round(origin_length*(testSize/100))
        number_for_val = round(origin_length*(valSize/100))
        random_data = np.random.permutation(self._rawData)
        self._raw_train = random_data[0:number_for_train]
        self._raw_test = random_data[number_for_train:number_for_train+number_for_test]
        self._raw_val = random_data[number_for_train+number_for_test:-1]


    def get_train(self):

        x_train = [x[:-1] for x in self._raw_train]
        y_train = self._raw_train[:][-1]
        return np.asarray(x_train),np.asarray(y_train)

    def get_test(self):
        x_test = [x[:-20] for x in self._raw_test]
        y_test = [x[-20:] for x in self._raw_test]
        return np.asarray(x_test),np.asarray(y_test)

    def get_val(self):
        x_val = [x[:-1] for x in self._raw_val]
        y_val = self._raw_val[:][-1]
        return np.asarray(x_val), np.asarray(y_val)




if __name__ == '__main__':
    class_data = RawData(sys.argv[1],181,70,20,10)
    x_train, y_train = class_data.get_train()
    x_test, y_test = class_data.get_test()
    x_val, y_val = class_data.get_val()
    clf = MLPClassifier(solver='sgd',hidden_layer_sizes=(5,2))
    clf.fit(x_train,y_train)

