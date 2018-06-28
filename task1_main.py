# from sklearn.neural_network import MLPClassifier
import sys
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import Embedding
# from keras.layers import LSTM


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
                returned_list.append(line[0:fixed_size])
                line = line[fixed_size + 1:]
        self._rawData = returned_list

    def divideToSets(self,trainSize,testSize,valSize):
        origin_length = len(self._rawData)
        number_for_train = round(origin_length*(trainSize/100))
        number_for_test = round(origin_length*(testSize/100))
        number_for_val = round(origin_length*(valSize/100))
        random_data = np.random.permutation(self._rawData)
        self._raw_train = random_data[0:number_for_train]
        self._raw_test = random_data[number_for_train:number_for_test]
        self._raw_val = random_data[number_for_test:number_for_val]


    def get_train(self):

        x_train = [x[:-1] for x in self._raw_train]
        y_train = self._raw_train[:][-1]
        print(len(y_train[0]))
        return x_train,y_train

    def createTestData(self):
        x_test = [x[:-1] for x in self._raw_test]
        y_test = self._raw_test[:][-1]
        print(len(y_test[0]))
        return x_test,y_test

    def createValData(self):
        x_val = [x[:-1] for x in self._raw_val]
        y_val = self._raw_val[:][-1]
        print(len(y_val[0]))
        return x_val, y_val




if __name__ == '__main__':
    class_data = RawData(sys.argv[1],181,70,20,10)
    x_train, y_train = class_data.get_train()

    # class_data.parseFile(181)
    # print(class_data.create_sets_of_fix_size(181))


# # LSTM
# max_features = 1024
#
# ''' Create LSTM Network '''
# model = Sequential()
# model.add(Embedding(max_features, output_dim=256))
# model.add(LSTM(128))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='hard_sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
#
# ''' Train Network '''
# model.fit(x_train, y_train, batch_size=16, epochs=10) # x_train in size 180, y_train in size 1
#
# y_hat = model.predict(x_test, y_test)
