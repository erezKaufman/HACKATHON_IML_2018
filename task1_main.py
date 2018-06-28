# from sklearn.neural_network import MLPClassifier
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM


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


# LSTM
max_features = 1024

''' Create LSTM Network '''
model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

''' Train Network '''
model.fit(x_train, y_train, batch_size=16, epochs=10)

'''x_test = m x 180 matrix
   y_hat = mx20 matrix of predicted next 20 digits for each example'''
m = x_test.shape[0]; n_pred = 20

def pred_action(x_test, n_pred, model):
    y_hat = np.zeros((m, n_pred))
    for i in range(n_pred):
        y_hat[:, i] = round(model.predict(x_test))

        x_test = np.append(x_test[:, 1:], y_hat[:,i], axis=1)

    return y_hat

y_hat_mat = pred_action(x_test, n_pred, model)