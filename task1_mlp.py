import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.neural_network import MLPClassifier

class RawData:

    def __init__(self,fileName,k,trainSize,testSize,valSize):
        self._file = fileName
        self.parseFile(k,trainSize,testSize,valSize)
        # self.divideToSets(trainSize,testSize,valSize)


    def parseFile(self,k,trainSize,testSize,valSize):
        file_open= open(self._file,'r')
        self._raw_train = []
        self._raw_val= []
        self._raw_val = []
        for i, line in enumerate(file_open):
            line_length = len(line)
            number_for_train = round(line_length * (trainSize / 100))
            number_for_test = round(line_length* (testSize / 100))
            number_for_val = round(line_length * (valSize / 100))
            self._raw_train.append(line[0:number_for_train])
            self._raw_val.append(line[number_for_train:number_for_train + number_for_test])
            self._raw_val.append(line[number_for_train+number_for_test:-1])


    # def divideToSets(self,trainSize,testSize,valSize):
    #     origin_length = len(self._rawData)
    #     number_for_train = round(origin_length*(trainSize/100))
    #     number_for_test = round(origin_length*(testSize/100))
    #     number_for_val = round(origin_length*(valSize/100))
    #     random_data = np.random.permutation(self._rawData)
    #     self._raw_train = random_data[0:number_for_train]
    #     self._raw_test = random_data[number_for_train:number_for_train+number_for_test]
    #     self._raw_val = random_data[number_for_train+number_for_test:-1]


    def get_train(self,index,k):
        y_train = []
        x_train = []
        line_length = len(self._raw_train[index])
        x_train.append(self._raw_train[index][0:k])
        y_train.append(self._raw_train[index][k])
        for i in range(1,(line_length-k),1):
            x_train.append((self._raw_train[index][i:i+k]))
            y_train.append((self._raw_train[index][i+k]))
        return np.asarray(x_train).reshape((len(x_train),1)),np.asarray(y_train).reshape((len(y_train),1))

    def get_test(self,index,k):
        x_test = []
        y_test = []
        x_test.append(self._raw_val[index][0:k])
        y_test.append(self._raw_val[index][k])
        for i in range(1, (len(self._raw_val[index]) - k), 1):
            x_test.append((self._raw_val[index][i:i + k]))
            y_test.append((self._raw_val[index][i + k]))
        return np.asarray(x_test), np.asarray(y_test)

    def get_val(self,index,k):
        x_val = []
        y_val = []
        x_val.append(self._raw_val[index][0:k])
        y_val.append(self._raw_val[index][k])
        for i in range(1, (len(self._raw_val[index]) - k), 1):
            x_val.append((self._raw_val[index][i:i + k]))
            y_val.append((self._raw_val[index][i + k]))
        return np.asarray(x_val), np.asarray(y_val)

def create_permutations(k):
    if k ==1:
        return ['0','1']
    else:
        ret_list = []
        last_permutations = create_permutations(k-1)
        for var in last_permutations:
            ret_list.append(var+'0')
            ret_list.append(var+'1')
        return  ret_list + last_permutations

def create_features(raw_x_set,k):
    permutaions =  create_permutations(k)
    feature_mat = np.zeros((len(raw_x_set),len(permutaions)+k+1))
    # print(feature_mat.shape)
    permutaions_features = []
    sum_feature = [0]*(k+1)
    for perm_index, i in enumerate(permutaions):
        cur_count = 0
        for x_index, j in enumerate(raw_x_set):
            if i in j: cur_count = j[0].count(i)



            if perm_index == 0:
                count_sums(x_index,perm_index, j, feature_mat,len(permutaions))

            feature_mat[x_index][perm_index] = cur_count
    # print(permutaions_features)
    return feature_mat

def count_sums(x_index, perm_index, j, feature_mat,perm_length):
    count = 0
    for bit in j[0]:
        if bit == '1': count += 1
    feature_mat[x_index][perm_length+count] = 1

# def create_dataFrame(x_set,features):
#     df = pd.DataFrame(index=features,columns=x_set)
#     print(df)

print('Creating Datasets...k')
k = 20
if __name__ == '__main__':
    class_data = RawData(sys.argv[1], k, 70, 20, 10)
    x_train, y_train = class_data.get_train(0, k)
    x_test, y_test = class_data.get_test(0, k)
    x_val, y_val = class_data.get_val(0, k)
    # print(x_train)
    # print(x_train.shape)
    # print(y_train.shape)
    feature_mat = create_features(x_train, k)
    # create_dataFrame(x_train,features)
    # clf = MLPClassifier(solver='sgd',hidden_layer_sizes=(5,2))
    # clf.fit(x_train.astype('float64'),y_train.astype('float64'))

# Network
def pred_action(x_test, n_pred, model):
    y_hat = np.zeros((m, n_pred))
    for i in range(n_pred):
        y_hat[:, i] = round(model.predict(x_test))

        x_test = np.append(x_test[:, 1:], y_hat[:,i], axis=1)

    return y_hat

max_features = 1024; m = x_test.shape[0]; n_pred = 20

''' Create LSTM Network '''

n_layers = 10; n_units = 30; d = feature_mat.shape[1]
# model = Sequential(Dense(n_units, input_shape=(d,), activation='relu'))
# for i in range(n_layers):
#     model.add(Dense(n_units, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

inputs = Input(shape=(d,))
x=inputs
for i in range(n_layers):
    x = Dense(30, activation='relu')(x)
outputs = Dense(1, activation='softmax')(x)

print('Training...')
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

''' Train & Test Network '''

# (Assumptions: x_test = m x 180 matrix
#    y_hat_mat, y_test_mat = mx20 matrix of predicted next 20 digits for each example)
model.fit(feature_mat, y_train, batch_size=16, epochs=10)

y_hat_mat = pred_action(x_test, n_pred, model)
DecayW = np.repeat(2**(-np.array([range(1,n_pred)], dtype=float)), m, axis=0)

score = np.sum((y_hat_mat == y_test_mat) * DecayW, axis=1)