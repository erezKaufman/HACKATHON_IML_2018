import numpy as np
import sys
import pandas as pd

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
        return  ret_list

def create_features(raw_x_set,k):
    permutaions =  create_permutations(k)
    permutaions_features = []
    sum_feature = [0]*(k+1)
    for index, i in enumerate(permutaions):
        cur_count = 0
        for j in raw_x_set:
            if j == i:

                cur_count+=1
            count_sums(index, j, sum_feature)

        permutaions_features.append(cur_count)
    return np.asarray(permutaions_features+sum_feature)

def count_sums(index, j, sum_feature):
    if index == 0:
        count = 0
        for bit in j[0]:
            if bit == '1': count += 1
        sum_feature[count] += 1

# def create_dataFrame(x_set,features):
#     df = pd.DataFrame(index=features,columns=x_set)
#     print(df)

if __name__ == '__main__':
    class_data = RawData(sys.argv[1],181,70,20,10)
    x_train, y_train = class_data.get_train(0,5)
    x_test, y_test = class_data.get_test(0,5)
    x_val, y_val = class_data.get_val(0,5)
    print(x_train)
    # print(x_train.shape)
    # print(y_train.shape)
    # features = create_features(x_train,5)
    # create_dataFrame(x_train,features)
    # clf = MLPClassifier(solver='sgd',hidden_layer_sizes=(5,2))
    # clf.fit(x_train.astype('float64'),y_train.astype('float64'))

