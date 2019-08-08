
import numpy as np

class RandomCyclePLA():
    def __init__(self):
        pass
    
    def get_training_data(self, file_path):
        X_train = []
        y_train = []
        training_set = open(file_path)
        for line in training_set:
            for str in  line.split(' '):
                if '\t' not in str:
                    X_train.append(float(str))
                else:
                    X_train.append(float(str.split('\t')[0]))
                    y_train.append(int(str.split('\t')[1].strip()))     
        X_train = np.asarray(X_train).reshape(-1, 4)
        X_0 = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((X_0, X_train))
        y_train = np.asarray(y_train)
        return X_train, y_train
    
    def fit_count(self, file_path):
        X, y = self.get_training_data(file_path)
        index = np.asarray([i for i in range(X.shape[0])])
        np.random.shuffle(index)
        count = 0
        w = np.zeros((X.shape[1]))
        while True:
            flag = 1
            for i in range(X.shape[0]):
                idx = index[i]
                if np.dot(X[idx, : ], w)*y[idx] <= 0:
                    w += y[idx]*X[idx, : ]
                    count += 1
                    flag = -1
            if flag == 1:
                break
        return w, count

if __name__ == '__main__':
    sum = 0
    for i in range(2000):
        perceptron = RandomCyclePLA()
        sum += perceptron.fit_count('hw1_7_train.dat')[1]
    print(sum / 2000.0)

