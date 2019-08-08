
import numpy as np
import copy

class PocketPLA():
    def __init__(self):
        pass
    
    def get_data(self, file_path):
        X = []
        y = []
        training_set = open(file_path)
        for line in training_set:
            for str in  line.split(' '):
                if '\t' not in str:
                    X.append(float(str))
                else:
                    X.append(float(str.split('\t')[0]))
                    y.append(int(str.split('\t')[1].strip()))     
        X = np.asarray(X).reshape(-1, 4)
        # add a column of 1s as the constant item.
        X_0 = np.ones((X.shape[0], 1))
        X = np.hstack((X_0, X))
        y = np.asarray(y)
        return X, y
    
    def fit(self, file_path):
        count = 0
        X, y = self.get_data(file_path)
        #get the shuffled array as the index of training data, 
        #no need to shuffle the whole training data 
        index = np.asarray([i for i in range(X.shape[0])])
        np.random.shuffle(index)
        w = np.zeros((X.shape[1]))
        #initialize the pocket points as 0, meaning the number of points 
        #classfied correctly by the PLA
        pocket_points = 0
        pocket_w = np.zeros((X.shape[1]))
        for i in range(X.shape[0]):
            idx = index[i]
            if np.dot(X[idx, : ], w)*y[idx] <= 0:
                w += 0.5*y[idx]*X[idx, : ]
                #once w updated, add one to count, till count equals to 50 
                count += 1
                '''''
                correct_points = 0
                for j in range(X.shape[0]):
                    if np.dot(X[j, : ], w)*y[j] > 0:
                        correct_points += 1
                #update w and pocket_points if we find a better w
                if correct_points > pocket_points:
                    pocket_points = correct_points
                    pocket_w = copy.deepcopy(w)
                if count >= 50:
                    break
                '''''
        return w #pocket_w

    def test_score(self, w, file_path):
        counts = 0
        X_test, y_test = self.get_data(file_path)
        for i in range(X_test.shape[0]):
            if np.dot(X_test[i, : ], w)*y_test[i] <= 0:
                counts += 1
        return counts/X_test.shape[0]

if __name__ == '__main__':
    error_rate = 0
    for i in range(2000):
        PocketPerceptron = PocketPLA()
        w  = PocketPerceptron.fit('hw1_18_train.dat')
        c  = PocketPerceptron.test_score(w, 'hw1_18_test.dat')
        error_rate+= c
    print(error_rate / 2000 )

