
import numpy as np

def get_data(file_path):
    X_y = []
    training_set = open(file_path)
    for line in training_set:
         for s in line.strip().split():
                X_y.append(float(s))               
    X_y = np.asarray(X_y).reshape(-1, 21)
    X, y = np.hsplit(X_y, [20])
    x_0 = np.ones((X.shape[0],1))
    y = np.squeeze(y)
    return np.hstack((x_0, X)), y

#The logistic function, aka sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Update the weight with vanilla gradient descent in one round
def vanilla_GD(X, w, y, learning_rate):
    # Calculate gradient at each point, then sum and average it.
    grad = sigmoid(-1*y*np.dot(X, w)).reshape(-1,1)*(-1*y.reshape(-1,1)*X)
    Gradient = sum(grad)/len(X)
    #Update
    w_updated = w - learning_rate*Gradient
    return w_updated

#Update the weight with stochastic gradient descent, using only one data point in one round
#T is how many round you woould like to update, Instead of choosing data point randomly, the
#data would be iterated in cyclic order.
def stochastic_GD(X, w, y, learning_rate, T):
    #Update the weight in cyclic order.
    for i in range(T):
        idx = i % X.shape[0]
        grad = sigmoid(-1*y[idx]*np.dot(X[idx], w))*(-1*y[idx]*X[idx]) 
        w = w - learning_rate*grad
    return w

#A function to calculate Error
def Calculate_Error(X, y, w):
    y_predict = np.sign(sigmoid(np.dot(X,w))-0.5)
    error_num = sum(y_predict != y)
    err_rate = error_num / X.shape[0]
    
    return err_rate

if __name__ == '__main__':
    
    #Solution for #18, get weight by Vanilla Gradient Descent
    #Init Weight w with zeros, other initialization tricks could be implemented.
    X, y = get_data('hw3_train.dat')
    w1 = np.zeros(X.shape[1])
    #Update w 2000 rounds, with fixed learning rate 0.001
    for i in range(2000):    
        w1 = vanilla_GD(X, w1, y, 0.001)
    #Calculate out-of-sample error Eout
    X_test, y_test = get_data('hw3_test.dat')
    Eout_18 = Calculate_Error(X_test, y_test, w1)
    #Print Eout with fixed learning rate 0.001
    print('The out-of-sample error Eout with fixed learning rate 0.001 is : ', Eout_18)
    
    #Solution for #19, tune the fixed learning rate to 0.01
    w2 = np.zeros(X.shape[1])
    #Update w 2000 rounds, with fixed learning rate 0.01
    for i in range(2000):    
        w2 = vanilla_GD(X, w2, y, 0.01)
    Eout_19 = Calculate_Error(X_test, y_test, w2)
    print('The out-of-sample error Eout with fixed learning rate 0.001 is : ', Eout_19)
    
    #Solution for #20, Using stochastic_GD in cyclic order.
    w3 = np.zeros(X.shape[1])
    w3 = stochastic_GD(X, w3, y, 0.001, 2000)
    Eout_20 = Calculate_Error(X_test, y_test, w3)
    print('The out-of-sample error Eout with stochastic_GD is : ', Eout_20)

