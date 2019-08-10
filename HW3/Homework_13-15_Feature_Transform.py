
import numpy as np
from random import choice
import copy

def target_function(x1, x2):
    if x1*x1+x2*x2-0.6>0:
        return 1
    else: return -1
    
def generate_data(N=1000):
    #Generate X randomly in N * 2 matrix
    X = np.random.uniform(-1, 1, (N,2))
    y = []
    #Generate constant item for X
    x_0 = np.ones((N,1))
    #Generate y
    for i in range(N):
        #Flip y with 10% probility
        if np.random.uniform() > 0.1:
            y.append(target_function(X[i,0], X[i,1]))
        else: 
            y.append(-1*target_function(X[i,0], X[i,1]))
            
    return np.hstack((x_0, X)), np.asarray(y)


def feature_transform(X):
    x1_x2 = (X[:,1]*X[:,2]).reshape(-1,1)
    x1_square = (X[:,1]*X[:,1]).reshape(-1,1)
    x2_square = (X[:,2]*X[:,2]).reshape(-1,1)
    return np.hstack((X, x1_x2, x1_square, x2_square))

#Get the weight of Linear Regression
def LR_Weight(X, y):
    # The close solution of Linear Regression is 
    # w = pseudo_inverse(X) * y
    # Use numpy calculate the pseudo inverse of X,
    X_pinv = np.linalg.pinv(X)
    w = np.dot(X_pinv, y)
    
    return w

#A function to calculate Ein
def Calculate_Error(X, y, w):
    y_predict = np.sign(np.dot(X, w))
    error_num = sum(y_predict != y)
    err_rate = error_num / X.shape[0]
    
    return err_rate

if __name__ == '__main__':
    #Solution for #13, calculate the average Ein of
    #linear regression weights of Xs without transform
    Total_Ein =0
    for i in range(500):
        #Generate data
        X, y = generate_data(N=500)
        #Get the weights of Linear Regression
        w = LR_Weight(X, y)
        #Calculate Ein
        Ein = Calculate_Error(X, y, w)
        #Calculate Total Ein
        Total_Ein += Ein
    #Print Average Ein
    print('The Average Ein is : ', Total_Ein/500)
    
    #Solution for #14, calculate the linear regression weights 
    #of Xs with 2-degree polynomial transform
    Total_w = 0
    for i in range(1000):
        X, y = generate_data(N=1000)
        X_t = feature_transform(X)
        w_t = LR_Weight(X_t, y)
        Total_w += w_t
        Average_w = Total_w/1000
    print('The w of transformed X is: ',Average_w)
    
    #Solution for #15, calculate the out-of-sample Eout 
    Total_Eout = 0
    for i in range(1000):
        X, y = generate_data(N=1000)
        X_t = feature_transform(X)
        Eout = Calculate_Error(X_t, y, Average_w)
        Total_Eout += Eout
        Average_Eout = Total_Eout/1000
    print('The Average Eout is : ', Average_Eout)

