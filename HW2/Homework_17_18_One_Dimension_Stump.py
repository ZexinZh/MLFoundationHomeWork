
import numpy as np
from numpy import random
import copy


def generate_data(dim, noise_rate):
    X = random.uniform(-1, 1, dim)
    X.sort()
    #Choose the idx affected by noise
    idx = random.choice(dim, int(dim*noise_rate), replace=False)
    y = np.sign(X)
    # 20% of y are flipped by the noise
    flipped_y = y * np.asarray([1 if i not in idx else -1 for i in np.arange(20)])
    return X, flipped_y

def get_Ein(X, y):
    thetas = np.asarray([-1.1]+[(X[i]+X[i+1])/2 for i in range(X.shape[0]-1)]+[1.1])
    Ein_lowest = 1
    sign = 1
    
    for theta in thetas:
        #Ys predicted by positive ray with theta
        y_pos = np.where(X > theta, 1, -1)
        #Ys predicted by negative ray with theta
        y_neg = np.where(X < theta, 1, -1)
        #Ein of positive ray
        Ein_pos = sum(y_pos!=y) / X.shape[0]
        #Ein of negative ray
        Ein_neg = sum(y_neg!=y) / X.shape[0]
        #Record the best theta and corresponding sign
        if Ein_pos <= Ein_neg:
            if Ein_pos < Ein_lowest: 
                Ein_lowest = Ein_pos
                sign = 1
                theta_best = theta
        else: 
            if Ein_neg < Ein_lowest: 
                Ein_lowest = Ein_neg
                sign = -1
                theta_best = theta
    #Take care of the two ends of [-1, 1]            
    if theta_best == 1.1 or theta_best == -1.1: 
        theta_best = int(theta_best)
                
    return theta_best, sign, Ein_lowest


if __name__ == '__main__':
    
    Total_Ein = 0
    Total_Eout = 0
    
    for i in range(5000):
        X, y = generate_data(20, 0.2)
        theta, sign, Ein = get_Ein(X, y)
        Total_Ein += Ein
        Total_Eout += 0.5 + 0.3 * sign * (abs(theta) - 1)
    
    print('The answer for #17: ', Total_Ein/5000)
    print('The answer for #18: ', Total_Eout/5000)

