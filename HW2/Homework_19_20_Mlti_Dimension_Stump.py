
# coding: utf-8

# In[62]:


import numpy as np
from random import choice
import copy


# In[63]:


def get_data(file_path):
    X_y = []
    training_set = open(file_path)
    for line in training_set:
         for s in line.strip().split():
                X_y.append(float(s))               
    X_y = np.asarray(X_y).reshape(-1, 10)
    X, y = np.hsplit(X_y, [9])
    y = np.squeeze(y)
    return X, y


# In[97]:


#One dimension stump
def one_dim_stump(X,y):
    
    X = np.squeeze(X)
    X_ = copy.deepcopy(X)
    X_.sort()
    thetas = np.asarray([X_[0]-1]+[(X_[i]+X_[i+1])/2 for i in range(X_.shape[0]-1)]+[X_[0]+1])
    
    Ein_lowest = 1
    sign = 1
    theta_sign_best = []
    
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
                #Compare Ein of positive ray with current lowest Ein
                if Ein_pos < Ein_lowest: 
                    Ein_lowest = Ein_pos
                    sign = 1
                    #If we get a strictly better theta, clear the list then append
                    theta_sign_best.clear()
                    theta_sign_best.append([theta,sign])
                #If we get a different theta with same Ein, append
                elif Ein_pos == Ein_lowest: 
                    sign = 1
                    theta_sign_best.append([theta,sign])
            else: 
                if Ein_neg < Ein_lowest:
                    Ein_lowest = Ein_neg
                    sign = -1
                    #If we get a strictly better theta, clear the list then append
                    theta_sign_best.clear()
                    theta_sign_best.append([theta,sign])
                #If we get a different theta with same Ein, append
                elif Ein_pos == Ein_lowest: 
                    sign = -1
                    theta_sign_best.append([theta,sign])
                    
    return Ein_lowest, theta_sign_best


# In[195]:


#Multi dimension stump
def multi_dim_stump(X, y):
    
    Ein_Optimal = 1
    # Param_Optimal: a list to record best Parameters, including theta, sign, idx
    Param_Optimal = []
    # Record the idx which leads to best performance
    idx = 0
    for feature in np.hsplit(X, X.shape[1]):
        #Get one dim stump performance
        Ein, param = one_dim_stump(feature, y)
        #Append idx into param
        for i in range(len(param)):
            param[i].append(idx)
            
        if Ein < Ein_Optimal:
            Ein_Optimal = Ein
            Param_Optimal = param
            
        elif Ein == Ein_Optimal:
            Param_Optimal = Param_Optimal + param
        # index plus one
        idx += 1
        
    Param_Optimal = choice(Param_Optimal)
    return Ein_Optimal, Param_Optimal


# In[253]:


if __name__ == '__main__':
    
    #Solution for #19, Get the best parameters and Ein
    X, y = get_data('hw2_train.dat')
    Ein_Optimal, Parameter_Optimal = multi_dim_stump(X, y)
    #If there are more than one optimal theta, choose one randomly
    Theta_Optimal, Sign_Optimal, index = Parameter_Optimal
    #Print best Ein ,theta, sign and index
    print('The Ein, Theta, Sign and index of best stump are : ', Ein_Optimal, Theta_Optimal, Sign_Optimal, index)
    
    #Solution for #20, Test the multi dim stump on test data
    X_test, y_test = get_data('hw2_test.dat')
    if Sign_Optimal == 1:
        y_predict = np.where(X_test[:,index] > Theta_Optimal, 1, -1)
    else:
        y_predict = np.where(X_test[:,index] < Theta_Optimal, 1, -1)
    Eout = sum(y_predict!=y_test) / X_test.shape[0]
    print('The Eout on testing data is : ', Eout)

