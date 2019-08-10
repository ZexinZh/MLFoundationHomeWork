import numpy as np


def get_data(file_path):
    X_y = []
    training_set = open(file_path)
    for line in training_set:
         for s in line.strip().split():
                X_y.append(float(s))               
    X_y = np.asarray(X_y).reshape(-1, 3)
    X, y = np.hsplit(X_y, [2])
    x_0 = np.ones((X.shape[0],1))
    y = np.squeeze(y)
    return np.hstack((x_0, X)), y

#Get the weight of Ridge Regression
def Ridge_Weight(X, y, lambda_value):
    # Calculate Ridge Regression Weight
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)+lambda_value*np.eye(X.shape[1])), X.T), y)

#A function to calculate Error
def Calculate_Error(X, y, w):
    scores = np.dot(X, w)
    predicts = np.where(scores >= 0, 1.0, -1.0)
    error_num = sum(predicts != y)
    return (error_num) / X.shape[0]

def Cross_Validation(X, y, lambda_value, n_splits):
    Total_Ecv = 0
    n_samples = X.shape[0]
    #Generate indices
    for indices in np.array_split(np.arange(n_samples), n_splits):
        #Generate training data
        X_train = np.delete(X, indices, axis=0)
        y_train = np.delete(y, indices)
        #Generate validation data
        X_valid = X[indices]
        y_valid = y[indices]
        #Using training data to calculate w
        w = Ridge_Weight(X_train, y_train, lambda_value)
        #Using validation data to calculate Ecv
        Ecv = Calculate_Error(X_valid, y_valid, w)
        Total_Ecv += Ecv
    return Total_Ecv/n_splits

if __name__ == '__main__':
    
    #Solution for #13, lambda value is 10
    X_train, y_train = get_data('hw4_train.dat')
    X_test, y_test = get_data('hw4_test.dat')
    #Calculate weights
    w = Ridge_Weight(X_train, y_train, 10)
    #Calculate Ein
    Ein = Calculate_Error(X_train, y_train, w)
    #Calculate Eout
    Eout = Calculate_Error(X_test, y_test, w)
    #Print Ein and Eout 
    print('Ein is {:.4f}, and Eout is {:.4f}'.format(Ein, Eout))
    
    #Solution for #14-15
    Ein_optimal = 1
    lambda_Ein_optimal = 0
    Eout_of_optimal_Ein = 0
    Eout_optimal = 1
    lambda_Eout_optimal = 0
    Ein_of_optimal_Eout = 0
    
    for i in range(2, -11, -1):
        w = Ridge_Weight(X_train, y_train, pow(10, i))
        Ein = Calculate_Error(X_train, y_train, w)
        Eout = Calculate_Error(X_test, y_test, w)
        
        if Ein < Ein_optimal:
            Ein_optimal = Ein
            lambda_Ein_optimal = i
            Eout_of_optimal_Ein = Eout
            
        if Eout < Eout_optimal:
            Eout_optimal = Eout
            lambda_Eout_optimal = i
            Ein_of_optimal_Eout = Ein
            
    print('Optimal Ein is :', Ein_optimal, 'log10_lambda is ', lambda_Ein_optimal, 'Corresponding Eout is', Eout_of_optimal_Ein)
    print('Optimal Eout is ', Eout_optimal, 'log10_lambda is ', lambda_Eout_optimal,'Corresponding Ein is', Ein_of_optimal_Eout)
    
    #Solution for #16-17
    #Choose first 120 data points as training data, the rest as valid data,
    #No randomness cuz the data are already permutated randomly.
    
    Ein_optimal = 1
    Eout_of_optimal_Ein = 0
    Eval_of_optimal_Ein = 0
    lambda_optimal_Ein = 0

    Eval_optimal = 1
    Eout_of_optimal_Eval = 0
    Ein_of_optimal_Eval = 0
    lambda_optimal_Eval = 0

    split = 120

    for i in range(2, -11, -1):
        w_ridge = Ridge_Weight(X_train[:split], y_train[:split], pow(10, i))
        Ein = Calculate_Error(X_train[:split], y_train[:split], w_ridge)
        Eval = Calculate_Error(X_train[split:], y_train[split:], w_ridge)
        Eout = Calculate_Error(X_test, y_test, w_ridge)

        if Ein < Ein_optimal:
            Ein_optimal = Ein
            lambda_optimal_Ein = i
            Eout_of_optimal_Ein = Eout
            Eval_of_optimal_Ein = Eval

        if Eval < Eval_optimal:
            Eval_optimal = Eval
            lambda_optimal_Eval = i
            Eout_of_optimal_Eval = Eout
            Ein_of_optimal_Eval = Ein
    print('log10_lambda is ', lambda_optimal_Ein, 'Optimal Ein is :', Ein_optimal,  'Corresponding Eval and Eout: ', Eval_of_optimal_Ein, Eout_of_optimal_Ein)
    print('log10_lambda is ', lambda_optimal_Eval,'Optimal Eval is ', Eval_optimal, 'Corresponding Ein and Eout: ', Ein_of_optimal_Eval, Eout_of_optimal_Eval)
    
    #Solution for #18
    w_18 = Ridge_Weight(X_train, y_train, pow(10, lambda_optimal_Eval))
    Ein = Calculate_Error(X_train, y_train, w_18)
    Eout = Calculate_Error(X_test, y_test, w_18)
    print('#18: Ein: ', Ein, ', Eout: ', Eout)
    
    #Solution for #19, cross validation
    X, y = get_data('hw4_train.dat')
    Ecv_optimal = 1
    lambda_Ecv_optimal = 0
    for i in range(2, -11, -1):
        Ecv = Cross_Validation(X, y, pow(10, i), 5)
        if Ecv < Ecv_optimal:
            Ecv_optimal = Ecv
            lambda_Ecv_optimal = i
    print('Optimal Ecv and log10_lambda are :', Ecv_optimal, lambda_Ecv_optimal)

    #Solution for #20
    X_train, y_train = get_data('hw4_train.dat')
    X_test, y_test = get_data('hw4_test.dat')
    w_20 = Ridge_Weight(X_train, y_train, pow(10, lambda_Ecv_optimal))
    Ein = Calculate_Error(X_train, y_train, w_20)
    Eout = Calculate_Error(X_test, y_test, w_20)
    print('Ein and Eout based on the log10_lambda -8 :', Ein, Eout)

