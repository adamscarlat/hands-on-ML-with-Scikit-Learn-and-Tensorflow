import numpy as np
from sklearn.metrics import mean_squared_error

def get_RMSE(labels, predictions):
    '''
    returns the root mean squared error of a model. 
    input is the true values and the predicted values.
    '''
    mse = mean_squared_error(labels, predictions)
    return np.sqrt(mse)

def display_cross_validation_scores(cv_scores):
    '''
    print array of scores, mean and std for k-fold cross validation.
    '''
    print("Scores:", cv_scores)
    print("Mean:", cv_scores.mean())
    print("Standard deviation:", cv_scores.std())