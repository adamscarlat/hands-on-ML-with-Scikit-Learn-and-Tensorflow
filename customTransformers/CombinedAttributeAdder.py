from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


#existing column index in the dataset
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    '''
    feature mapper tailored to match scikit learn pipeline. it will add 3 new columns
    to the data sets. the new columns are combined existing columns 
    '''
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        '''
        any constructor parameter is considered a hyperparameter that scikit learn
        can automatically tune
        '''
        self.add_bedrooms_per_room = add_bedrooms_per_room 

    def fit(self, X, y=None):
        return self # nothing else to do 

    def transform(self, X, y=None):
        '''
        adds the new column combinations to the data set
        '''
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix] 
        population_per_household = X[:, population_ix] / X[:, household_ix] 
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        return np.c_[X, rooms_per_household, population_per_household]