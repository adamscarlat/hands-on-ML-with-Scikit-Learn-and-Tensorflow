from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin): 
    '''
    this selector takes a pandas DataFrame and returns a numpy array of the given columns. 
    can be used as a selector for a scikit learn Pipeline transformer
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names 
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        '''
        return a numpy array of given columns
        '''
        return X[self.attribute_names].values