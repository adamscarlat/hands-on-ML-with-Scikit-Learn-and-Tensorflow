import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

'''
Example for loading a pickled model and using it to make predictions over "new" data points.
'''


class ClusterSimilarity(BaseEstimator, TransformerMixin):
  def __init__(self, n_cluster=10, gamma=1.0, random_state=None) -> None:
    super().__init__()

    self.n_cluster = n_cluster
    self.gamma = gamma
    self.random_state = random_state

  def fit(self, X, y=None, sample_weight=None):
    self.kmeans_ = KMeans(self.n_cluster, random_state=self.random_state)
    self.kmeans_.fit(X, sample_weight=sample_weight)

    return self
  
  def transform(self, X):
    return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
  
  def get_feature_names_out(self, names=None):
    return [f"Cluster {i} similarity" for i in range(self.n_cluster)]
  
# Takes a numpy array and does a ratio of the first and second columns
def column_ratio(X):
  return X[:, [0]] / X[:, [1]]

# Returns a name. Used in the FunctionTransformer. The name will get appended to the ColumnTransformer
# column name
def ratio_name(function_transformer, feature_names_in):
  return ["ratio"]

model = joblib.load("ch2-end-to-end/california_housing_model.pkl")

# Assume these were sent from a source that required predictions
data = pd.read_csv("ch2-end-to-end/datasets/housing/housing.csv")
data_no_labels = data.drop("median_house_value", axis=1)
example_data_points = data_no_labels.iloc[:5]

predictions = model.predict(example_data_points)
print (predictions)