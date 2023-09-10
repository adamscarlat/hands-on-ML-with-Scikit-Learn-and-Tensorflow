import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from pandas import DataFrame
from sklearn.calibration import LabelEncoder
from sklearn.pipeline import Pipeline

def apply_label_encoding(X: DataFrame):
  transformed_df = DataFrame()
  for column_num in range(X.shape[1]):
      column = X[:, column_num]
      transformed_df[column_num] = LabelEncoder().fit_transform(column)

  return transformed_df

model: Pipeline = joblib.load("heart_disease_model.pkl")

data = pd.read_csv("./data/heart.csv")
data_no_label = data.drop("HeartDisease", axis=1)
labels = data["HeartDisease"]

n = len(data)
data_no_label_5_samples = data_no_label[:n]
labels_n_samples = list(labels[:n])

predictions = list(model.predict(data_no_label_5_samples))

print (accuracy_score(labels_n_samples, predictions))

# print ("Predictions: ", predictions)
# print ("Actuals:     ", labels_n_samples)

