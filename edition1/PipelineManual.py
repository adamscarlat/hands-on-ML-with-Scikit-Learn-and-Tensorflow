import pandas as pd
import DataHandler as handler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from customTransformers.CombinedAttributeAdder import CombinedAttributesAdder

'''
manually building only the preprocessing steps. serves as an example. 
see PipelineAuto for a full pipeline
'''

#fetch data and load as data frame
handler.fetch_data(handler.HOUSING_URL, handler.HOUSING_PATH, 'housing.tgz')
housing = handler.load_data(handler.HOUSING_PATH, 'housing.csv')

#feature mapping - better correlation with the target value
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)

#stratified split by median income
train_set, test_set = handler.stratified_split(housing, 'median_income', 5)

#seperate labels from data
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

#some columns have missing values, 3 options:
#1. remove missing rows (dropna)
#2. remove whole column (drop)
#3. fill in with mean/median values of column (fillna, Imputer)
#imputer takes care of all the missing values
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1) #imputer only works on numerical values
imputer.fit(housing_num)
X = imputer.transform(housing_num) #fill in missing values with the column's median
housing_tr = pd.DataFrame(X, columns=housing_num.columns) #convert filled in training set back to pandas data frame

#encode the text column to numerical values
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)

#covert to one-hot-encoding to avoid proximity bias (e.g 2,3 closer than 2,5)
#result is a scipy sparse matrix where each row is a binary string (e.g. 2 -> 0, 1, 0, 0)
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))  

#see sklearn.preprocessing.LabelBinarizer for completing the above 2 steps at once

