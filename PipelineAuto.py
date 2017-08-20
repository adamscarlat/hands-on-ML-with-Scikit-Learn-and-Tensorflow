import numpy as np
import DataHandler as handler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer #for missing values
from sklearn.preprocessing import LabelBinarizer #combined LabelEncoder and OneHotEncoder
from customTransformers.CombinedAttributeAdder import CombinedAttributesAdder
from customTransformers.DataFrameSelector import DataFrameSelector
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from validation import Validator
from sklearn.ensemble import RandomForestRegressor

'''
Using the Scikit learn Pipeline class to preprocess the data and train a model.
part 1 - preprocessing
part 2 - training different models
part 3 - measuring accuracy
part 4 - fine tuning chosen models
'''

'''part 1 - preprocessing'''
#fetch data and load as data frame
handler.fetch_data(handler.HOUSING_URL, handler.HOUSING_PATH, 'housing.tgz')
housing = handler.load_data(handler.HOUSING_PATH, 'housing.csv')

#stratified split by median income
train_set, test_set = handler.stratified_split(housing, 'median_income', 5)

#seperate labels from data
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

#seperate numerical data and textual data
#remove text column from data set  
housing_num = housing.drop("ocean_proximity", axis=1) 

#prepare numerical/text data attributes
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

#pipeline for numerical attributes
num_pipeline = Pipeline([
             ('selector', DataFrameSelector(num_attribs)),
             ('imputer', Imputer(strategy="median")),
             ('attribs_adder', CombinedAttributesAdder()),
             ('std_scaler', StandardScaler()),
])

#pipeline for textual attributes
cat_pipeline = Pipeline([
             ('selector', DataFrameSelector(cat_attribs)),
             ('label_binarizer', LabelBinarizer()),
])

#combining the two pipelines. each will run in parallel
full_pipeline = FeatureUnion(transformer_list=[
             ("num_pipeline", num_pipeline),
             ("cat_pipeline", cat_pipeline),
])

#run pipeline
housing_prepared = full_pipeline.fit_transform(housing)


'''part 2 - training different models'''
#linear regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#tree regression - a more complex model
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

#random forest - more complex model, can avoid overfitting
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

'''part 3 - measuring performance'''
#training error linear regression - RMSE - underfitting, both training and validation errors are high
housing_predictions = lin_reg.predict(housing_prepared)
print ('\nRMSE linear regression: ', Validator.get_RMSE(housing_labels, housing_predictions))

#k-fold CV error for linear regression
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print ('CV scores for linear regression')
Validator.display_cross_validation_scores(lin_rmse_scores)

#training error tree regression - RMSE - overfitting, training error low, validation error high
housing_predictions = tree_reg.predict(housing_prepared)
print ('\nRMSE tree regressor: ', Validator.get_RMSE(housing_labels, housing_predictions))

#k-fold CV error for tree regressor
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print ('CV scores for tree regressor')
Validator.display_cross_validation_scores(tree_rmse_scores)

#training error forest regression - RMSE - less overfitting, training error medium, validation error high
housing_predictions = forest_reg.predict(housing_prepared)
print ('\nRMSE forest regressor: ', Validator.get_RMSE(housing_labels, housing_predictions))

#k-fold CV error for tree regressor
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
print ('CV scores for forest regressor')
Validator.display_cross_validation_scores(forest_rmse_scores)

