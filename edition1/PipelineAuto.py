import numpy as np
import DataHandler as handler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer #for missing values
from sklearn.preprocessing import LabelBinarizer #combined LabelEncoder and OneHotEncoder
from customTransformers.CombinedAttributeAdder import CombinedAttributesAdder
from customTransformers.DataFrameSelector import DataFrameSelector
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.model_selection import cross_val_score
from validation import Validator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

'''
Using the Scikit learn Pipeline class to preprocess the data and train a model.
part 1 - preprocessing
part 2 - training different models
part 3 - measuring accuracy
part 4 - fine tuning chosen models
part 5 - evaluate best model on test set
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

#SVM regressor - linear kernel
svm_lin_reg = svm.SVR(kernel='linear')
svm_lin_reg.fit(housing_prepared, housing_labels)

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

#training error linear kernel SVR - RMSE
housing_predictions = svm_lin_reg.predict(housing_prepared)
print ('\nRMSE SVR: ', Validator.get_RMSE(housing_labels, housing_predictions))

# #k-fold CV error linear kernel SVR
# scores = cross_val_score(svm_lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# svr_rmse_scores = np.sqrt(-scores)
# print ('CV scores for linear kernel SVR: ')
# Validator.display_cross_validation_scores(svr_rmse_scores)

'''part 4 - fine tune selected models'''

#grid search on random forest regressor
#grid search will try every combination of parameters in each dictionary
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

#train model with grid search and cv. good for small set of hyper parameters
print ('\nGrid search fitting for random forest regressor. May take some time...')
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

print ('\nRandom forest regressor grid search best params: ')
print (grid_search.best_params_)
print('\nRandom forest regressor grid search results: ')
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

#see which importance level of each feature
print ('\nFeature importance: ')
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"] 
attributes = num_attribs + extra_attribs
print (sorted(zip(feature_importances, attributes), reverse=True))


'''part 5 - evaluate best model on test set'''

#get the best model from grid search
final_model = grid_search.best_estimator_

#prepare test set
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

#predict and compute RMSE on test set
final_predictions = final_model.predict(X_test_prepared)
final_rmse = Validator.get_RMSE(y_test, final_predictions)

print ('\nRMSE on test set: ', final_rmse)


'''Example - full pipeline'''

print ('\nFull pipeline: ')
#adding the ML model to the pipeline
predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('svm_reg', RandomForestRegressor(**grid_search.best_params_))
])
predict_pipeline.fit(housing, housing_labels)

#predict using full pipeline
predictions = predict_pipeline.predict(housing)
print ('\nRMSE tree regressor (full pipeline): ', Validator.get_RMSE(housing_labels, predictions))


