import DataHandler as handler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer #for missing values
from sklearn.preprocessing import LabelBinarizer #combined LabelEncoder and OneHotEncoder
from customTransformers.CombinedAttributeAdder import CombinedAttributesAdder
from customTransformers.DataFrameSelector import DataFrameSelector

'''
Using the Scikit learn Pipeline class to preprocess the data and train a model
'''

#fetch data and load as data frame
handler.fetch_data(handler.HOUSING_URL, handler.HOUSING_PATH, 'housing.tgz')
housing = handler.load_data(handler.HOUSING_PATH, 'housing.csv')

#stratified split by median income
train_set, test_set = handler.stratified_split(housing, 'median_income', 5)

#seperate labels from data
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

#seperate numerical data and textual data
#1. remove text column from data set  
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

housing_prepared = full_pipeline.fit_transform(housing)
print (housing_prepared.shape)


