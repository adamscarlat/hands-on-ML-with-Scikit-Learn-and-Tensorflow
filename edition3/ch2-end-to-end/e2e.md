End to end ML project
---------------------
* Problem - develop a house price prediction model. 
  - The result of this model is expected to be an input to another model, that with additional parameters, will
    determine if it's worth investing in a certain area.

Frame the problem
-----------------
* It's important to know how the company is expecting to use the model and as much information as possible. This will
  help determine:
  - Which performance metrics you'll use to evaluate the model
  - How much effort you'll use to tune the model

* Find the current baseline to the problem 
  - Is there a baseline that we can compare a trained model against?
  - For example, it could be a team of workers who make the prediction manually based on a set of rules. We would want to
    compare the model against the baseline.
  - A generally good baseline for regression problems are taking comparing the prediction against the mean.
  - Another generally good baseline for regression problems is to fit a simple linear regression model and compare
    against it.

* Define your problem
  - Supervised, multiple, univariate regression problem using batch learning
    * Supervised - data is labeled (house prices)
    * Multiple regression - we're predicting based on multiple features
    * Univariate regression - we're predicting a single value, the house price

* Set the performance measure
  - RMSE (aka L2 norm)
    * (1/M) * sum(h(x(i) - y(i))^2)
  - MAE (aka L1 norm)
    * (1/M) * sum(h(x(i) - y(i)))
    * Better if there are many outliers in the data
  - The higher the norm value, the more sensitive it is to outliers and large values.


Get the data
------------
* Next step is to get the data, preferably into a pandas data frame.

* Explore the data by looking at the top few rows
  - `data_frame.head()`

* Check how many features have missing/null values and the types of values
  - use the `data_frame.info()` method 

* For categorical types, get all the unique values

* Visualize the data distributions. We want to look for:
  - Skewed feature distributions
  - Capped features (e.g median income is capped at 150K. Anything over it is capped as 150K)
  - Different scales of data (e.g number of rooms 1-5 and square feet 100-3000)


* Split the data to train/val/test sets
  - It's important that the same instances remain in the test in subsequent training of the model
  - Therefore, think of a way such that data is split across train/test sets the same every time. 
    One such approach is to take the hash of each instance and if it's lower than a certain threshold,
    add it to the test.

* Stratification and bias
  - If we split the data without considering data stratas, we're risking introducing bias to the training.
    * The distribution of the train/test sets (the samples) with regards to certain features or even the label itself
      can be different than the distribution of the overall dataset.
    * Stratification helps maintain the same data distribution of the overall dataset in your sampled dataset.
  - For example, the house prices dataset contains a median_income column. 
    * This column has salaries in the range 15K-150K. The distribution of these salaries is not normal
    * If we do a train/test split over the data, we're running the risk that we sample more instances
      of certain income than others. For example, our training set can contain almost all salaries between
      15K-60K and none that are over 120K. 
    * The training model is biased towards the lower salaries now
  - To do a split that takes stratas of a continuous value into consideration, we put the values in bins
    [0,1.5,3,4.5,6,np.inf]
    * Then we can use scikit learn's stratified split

Scikit-Learn's API
------------------
* Estimators 
  - Any object that can estimate some parameters based on a dataset
  - Estimators have a `fit` function which takes as input:
    * The dataset
    * Labels (in the case of estimators for supervised learning)
    * Hyperparameters
  - Hyperparameters are always accessible on the object
  - Learned parameters are accessible as well, they are suffixed with an underscore
  - Example - `SimpleImputer`
    * Given a dataset and a strategy (e.g median, a hyperparameter), it'll fit (compute in this case) the median
      of all columns. These are the parameters of this estimator (`imputer.statistics_` will output them).

* Transformers
  - Some estimators can transform a dataset
  - Transformers rely on the parameters learned during `fit`
  - Transformers `transform` function takes in a dataset and returns the transformed dataset
  - Example - `SimpleImputer`
    * It'll transform a given dataset using the imputation values it learned during fit. In the case
      of `SimpleImputer`, it'll fill null values.

* Predictors
  - Some estimators, given a dataset, are capable of making predictions.
  - Predictors `predict` function takes a dataset and returns a dataset of predictions
  - They also have a `score` function which measures the quality of the predictions, given a test set
  - Example, `LinearRegression` model is an estimator that is also a predictor

