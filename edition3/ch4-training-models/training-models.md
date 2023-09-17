Training Models
---------------

Linear Regression
-----------------
* A linear model makes a prediction by computing a weighted sum of input features plus a constant called the bias
  term:
  ```js
  y^ = p0 + p1*x1 + p2*x2 + ... + pn*xn
  ```
  - y^ is the predicted value
  - n is the number of features
  - xi is the ith feature value of a single training instance
  - pj is a model parameter

* Same equation in vectorized form:
  ```js
  yi^ = h(Xv) = Pv . Xi
  ```
  - Pv is a parameter vector containing [p0, p1, ... , pn]
  - Xi is a feature vector containing [x0, x1, ... , xn] of the ith training instance
  - Pv . Xi is the dot product of the two vectors

* The goal of model training is finding the best parameters Pv that minimize the error between ALL Y^ and Y.
  - We capture the error between Y^ and Y using a cost function 
  - The goal of the training is finding the parameter vector Pv that minimize the cost function
  - In other words, we have a single set of trainable parameters - P and we need to fit it to ALL training
    examples X

* Example for a cost function for a linear regression model - Mean Squared Error (MSE)
  ```js
  MSE(X, h) = (1/m) * sum(P . Xi - yi)^2
  ```
  - Xi is the feature vector of sample i
  - yi is the actual label of sample i
  - Notice that the function is averaging the squared errors over the entire dataset

Linear Regression - The Normal Equation
----------------------------------------
* This is a closed-form approach to find the set of P that minimizes MSE:
  P = (Xt . X)^-1 . (Xt . y)
  - Xt is the transpose of X
  - The shape of P at the end will equal the number of columns in X:
    * X.shape  ->    (m,n)
    * Xt.shape ->    (n,m)
    * y.shape  ->    (m,1)
    * [(n,m) . (m,n)] . [(n,m) . (m,1)] = (n,n) . (n,1) = (n,1)

* The normal equations will not work if the matrix X is not invertible (e.g m < n, less examples than features - wide matrix),
  or if some features are redundant (see matrix inversion rules).
  - Libraries use pseudoinverse of matrices which always exists

* Computational complexity
  - The (Xt . X) operation is a matrix multiplication with worst case of O(n^3) or  O(n^2.4) where n is the number of 
    columns (features).
  - The inverse of the matrix is done using SVD which is an O(n^2) operation
  - Notice that the complexity has a non-linear proportion to the number of features but is linearly proportional to the
    number of examples (rows). 
  - In practice, the normal equation works slowly when there are many features and requires that the entire dataset
    fits in memory.

Linear Regression - Gradient Descent
-------------------------------------
* This is a general approach to finding the optimal solutions to a wide range of optimization problems.

* The general ideas is to tweak the parameters iteratively in order to minimize the cost function.
  - The algorithm measures the local gradient of the cost function with regards to parameter vector Pv.
    * A partial derivative with respect to each pi in Pv is computed. In other words, we compute how each pi affects
      the cost function (that's the gradient).
    * Then we tweak the parameters Pv 
    * Essentially, the gradient is a vector that tells you the direction towards the minimum of the cost function
      with respect to Pv. Tweaking Pv with values of the gradient will create Pv values that move the function
      towards a minimum.

* 







  