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

* The learning rate is the most important parameter in the GD algorithm. It's the proportion of the step we take towards
  the minimum. 
  - If we set it too low, it can take a long time for the algorithm to converge.
  - If we set it too high, it can overshoot the minimum

* GD formula
  ```js
  Pv = Pv - step * grad(Pv)
  ```
  - grad(Pv) is the gradient of the cost function with respect to the parameter vector

* Some cost functions, such as MSE, are convex - meaning that if we draw a line between any 2 points on the curve, the line
  will never be below the curve.
  - In other words, there are no local minima and the function is shaped like a bowl.

* GD is sensitive to feature scale. When using it to train a model, scaling of features is very important.

* Training a model using GD is essentially searching for the set of parameters that minimizes the cost function.
  - The more parameters used, the greater the search space becomes.

* Each update of the weights with respect to the entire dataset is called an `epoch`. Usually we define the number of epochs
  we want the algorithm to run for.
  - Too many epochs and we can waste a lot of unnecessary time. Too little epochs and we may not converge.
  - A good solution is to set the epoch number to very high and set a `tolerance` value. This is a very small value that
    is checked at the end of every epoch against the norm of the theta gradient vector.
    * If the norm of the gradients is less than the tolerance we stop the algorithm.
    * It indicates that the change in slope is very small, so the function is flat at that point, hence a minima was reached.

Batch Gradient Descent
----------------------
* In this version of GD, we compute the gradient vector (amount of change for each parameter) with respect to the entire
  dataset and then update the weights.
  
* Batch gradient descent is less efficient than its other versions (below). The reasons are:
  - To compute the theta updates with respect to the entire dataset, the entire dataset has to fit in memory.
    if it doesn't, expensive modifications to the matrix multiplication must be made.
  - It's less amenable to parallelization than stochastic or mini batch GD

* One advantage batch GD has over stochastic or mini batch GD is that since it compute the weight updates with respect
  to the entire dataset, you know that you're moving in the direction of the true minimum.
  - With stochastic or mini batch GD, we compute the weight updates with respect to a sample of the dataset and when
    all weight updates for all mini-batches were computed, we take their average.
  - This is not the "true" direction of towards the minimum but an approximation of it. In practice it works quite well.


Stochastic Gradient Descent
---------------------------
* The opposite of batch GD, it computes the weight updates based on a single training instance picked at random
  from the dataset.
  - This makes it much faster than batch GD since it only needs to operate on a single instance to get weight
    updates rather than a whole dataset.
  - It also makes it less accurate in finding the true minimum since each instance by itself adds much noise 
    to the weight updates.

* Stochastic GD is better at avoiding local minima because of the noise added by each instance. The algorithm
  is less good in settling at a single point.
  - It's common to use a technique called `simulated annealing` when using stochastic GD. The idea is to reduce 
    the learning rate as the algorithm proceeds. This way you get the benefits of the added randomness (avoiding
    local minima) and adding the ability to converge.


Mini-Batch Gradient Descent
---------------------------
* This variation of GD falls between batch and stochastic GD - it updates the weights based on a small batch drawn
  randomly from the dataset.

* Using this approach we get the speed of a more stochastic GD with less of the noise.

* Remember, the reason why stochastic and mini-batch GD are faster than batch GD is because the weights
  get updated after each example (stochastic) or mini-batch.
  - With batch GD, the weights get updated after computing the gradient for the entire dataset.
  - Computing the gradient for a single example or a mini-batch is much faster.
  - The gradient gets updated more often (although not as accurately).


Polynomial Regression
---------------------
* Data that is non-linear can still use a linear regression model to fit the data if we tweak the features by
  adding non-linear ones.
  - We can add non-linear features by taking the powers of existing features

* The idea is - taking data that is non-linear (e.g you can't fit a straight line onto it) and adding to it 
  non-linear features (powers of the other features) so that a model can fit a curve to the data.
  - The model will have to use additional parameters for the non-linear features added. This will cause the 
    weights to be the coefficients of a polynomial.

* Assume that you have data where each point matches the following function:
  ```js
  y = X + X^2 + 2
  ```
  - To fit a curve to this function we can see that we need two parameters. By adding another column which is 
    a power of the first one, we'll naturally add another weight to the model. This will allow us to fit 
    this function.

* When adding polynomial features, the higher the degree of the polynomial, the more combinations of features that
  we're adding. For example, with degree=3 when there are 2 features (x1, x2):
  ```js
  y = b + x1 + x2 + x1^2 + x2^2 + x1*x2 + x1^2*x2 + x2^2*x1
  ```
  - This means that a polynomial regression model can help find relationships between features.
  - Notice that for degree=d and an array containing n features, the number of new features will equal:
  ```js
  n_features = (n+d)! / d!n!

  // for example d=4, n=3
  (4+3)! / 4!*3! = 35
  ```

Learning Curves
---------------
* The more polynomial features we add to the data, the higher the overfitting will get.

* Training curves help us find when a model is underfitting or overfitting. To get a learning curve: 
  - For each dataset (train, validation)
  - We plot the error during each iteration.

* A learning curve has:
  - Number of example as X-axis (increasing)
  - Cost function error as Y-axis
  - Two curves: training and validation errors as the dataset size increases

* The idea is that as we increase the dataset and for each new dataset size `we retrain the model`:
  - The training error increases as the model needs to account for patterns of the entire dataset
    * We start training with one example, then predict on that one example using the training set
  - The validation error decreases as the model can generalize better on new examples
    * Training a model on a single example is not enough for the model to start generalizing

* Analyzing a learning curve:
  - Training error that keeps increasing may point to a model that's underfitting 
  - The bigger the gap between the validation error to the train error the higher the overfitting of the model


Bias / Variance Tradeoff
------------------------
* Bias - the assumptions that the model is making from learning over the training set. This is a measure
  of underfitting of a model.

* Variance - the model's sensitivity to small variations in the data. Models with high variance are prone
  to overfitting (small variations make the curve overfit the data).

* The tradeoff:
  - Increasing a model's complexity, increases the variance (overfitting) and reduces the bias. Increasing 
    complexity of a model can be done by:
    * Adding more features
    * Reducing regularization
  - Reducing a model's complexity, reduces the variance and increases the bias. Reducing complexity can be done
    by:
    * Reducing feature number
    * Increasing training examples
    * Adding regularization


Regularized Linear Models
-------------------------
* Regularization works by reducing the degrees of freedom a model has to fit its parameters to the data.

Ridge Regression (l2)
---------------------
* We add another term to the cost function:
```js
a/m * |Pv|^2
```
- Where |Pv| is the l2 norm of the weights vector.
- a is a knob that increases or decreases the regularization. The higher it is, the less degrees of freedom
  that the model has to fit the data.

* Why does it work?
  - To see how this added term reduces the degrees of freedom of a model, consider that during backpropagation
    we're taking the partial derivative of the (now regularized) cost function with respect to each parameter
    in the weights vector.
  - When we take the partial derivative of weight pi, all other weights cancel out and we're left with a 
    constant pi to the gradient (not including the reg constant and other terms for simplification).
  - Since we're trying to minimize the cost function, this will naturally push the value of the parameters
    lower as now they are added again to the cost function.
  - Lower values for the parameters mean less degrees of freedom for the fit

Lasso Regression (l1)
---------------------
* Similar to ridge regression only - it adds a regularization term (l1 norm) to the cost function. The difference is that the 
  term is not squared:
  ```js
  a * |Pv|
  ```

* An important characteristics of l1 reg is that it eliminates weights of the least important features (sets them to zero),
  hence eliminating non-important features.

Elastic Net Regression
----------------------
* This is a reg term that's equal to a weighted sum of l1 and l2 regs.

* It's controlled by a term r
  - r is in the range [0,1]
  - When r=0, it's l2 reg
  - When r=1, it's l2 reg


When to use each regularization?
--------------------------------
* l1 (lasso)
  - If you have a high number of features and suspect that not all of them are useful

* l2 (ridge) or elastic net
  - If the number of features is small
  - If the number of features is greater than the number of training instances
  - When you have strongly correlated features

Early Stopping
--------------
* If we plot the error with respect to the epoch number, we'll see that at some point the error
  after the validation set error drops, it starts increasing again.
  - This is the point of overfitting. After this point, any additional training epochs hurt the model's
    ability to generalize.

* Early stopping is a technique to monitor the error over the validation set after every epoch and when
  it stops decreasing for a few epochs, it stops training.
  - It's important not to stop training right away when error stops decreasing when using mini-batch or SGD
    since the error is not monotonically decreasing using these algorithms.
  - You'd want to make sure that the error over the validation set is not reducing for multiple epochs before
    deciding to stop.


Logistic Regression
-------------------
* This is an application of a regression algorithm for classification problems. The model returns a probability
  that an instance belongs to a certain class.
  - E.g. output the probability that this email is spam.

* The model works very similarly to linear and polynomial regression - it computes a weighted sum of the input
  features and the weights, but instead of outputting the result directly like the lin reg models, it outputs
  the logistic of this result:
  ```js
  p = h(x) = logit(Pv . X)
  ```
  - logit is a function which outputs a number between 0 and 1:
  ```js
  logit(t) = 1 / (1 + exp(-t))
  ```
  - This makes p a probability. We use it as follows:
    * if p >= threshold -> y=1
    * if p < threshold -> y=0

* The cost function for logistic regression is derived from the above step function:
  ```js
  cost(Pv) = -(1/m) * sum(yi*log(pi) + (1-yi)*log(1-pi))
  ```
  - First notice that it's the average of errors over the entire dataset
  - Then notice that if yi=0, we're left with the second term: `(1-yi)*log(1-pi)` only
    * With this term only, the error is minimized (equals 0) when the log is taken over 1 (e.g log(1) = 0).
    * Hence pi=0 is a perfect prediction of the y=0 class.
  - Notice that if yi=1, we're left with the first term: `yi*log(pi)`
    * With this term only, the error is minimized when the log is taken over 1 again.
    * Hence pi=1 is a perfect prediction for the y=1 class.
  - All in all, notice that the purpose of this cost function is to train the model to output higher probabilities
    for positive classes and vise versa.


Logistic Regression and Decision Boundaries
-------------------------------------------
* Notice above that the logistic regression algorithm  works with a threshold to determine which class is predicted
  given an output of probability.

* If we increase that threshold, the model is less inclined to output positive and vise versa


Softmax Regression
------------------
* Notice that the logit function is built for a binary decision. We can use OvA technique
  on a dataset with multiple labels to extend it to multiple labels.

* Or we can use the softmax function to support multiple labels directly. The way it works:
  - Given an example x, we compute the score with respect to each class k.
    * Each class k has its own parameter vector Pk
    * For each parameter vector for each class k, we compute sk(x), a scalar value. Together we get a vector with the 
      score that the instance belongs to each class:
      ```js
      sk(x) = Pk . x
      ```
      - Each class k generates a scalar number sk(x) for instance x. 
      - Notice that the vector is not probabilities yet. We still need to normalize it and that's what the 
        softmax function does.
  - Once we have sk(x) for each class stored in a parameter matrix, we compute the probability that the instance
    belongs to each class using the softmax function:
    ```js
    pk = softmax(Sk(x))
    ```
  - The result is a vector with the probability that the instance belongs to each class k.

* During training, we update the vector Pk for each class using the following cost function:
  ```js
  J(P) = -(1/m) sum(sum(yki * log(pki)))
  ```
  - P is the matrix containing all vectors Pk.
  - Inner sum is over the number of classes k
  - Outer sum is over the number of training instances




  