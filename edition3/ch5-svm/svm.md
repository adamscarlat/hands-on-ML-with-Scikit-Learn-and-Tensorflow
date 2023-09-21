Support Vector Machines
-----------------------
* Model for classification, regression and novelty detection

* Great for small to medium datasets
  - hundreds to thousands
  - Does not scale well

* SVMs are sensitive to feature scales

Linear SVM Classification
-------------------------
* SVM is a large margin classifier - it attempts to find a boundary between classes that has the largest 
  margin between the classes.

* The support vectors are instances of each class located at the edge of the margin - they determine the boundary.

* SVMs margins can be:
  - Hard margin classifiers
    * The margin has to purely separate all classes (without single instances of one class polluting the other).
    * This can only be achieved if the instances are linearly separable.
    * Even if the instances are linearly separable, it can create very narrow and not robust margins.
  - Soft margin classifiers
    * A more relaxed approach to SVMs. Trying to keep the margin as large as possible and limiting margin 
      violations - instances inside the margin or on the wrong side of it.
  - SMVs have a hyperparameter (C in sklearn) which is inversely proportional to the size of the margin. For example,
    * C=1 means very large margin
      - An overfitting model will benefit from a lower C
    * C=100 means very narrow margin
      - An underfitting model will benefit from a higher C

Nonlinear SVM Classification
----------------------------
* Similarly to how we added polynomial features for the linear regression model, we can use the same technique here
  and use a straight line to separate the data.
  - Remember, adding polynomial features to the data changes the data's shape to become non-linear, possibly allowing
    a straight line to separate it.
  - See page 179 in book for an example

* Polynomial kernel
  - Adding polynomial features doesn't scale well for complex datasets. It cannot transform complex datasets
    without a high polynomial degree which slows down training and it also increases overfitting.
  - The `kernel trick` (explained in depth later) is a way to achieve the same result without the combinatorial
    explosion of features that polynomial features cause.

    
Under the Hood of SVMs
----------------------

Linear SVMs
-----------
* When making a prediction, a linear SVM computes the decision function which is a dot product of the training instances 
  and the parameter vector:
  ```js
  pv @ xi = p0*x0 + p1*x1 + ... + pn*xn
  ```
  - If the result is positive, the predicted class is the positive class and vise versa.

* For training, the optimization problem the SVM model is trying to solve is finding the set of parameters pv that 
  creates the widest margin with the least number of violations.
  - A violation is an instance of class 1 in the space of class 0 (or vise versa).

* The first goal of SVM is to increase the borders of the margin .We can see that to increase the borders of the margin, 
  we can reduce the parameters. To see why:
  - Assume we have a single feature and a single parameter w1
  - Say we define the borders of the margin to be -1 and 1
  - We can see that as w1 decreases, the border of the margin increases:
    * if w1 = 1 the points which satisfy the margin border to be -1 and 1 are x1=-1 and x2=1:
      ```js
      w1*x1 = 1*-1 = -1
      w1*x2 = 1*1 = 1
      ```
      - In this case the size of the margin is abs(-1 - 1) = 2
    * if w1=0.5 the points which satisfy the margin border to be -1 and 1 are x1=-2 and x2=2:
      ```js
      w1*x1 = 0.5*-2 = -1
      w1*x2 = 0.5*2 = 1
      ```
      - In this case the size of the margin is abs(-2 - 2) = 4
  - So we can see that our first optimization goal is to keep the parameter vector small. This will increase the size of the 
    margin.

* The second goal of SVM is to reduce margin violations - instances of class 1 in the space of class 0 and vise versa.
  - Continuing the example above (where we defined the margin borders to be -1 and 1), we need the decision function 
    to output over 1 for positive instances and less than -1 for negative instances.
  - We define a new variable t(i) where:
    * t(i) = -1 when y(i) = 0
    * t(i) = 1 when y(i) = 1
  - Now we can define a constraint to the first optimization (minimizing pv):
    ```js
    t(i)*(pv @ xi + b) >= 1 for i = 1,2, ..., m
    ```
    * This constraint makes negative examples generate a negative dot product and positive example a positive one.
    * It helps create a single weight vector that supports both negative and positive examples.

* Putting the optimization goal and the constraint together:
  ```js
  minimize norm(pv)
  subject to t(i)*(pv @ xi + b) >= 1 for i = 1,2, ..., m
  ```
  - The lower pv gets the greater the margin gets.
  - The constraint enforces that negative instances generate a negative dot product and vise versa. This ensures that 
    negative instances are on one side of the margin and positive instances are on the other side of it.
  - To see why, think of the dot product and bias term `pv @ xi + b` as a line. The optimization problem tries
    to find the line that satisfies the two constraints for all training examples. It can find more than one line
    and then it'll take the one that has the lowest pv

* The above optimization is for a hard margin classifier - there is no violation of the constraint - all positive
  examples and separated from negative ones.
  - To build a soft margin classifier where some violation is allowed, we introduce another constrained variable
    Ci for each training instance
  - This variable defines how much each instance is allowed to violate the margin.
  - The objective is to keep it as low as possible.
  ```js
  minimize norm(pv) + sum(Ci) for i = 1,2, ..., m
  subject to t(i)*(pv @ xi + b) >= 1 - Ci for i = 1,2, ..., m
  and to Ci >= 0 for i = 1,2, ..., m
  ```
  - Note how Ci allows for more degrees of freedom in the first constraint.


Non-Linear SVMs
---------------
* For non-linear SVMs we can add polynomial features to the data set. The issue with that is that it makes the
  data much more complex, prone to overfitting and expensive to store in memory.
  - Also, the higher the polynomial degree, the greater the combinatorial explosion of polynomial features.

* The kernel trick provides an alternative to polynomial features. It allows an implicit creation of polynomial 
  features without actually storing them in the dataset itself.

* First consider how taking the dot product of two high degree polynomial can be simplified:
  - Consider to vectors a,b:
  ```js
  a = (a1, a2)
  b = (b1, b2)
  ```
  - For each one of the vectors, we raise it to the second degree (what happens to a dataset when we add polynomial
    features to it is similar):
  ```js
  a_2 = (a1^2, sqrt(2)*a1*a2, a2^2)
  b_2 = (b1^2, sqrt(2)*b1*b2, b2^2)
  ```
  - Now we take the dot product of the two vectors:
  ```js
  a_2 @ b_2 = a1^2*b1^2 + 2*a1*b1*a2*b2 + a2^2*b2^2 = (a1b1 + a2*b2)^2 = (a @ b)^2
  ```
  - We learn that the square of the dot product of the original vectors a and b is equal to their polynomial version's
    dot product.

* The key insight here is this:
  - If we take the square of every pair of samples in the dataset - `(xi @ xj)^2` - we don't need to transform any 
    of the features.
  - This is much more computationally efficient and doesn't have the negative side-effects of adding polynomial features.

* How do we use it in a model?
  - We define a kernel function K, which can take two vectors and compute their higher degree polynomial dot product
    based on the original vectors, without having to compute or know about the polynomial transformation.
    * For example, a polynomial kernel takes two vectors and computes their polynomial transformed square as follows:
      ```js
      K(a,b) = (y * a @ b + r)^d
      ```
      - Where y and r are constants and d is the polynomial degree.
  - To use the kernel in model training and predictions, we include it in the dot product with the parameter vector.
    * Remember that we have to do the pairwise dot product for each training example for this to work.
    * To predict example x_n:
    ```js
    sum(K(x_n, xi) + b) for i = 1,2, ... , m
    ```
    - This equation is very much simplified but it illustrates the point - the kernel takes into account all pairwise
      examples.
