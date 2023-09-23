Decision Trees
--------------
* Good for regression, classification and multioutput tasks.

* Not sensitive to feature scales

* White box model
  - Easy to understand why it makes its predictions. It's very interpretable (unlike other models
    such as a neural network).

* Non-parametric model
  - Unlike models which have a set number of parameters that they have to fit to the dataset, decision trees
    define their shape according to the data and don't have any limitation of set parameters.
  - This means that left unregulated, this model will surely overfit the data.

Making Predictions 
------------------
* Before seeing how decision trees models are trained, we'll see how predictions are done on a trained decision tree.
  - See the Graphviz example in the associated notebook

* Assume we trained a decision tree model on the iris dataset with two features:
  - petal length (cm) 
  - petal width (cm)

* The model builds a tree data structure which has:
  - A root node
  - Split nodes
  - Leaf nodes

* To make predictions, each node that's not a leaf node presents a T/F decision based on one of the data's features.
  - For the iris dataset, the root node's decision is `petal length < 2.45cm ?`
    * If true go left, right otherwise.
  - If we move left we get to a leaf node. This leaf node represents one of the classes that we're trying to predict.
  - If we move right, we get to a split node that asks `petal width  < 1.75 cm?` 
    * Here each decision will get us to a leaf node representing the other two classes.

* Each node in the tree has a count for how many instances of each class it contains (has underneath it).
  - Even though a leaf node is supposed to represent a single class, the model tries to find the optimal 
    fitting which may contain some impurities.
  - It's possible to build a tree that can have pure leaf nodes but it's likely to overfit.

* To measure a node's impurity, we can use a score called the gini impurity score.
  - If the node contains instances only from a single class, it's gini impurity score is 0
  - The gini impurity equation for the ith node:
  ```js
  Gi = 1 - sum(Pi,k)^2 for k = 1, ..., n
  ```
  - Where n is the number of classes
  - Pi,k is the ratio of class k instances among all training instances in the ith node.
  - For example, 
    * Assume 3 classes and 150 instance.
    * A node has 50 instances, 44 of class 1, 6 class 2, 0 class 3:
    ```js
    Gi = 1 - (44/50)^2 + (6/50)^2 + 0 = 0.24
    ```

* This score can be used as the probability (or confidence) of the prediction.
  - Note that all predictions that end up on that leaf will have the same probability.

Training a Decision Tree Model
------------------------------

The CART Training Algorithm
---------------------------
* CART - Classification and Regression Tree

* During training the model iteratively searches for pairs (k,tk) where
  k - feature
  tk - threshold for that feature

* It searches for pairs (k,tk) that split the tree and minimize the impurity of the subtrees it produces.
  - Once it split the tree in two in proceeds to split the subtrees again until it reaches pure leafs (could
    even be leafs of one item) or that it reached the defined max_depth parameter.

* The process (in detail)
  - Starting at the root node: 
    * for each feature k, try different values tk and measure the impurity for each (k,tk) pair.
    * Choose the (k,tk) pair that reduces impurity; creates the best separation.
    * Split the tree according to that pair. Now you have a decision node - for example xi, is feature k above tk?
      if yes go left, else right.
  - Recurse to beginning and repeat process until:
    * All leaf nodes are pure
    * Or, max_depth has been reached (or any of the other hyperparameters related to stopping the training)


Regularization Hyperparameters
------------------------------
* Decision trees are non-parametric models and are prone to overfit (see explanation at the beginning).

* To restrict the freedom of training of a decision tree, we can use the following hyperparameters:
  - `max_depth`
  - `max_features`
    * Maximum number of features to check in each split
  - `max_leaf_nodes`
  - `min_samples_split`
    * Minimum number of samples a node must have before it can split
    * For example, if it's set to 10, a node with less than 10 samples cannot split
  - etc...

* Increasing the `min_*` parameters and reducing the `max_*` parameters regularizes the model

Regression Using Decision Trees
--------------------------------
* Works similarly to classification training and prediction only that instead of a class at the leaf node,
  we have a value and in each node we measure a regression metric such as MSE.

* In each node of the tree, we have the average prediction value over all samples in that node and the MSE.
  - For example, 
    * We have a tree in which the leaf node contains 110 samples
    * We get the y value for each one of these samples (each sample is labeled with a y value in the dataset since
      it's a regression problem).
    * We compute the average of these values and that's the node's value
    * To get the MSE for the node, we take that average value - `y_hat` and use it to compute the MSE at the node:
      ```js
      MSE_node = (sum(y_hat - y_i^2) / m_node) for i in 1,2, ..., m_node
      ```
      - We're essentially computing the distance between the average y at that node to each actual y at the node and 
        taking the average of that distance.
      - In other words, the MSE at the node checks the average distance between y_hat to all y at the node.
    * Similar to the classification training of a tree, the algorithm will choose a pair (k,tk) which minimizes
      the MSE at the node.

Sensitivity to axis orientation (trees + PCA = <3)
-------------------------------
* Notice that in all the plots of tree predictions we plotted, the decision boundaries are either vertical or 
  horizontal.

* It turns out that decision trees work better with data that can be separated that way, as opposed to data
  that requires a diagonal line or a curve.

* One way to get the data to be shaped in such way is to reduce its dimensionality using PCA

