Decision Trees
--------------
* Good for regression, classification and multioutput tasks.

* Not sensitive to feature scales

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
    

