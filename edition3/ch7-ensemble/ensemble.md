Ensemble Learning
-----------------
* Ensemble learning
  - Training a bunch of different models on the same data (or on subsets of the data).
  - When inference is needed, get a prediction from all of them and (in the case of a voting ensemble):
    * For classification tasks, take the most frequent answer
    * For regression tasks, take the average answer

* The `wisdom of the crowds` produces a better answer in most cases.

* Usually you will train a few good models and at the end of the pipeline you'll combine them into
  an ensemble model to increase performance.

* There are different types of ensemble methods. i.e. different ways to train and obtain a single answer from a bunch
  of models.

Voting Classifiers
------------------
* When inference is needed, get a prediction from all of them and:
  - For classification tasks, take the most frequent answer
  - For regression tasks, take the average answer

* This is called `hard voting`

* It's shown that even if your classifiers are `weak learners`, meaning they do just a little better than random 
  guessing, taking a vote from many of them can yield a result of a `strong learner`, given that you have enough
  and that they are sufficiently diverse.

* What does sufficiently diverse classifiers mean?
  - Since we'll most likely be training our classifiers on the same data, they will likely make the same 
    mistakes, increasing the count of the wrong class in the vote.
  - This will lower the ensemble's accuracy.
  - Since we have to train the classifiers on the same data, two ways of increasing independence of the predictions:
    * Train the classifiers on subsets of the data
    * Use different models in the ensemble

* Soft voting
  - If all models in your ensemble can return a probability (e.g. they have `predict_proba()`), you can use a soft
    voting method in your ensemble.
  - In soft voting, instead of taking the most frequent prediction of all classifiers for a given instance, we 
    compute the average probability and choose the highest one.
    * For example, assume your task is classification over 3 classes. You're using 5 classifiers.
    * After predictions over a test set, each one of the 5 classifiers returns an array of 3 probabilities for each 
      instance (assume inference over one instance for simplicity):
      clf1 - [0.2, 0.2, 0.6]
      clf2 - [0.1, 0.1, 0.8]
      ...
      clf5 - [0.3, 0.1, 0.6]
    * Then we take the average probability for each class for each classifier - [0.1, 0.1, 0.8] and choose the highest
      one.
  - Soft voting is considered more accurate than hard voting since it gives weight to each prediction.

Bagging and Pasting
--------------------
* As said above, ensemble methods will work better if there is greater variations between the models in the ensemble.
  - Models that make different types of errors for example produce greater variance.

* Two ways to achieve greater variance in the ensemble is to train the models in it on different subsets of the data.
  There are two ways to do that:
  - Bagging
    * Sampling is performed with replacement for the same predictor
  - Pasting
    * Sampling without replacement for the same predictor

* In both cases, instances can be sampled multiple times for different predictors.

* Bagging and pasting allow training and predictions to scale well since training and inference can run on parallel servers
  - The models in the ensemble are trained independently and they do inference independently.

* When using these technique, each model's bias is high (underfitting) since the dataset they are trained on is smaller.
  - The variance of the ensemble is lower (less overfitting)


Out of bag evaluation
---------------------
* When using bagging, each predictor gets to train on a subset of the dataset. 

* This creates a validation set for each predictor in the ensemble that we can use to evaluate the model.

Feature sampling
----------------
* Same way as we sample data to train each individual model in the ensemble to increase variability, we can also 
  do the same with the features. 
  - Instead of having all models use all features, we sample features and each model trains on a different set
    of features.

* This technique is useful in datasets that are very feature rich.

* When using sklearn's `RandomForestClassifier` (or regressor), you can control the feature sampling with the
  `max_features` parameter.

* `ExtraTreeClassifier`
  - This is an extension of `RandomForestClassifier` that allows for yet another bagging optimization for higher
    variability. 
  - It lets you set a threshold for each tree's feature selection. Instead of the trees finding the best feature
    threshold at each iteration, they stop at a set threshold (defined by the model).
  - This increases variability but also reduces training time since finding the best threshold is the longest 
    task in training decision trees.

* Random forests are handy to get a quick understanding of what features actually matter.


Boosting
--------
* The idea behind boosting is an ensemble where weak learners are trained sequentially and each learner 
  it trying to correct its predecessor's prediction.

Boosting - AdaBoost
-------------------
* The idea behind adaboost is that each model pays more attention to the misclassified instances from its
  predecessor.
  - First a model is trained and we collect the instances that it misclassified
  - Then a second model gets the training set with the misclassified instances getting more weight.
  - Process continues.

* With adaboost we consider the learning rate the factor by which we multiply the instances between learners.

* Once the all models in the boosted ensemble are trained, predictions are made in the same way as with bagging.
  - The major difference here is that each model's weights is different 

* Since this technique requires models to be trained in a sequence, it cannot be parallelized.

* To handle overfitting adaboost ensemble, reduce the number of learners of use regularization in each learner.

* AdaBoost in details:
  - After the first model is trained, we need to get an error rate. We iterate over the predictions and 
    for each prediction and add a weight for each prediction that's not equal the true value:
    ```js
    rj = sum(wi) for i = 1,2,...,m and when yi_hat != yi
    ``` 
    * These are not the model's parameters (pv)
    * Initially w is set to 1/m
    * The more misclassified instances the predictor has the higher rj is

  - We then compute the predictor's weight using r:
    ```js
    aj = rate * log((1-rj) / rj)
    ```
    * The more accurate predictor j is, the higher its weight aj becomes.
  
  - Once we have the weight aj for predictor j, we update the instances using it:
    ```js
    wi = wi         if yi_hat = yi
    wi = wi * e^aj  if yi_hat != yi
    ``` 
    * Remember that wi is the weight for instance i
  
  - Next step is to normalize the weights, we divide each weight wi by the sum(wi) over i=1 to m.
    * This makes the weights sum to 1.

  - Now we have a set of weights to be used by the next learner. There are different ways for these weights 
    to be used during training:
    * One way is to tweak the dataset by adding duplicates instances of examples with proportion to their weight.
      - The higher the weights of an instance, the more it was misclassified by learners. Duplicating it across
        the dataset will emphasize it more.
    * Another way, if using a tree based algorithm, is to have the algorithm focus on the features that would help
      split the misclassified instances first.

  - To make inferences, all predictors in the ensemble output a value
    * Then we see how many different classes are outputted in total - k
    * For each different class, we compute the sum of all weights over the predictors. The class with the highest 
      total weight gets chosen:
      ```js
      y_hat(xi) = argmax_k sum(aj) for j=1 to N and y_hat_j(x) = k
        where N is the number of predictors
      ```

Gradient Boosting
-----------------
* Similar to adaboost only that instead of adding weights to misclassified instances, this method tries to fit
  the new predictor to the `residual error` made by the previous predictor.

* Gradient boosting details:
  - First learner uses the original y to train:
  ```js
  learner1.fit(X,y)
  ```
  - Second learner uses the previous learners residual error to adjust y:
  ```js
  y2 = y - y_hat_1
  learner2.fit(X,y2)
  ```
  - Third learner uses the previous learners residual error to adjust y2:
  ```js
  y3 = y2 - y_hat_2
  learner2.fit(X,y3)
  ```
  - And so on...

* The main point is that learners learn on the error residual of previous learners. This reduces the values of 
  correctly classified/regressed values and increases the values of misclassified ones.

* It's like a bunch of students learning together efficiently and in sequence:
  - Student 1 focuses on what they can solve and then passes to student 2 the areas he was weak in.
  - Student 2 takes what student 1 couldn't solve, works on it and then passes to student 3 their work
  - and so on...

* To make an inference, we get a prediction from each model in the ensemble and add together their results.

* When using gradient boosting we have a few hyperparameters to use:
  - `learning_rate`
    * A multiplier to use in each learner's predictions.
    * The higher it is, the faster the algorithm will converge and less learners are needed.
  - `n_estimators`
    * Number of estimators in the ensemble
    * The higher it is, the lower we can set the learning rate. 

Stacking
--------
* This ensemble method starts the same as the voting ensemble, each trained model in the ensemble gets to 
  predict, but instead of voting or averaging the results, the predictions are inputted to a final model
  which outputs a value.

* The dataset that the final predictor receives has:
  - The same instances as the original training set (m rows)
  - Instead of the same features as the original training set, it has a prediction of each learner as a feature. 
  
