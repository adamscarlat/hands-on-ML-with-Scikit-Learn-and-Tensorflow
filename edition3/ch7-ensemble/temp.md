1. Yes, using ensemble methods. For example, we could use a voting classifier which takes the most frequent
  result from the 5 models.

2. Hard voting - takes the most frequent value or the average of the predicted value in case of regression tasks
  Soft voting - gets the probabilities from each learner for each sample, then chooses the one with the highest
  probability across all learners (average). This way we gain more performance by using the probabilities as weights.

3. Scaling training across multiple servers as it pertains to different ensemble methods
  - bagging: yes, we can distribute training across servers as each learner uses a subset of the dataset
  - pasting: yes, same reason as bagging
  - boosting: no, training is sequential
  - random forests: yes, (assuming bagging/pasting is used)
  - stacking: yes, each learner uses a subset of or copy of the entire dataset

4. During bagging and pasting, out-of-bag evaluation takes advantage of the fact that each learner trains on a subset 
  of the dataset and that there are instances that are not used for training for each learner. OOB evaluation uses these
  to get a validation score for each learner. Essentially, testing the learner on instances it never saw.

5. Extra trees ensemble sample the features in addition to the data. Each learner only gets a subset of the features
  to train on. This increases the variability of the ensemble as each learner will make different types of errors
  on the same data. In addition, since we only use a subset of the features to train each learner, the training is faster.

6. If AdaBoost underfits the data, increase the number of estimators and reduce the learning rate.

7. If a gradient boosted model overfits the data, increasing the learning rate can help