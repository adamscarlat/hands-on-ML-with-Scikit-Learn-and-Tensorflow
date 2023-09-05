ML Intro
--------

Classifying ML Systems by type
------------------------------
* We can classify an ML system by these characteristics:
  - Learning supervision 
    *  Supervised, unsupervised, semi-supervised learning
  - Type of problem 
    - Classification, regression, clustering
  - Online vs batch training
  - Instance-Based Versus Model-Based Learning
    * Model based training is the known way of training an ML model (training it using data and making inferences 
      using a trained model).
    * Instance based training is a technique where the model doesn't really learn patterns in the data. Instead, 
      it stores a representation of the data (e.g. vectorized) in memory and when a new instance comes in, it compares it to the data and classifies it (or does regression) based on the instances it's the closest to.
      - KNN is an example for such model.

* Semi-supervised learning
  - Some of the data is labeled and some isn't
  - Usually the semi-supervised model is a combination of supervised and unsupervised models
  - For example, an online photo service (google photos) may cluster your photos in an unsupervised
    manner. It'll group photos of the same people together, but it won't be able to label them.
    Then you come in, label the person in each cluster. Now the model has labels and can proceed
    to assign names to the rest of the photos.


Problems with ML
----------------
* Data problems
  - Insufficient data
  - Biased data
  - Poor quality data
  - Irrelevant features
  - Data mismatch

* model problems
  - Overfitting
    * Reduce feature complexity
    * Add regularization
    * Get more training data
  - Underfitting
    * Introduce more features
    * Reduce regularization

* Data split
  - train / validation / test
  - train / validation / dev / test
    * Useful when you don't have much data to train on and choose to train on data from other sources that is similar
      to your data (but not quite the same). 
      - The risk here is data mismatch (e.g train flower identification using pictures from online sources. It may not
        match the pictures taken with your camera, which you don't have enough of to train the model).
    * You split the online data into train / validation sets. Once it performs well on both, you test it on the 
      dev / test sets which contain your labeled data. If it performs well on the first sets (online data) but not
      on your data, it can be a data mismatch issue. 
    * In summary, the train-dev set is used to validate your model on data types that it hasn't seen at all during
      training, which most likely came from a different source. The process for training and validating using a 
      train-dev set:
      - Train your model on the training set (online data)
      - Validate your model on the validation set (still online data)
        * If validation here is not great, take corrective actions for overfitting (regularization, feature engineering,
          etc.) and repeat
      - Validate your model on the train-dev set (your labeled data - the smaller dataset)
        * If validation here is not great, this is related to data mismatch. Take corrective actions for overfitting
          and for data augmentation and repeat.
      - Validate your model on the final test set (your labeled data)