Loading and processing Data with Tensorflow
--------------------------------------------
* `tf.data` is tensorflow's data management library it offers efficient loading of big data, reading multiple files in 
  parallel, shuffling, batching and more.
  - This API lets you handle datasets that don't fit in memory
  - It can handle CSV, binary (fixed-sized records) and files in TFRecord format (supports records of varying sizes)
  - It also has support to reading from SQL databases and other storage solutions.

* Keras also has a data preprocessing library which can be incorporated into the model, thereby reducing the risk
  of mismatch between training preprocessing and inference preprocessing.

Keras Preprocessing Layers
--------------------------
* Whatever preprocessing we do during training will have to be done exactly the same during inference 
  and testing.

* Keras offers preprocessing in the form of layers, so that you can add them directly into your NN.

Embedding
---------
* Embedding is a dense representation of higher-dimensional data such as a category or text.
  - For example, assume that you have a vocabulary of 10K words and you want to create a one
    hot vector of the sentence "the quick brown fox". It will be a very sparse vector.

* Embedded vectors are dense in the sense that they represent the information in a dense numerical vector
  with far less dimensions than the original form.

Tensorflow Datasets
-------------------
* Online bank of datasets in many categories

* Downloaded datasets (using the `tensorflow_datasets` library) are downloaded to `~/tensorflow_datasets`.
