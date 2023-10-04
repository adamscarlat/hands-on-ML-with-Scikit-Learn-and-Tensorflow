Keras
-----
* The official high-level API for Tensorflow 2.

* When working with Keras we first choose a model type. For example, a `Sequential` model is the simplest type of NN.
  - It represents a NN of layers that are stacked and connected sequentially.

* Then we start adding layers
  - First layer we add is `Input` which takes the input shape. In the case of an image, it can be a 28x28 pixels input.
  - Next layer is `Flatten` which only flattens the input into a 1D array.
  - Next layer is `Dense` which is fully connected to the inputs of the previous layer and has an activation function.
    * We add a few of these to increase the NN's complexity and degree of freedom to fit a function to the data.
  - Last layer is also a `Dense` layer, but it has a `softmax` activation function and 10 neurons, 1 for each class.
    * We use softmax since the classes are exclusive (multiclass task, not multilabel task).
  
* Example of breaking down the Keras parameters in a sequential model:
  ```python
    model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=[28, 28]),
            tf.keras.layers.Dense(300, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
    ])
  ```
  - Flatten to dense 1: (batch_size, 784) @ (784, 300) ->  (batch_size, 300)
    * Dense 1 has 784x300 = 235200 parameters (weights or connections).
    * The layer also adds 300 bias terms so the total # of params of the layer is 235500
  - Dense 1 to Dense 2: (batch_size, 300) x (300, 100) -> (batch_size, 100)
    * Total number of parameters (plus bias) is 30000 + 100 = 30100
  - Dense 2 to Dense 3: (batch_size, 100) x (100, 10) -> (batch_size, 10)
    * Here each item in the batch is a probability for one of the 10 classes.

* Input shape is not mandatory to specify. Keras will figure it out once it sees the first input.
  - However, we won't be able to see the full summary and info about the model.

Compiling and fitting the Keras Model
-------------------------------------
* When we compile the model, we define the loss function, optimizer and additional metrics to compute during training
  and evaluation:
  ```python
  model.compile(loss="sparse_categorical_crossentropy",
                optimizer="sgd",
                metrics=["accuracy"])
  ```
  - `sparse_categorical_crossentropy`
    * This loss function is designed for exclusive (multiclass, not multilabel) classes that are in index form.
      For example, [0,1,2,3,4,...]
    * If our labels were one-hot encoded vectors, we would have used `categorical_crossentropy`.
    * Binary classification can use `loss=sigmoid` with a final layer's activation that's `binary_crossentropy`.
    * Using the wrong loss function results in a shape error. For example, using `categorical_crossentropy` on an 
      indexed (not one-hot) y vector, results in the following error:
      ```python
      ValueError: Shapes (None, 1) and (None, 10) are incompatible
      ```
  - `sgd`
    * Stochastic gradient descent. Using backpropagation.

* Fitting the model:
  ```python
  history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
  ```  
  - `validation_data`
    * Keras will measure the loss and extra metrics on this set at the end of each epoch.
    * This can help us see if the model is underfitting/overfitting during training.
    * Keras can also do the validation split for you (no need to create separate validation set) using the 
      `validation_split` parameter which takes a value between 0 to 1.
  - For the training, keras will use a batch size of 32 by default. Each epoch it'll split the training set into
    `m / batch_size` batches.
    * Looking at the fashion MNIST training output, we see that each epoch has 1719 iterations
    *  55,000 / 32 = 1718.75 -> 1718 * 32 = 54,976 + 24 extra instances
  - For the fashion MNIST, we can see that: 
    * Training loss = 0.1366, Training accuracy = 0.9520
    * Validation loss = 0.3211, Validation accuracy = 0.8924
    * There is some overfitting

* If the labels were unbalanced, we can use `class_weights` when calling `fit` on the model:
  ```python
  y_example = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2]

  class_weights = compute_class_weight('balanced', classes=np.unique(y_example), y=y_example)
  class_weights_dict = {i:weight for i,weight in enumerate(class_weights)}
  # {0: 3.33, 1: 0.51, 2: 1.33}

  history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, class_weight=class_weight_dict)
  ```
  - Keras will use the class weight when computing the loss function. It'll multiply the class weight with the loss.
    This way, classes with smaller representations can make up for it with greater loss. 
    * Loss to a single minority instance is higher than to a majority instance so that parameters are updated as 
      if multiple instances passed thru the model.

* If you look at the results and notice that more training is required, you can simply call fit again.
  - Keras will continue where it left off, not from the beginning.

Keras Functional API
--------------------
* We've seen the Sequential NN architecture that Keras offers. Using the functional API we can build NNs with more
  complex architectures.

* One such example is the `wide and deep` network architecture. In it, the input layer is connected to the sequential
  network (hidden layer 1 then 2, etc...) as we've seen before.
  - It also connects the input layer directly to the output layer.
  - This allows the network to learn non-linear complex patterns (the deep part) and also simple patterns (the direct
    part).
  - Simple patterns sometimes get diluted and distorted when passed thru the network.

* One of the examples is a NN with two inputs, one thru the deep path and another thru the wide path.
  - For the example, we split the data vertically - columns 0-4 go thru the wide portion and columns 2-7 thru the deep portion.

* Another common NN architecture is multioutput. Using it we can solve tasks such as:
  - Locating an object on an image (regression) and classifying it.
  - Learning multiple independent tasks from the same data. For example, classifying a person, outputting if they are smiling
    and what they are wearing. 
    * This can perform better than a single network per task because the NN can learn features in the data that are 
      useful across tasks.




