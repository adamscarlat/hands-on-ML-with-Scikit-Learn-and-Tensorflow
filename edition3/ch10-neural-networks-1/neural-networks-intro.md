Introduction to Neural Networks
-------------------------------
* NN are getting popular again after a long winter (since the 1960's). This mostly because:
  - Data
    * There are huge quantities of data available and NN outperform other models on large 
      datasets.
  - Computing power
    * Advancement of computing power such as GPUs allow running these NNs on larger datasets.
    * Cloud computing made this computing power available
  - Algorithms
    * Algorithms are getting better and better
  - Investments
    * More money is dedicated towards research of NNs and applications using it.
    * For example, GPUs dedicated to matrix operations which are the bread and butter of NNs

The Perceptron
--------------
* TLU - threshold logical unit
  - Takes in inputs x1,x2,...,xn
  - Computes a linear transformation over the inputs:
    ```js
    z = w @ x + b = w1*x1 + w2*x2 + ... + wn*xn + b
    ```
  - Runs a step function over z:
    ```js
    result = h(z)
    ```
  - The step function can be one of many non-linear functions. For example, taking z and returning
    +1 if z >= 0 and -1 otherwise.

* Using a single TLU, we can start training models for binary classification.

* A perceptron is a single layer of multiple TLUs. These TLUs are usually connected to all inputs.
  - This is also known as a fully-connected layer or a dense layer.
  - With multiple TLUs, we can start building models for multi-class classification.

* Training a perceptron
  - During the forward pass, we compute the activation of all TLUs in the layer using the following function:
  ```js
  output = h(X @ W.T + b)
  ```
    * This is the same as the TLU equations above.
    * Notice that it can be done using the entire data set (not just a single example at a time). For example,
  ```js
  X -> (m,n)
  W -> (l,n)
  X @ W.T -> (m,n) @ (n,l) -> (m,l)
  ```
    * Where l is the number of output TLU units in the layer.
    * (m,l) - for each training instance, we get an output value 
    * Each one of the l outputs goes thru a step function
  - This output can represent the probability of choosing one of l classes for a classification problem.
  - Now a cost function is computed over the l outputs. The cost function checks the error using the correct labels and
    the weights are updated accordingly. Below is the update rule:
  ```js
  w_ij = w_ij + step * (yj - yj_hat) * xi
  ```
    * w_ij - is the weight connecting input i to output j
    * xi - the value of the input
    * yj - yj_hat - the cost (diff between prediction and real label)
    * step - learning step
  - 

* A single perceptron layer can only solve linearly separable datasets. If we stack multiple layers, we can start
  solving for non-linear datasets.

Training MLPs
-------------
* Training MLPs using gradient descent requires computing large gradients over multiple layers as each weight
  contributes to the error in some way.

* Computing these large gradients is done using a technique called `reverse mode auto-diff` which does so efficiently.

* The forward pass computes the predictions and the error and the backward pass (backpropagation), tweaks the weights
  according to the gradient.

* In order to do backpropagation we need to tweak a few things about the perceptron:
  - First we need a non-linear activation function
    * Without it, we're just chaining linear transformations which yield a linear transformation.
    * This means that we won't be able to separate non-linear data.
  - The activation function has to be differentiable at all points.
    * Otherwise, we can't use backpropagation which computes the gradient of the cost function with respect 
      to each weight.

Regression MLPs
---------------
* Can be used for:
  - Single value regression 
    * Single output neuron
  - Multi value regression
    * Multiple output neurons
    * For example, using 5 outputs it can be used to output the center of an image and a bounding box around it.

* MLP regressor allows for the creation of multilayered NNs but is limited in its configurable parameters.
  - For better NN architecture use Keras.


Classification MLPs
-------------------
* For binary classification, a single sigmoid output neuron is used.
  - The output is the probability of the positive class (1-value for the probability of the negative class).

* For multiclass tasks, multiple output neurons are used.
  - Their result is passed to a softmax layer which returns a probability for each output.

* For multilabel tasks, multiple independent neurons are used.
  - Each neuron has its own sigmoid activation function.
  - For example, classifying whether an email is ham or spam and if it's urgent or non-urgent.

* The loss function for classification function is cross-entropy.
  - For binary classification:
  ```js
  ce = −(1/N) * ∑ (y_i * log(p_i) + (1−y_i) * log(1−p_i)) for i=1 to N
  ```
  
  - For multiclass classification:
  ```js
  ce = -∑∑ y_ij * log(p_ij) for i=1 to N and j=1 to C
  ```
  * Inner sum is over the number of classes j=1 to C
  * Outer sum is over the number of instances i=1 to N
  * y_ij equals 1 if class j is the correct class for instance i, 0 otherwise
  * p_ij is the probability that instance i belongs to class j
  * Notice how this function works:
    - If the model predicted perfectly, an example will have p_ij = 1 and y_ij = 1 and the product will be 1
    - If the model outputted a false positive p_ij = 1 and y_ij = 0 and the product will be 0
      * False negative is also a product of 0
    - A correct answer results in a higher product value (the more confidence, the higher the product). A wrong answer
      returns a product of 0. The sum will increase if the model is correct and the end result is multiplied by -1.
    - In summary, correct answers reduce the loss to a negative result.

