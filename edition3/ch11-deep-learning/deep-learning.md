Training Deep Neural Networks
-----------------------------
* In NN-1 we saw some NNs with a few layers and neurons that were good for the example tasks.

* If we need to use a NN to classify hundreds of different labels from images, each with high definition (more
  pixels than 28x28), we'd need deeper networks with dozens of layers and hundreds of neurons.

* The issues with training such deep networks:
  - Vanishing gradients
  - Deep networks require more data (otherwise they overfit)
  - Training time gets longer
  - Overfitting is a real concern

Vanishing and Exploding Gradients Problems
------------------------------------------
* As the backpropagation algorithm starts computing the gradients going back from the outputs to the input layer,
  the gradients get smaller and smaller as it gets to the lower layers.
  - This causes the lower layers connection weights virtually unchanged, making the training to never converge.
  - In RNNs, the problem is exploding gradients, where the gradients become larger and larger to the point that 
    they make the training not converge.
  - In both situations, the gradients are unstable throughout the network and different layers learn at widely different
    speeds.

* The main reason for the vanishing gradient in a DNN, is that the variance of the inputs going into each layer is much
  lower than the variance of the outputs of that layer.
  - When the activations (output values) of neurons within a neural network layer have a greater variance, 
    it means that these activations are more spread out across a wider range of values. 
  - Conversely, when the activations have a lower variance, it means that they are more concentrated or confined to a 
    narrower range of values.
  - Now take an activation function such as sigmoid and its derivative sigmoid`, larger values produce values
    closer to 0 with sigmoid`. As we start computing the gradient of layers at the top of the network (first 
    layers to get their gradient computed in backpropagation), we get gradient values that are lower than 1.
  - Now take the chain rule which multiplies the derivative of previous layers (top layers) with the derivatives 
    of lower layers. Because we're multiplying values below 1, the gradient continues to shrink as the signal
    moves down the network.

Weights Initialization techniques - fighting the vanishing gradient
-------------------------------------------------------------------
* The idea is to reduce the variance between input and output layers across the network.

* One such approach is Xavier/Glorot Initialization
  - Here we start by sampling the weights from a normal distribution
  - Then we scale the weights by a factor of:
  ```js
  sqrt(2 / (n_in + n_out))
  ```
  n_in - number of input connection to the layer
  n_out - number of output connections from the layer
  - This initialization and scaling of the weights help reduce the variance between inputs and outputs across
    the network.


Changing the activation functions - fighting the vanishing gradient
-------------------------------------------------------------------
* Sigmoid is known to have gradients close to zero as their input increases:
  ```js
  sigmoid(x) = 1 / (1 + e^-x)
  sigmoid_der(x) = sigmoid(x) * (1 - sigmoid(x))

  sigmoid(3) = 0.95, sigmoid_der(3) = 0.045
  sigmoid(10) = 0.99, sigmoid_der(10) = 0.000045
  sigmoid(100) = 1, sigmoid_der(100) = 0
  ```

* ReLU helps with the vanishing gradient since it's gradient is 1 if the input is positive.
  - If we break down the chain rule computation throughout the backpropagation algorithm, we can see that in many
    places the 1 derivative helps propagating the signal to the lower layers.
  - ReLU has a different issue though, for negative inputs (negative dot product between the weights and the 
    input), it outputs zero. 
  - This causes the neuron to "die" since it no longer propagates signal through the network.

