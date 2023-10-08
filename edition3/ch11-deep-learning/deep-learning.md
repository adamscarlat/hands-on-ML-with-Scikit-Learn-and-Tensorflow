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
  - This causes the neuron to "die" since it no longer propagates signal through the network. ReLU will output 0
    if its inputs are negative, same for the ReLU derivative.

* Leaky ReLU
  - A variant of ReLU - `max(a * z, z)` where `a` is a constant which defines the slope when z < 0.
    * Usually a value in the range 0.001 - 0.2.
    * Notice that if z < 0 and a < 1, the function `max(a * z, z)` will always choose `a * z` and if z > 0, the function
      will choose z.
  - The leak in leaky ReLU prevents the ReLU function from "dying" and makes sure that it always outputs a value 
    other than 0. 

* Parametric Leaky ReLU (PReLU)
  - Same as leaky ReLU only that `a` is not a hyperparameter but a parameter learned during training.
  - PReLU was observed to outperform leaky ReLU on large image datasets but also noticed to overfit on smaller 
    datasets.

* ReLU, Leaky ReLU and PReLU all suffer from the same problem - the gradient is not smooth at 0. It jumps and that makes
  convergence of the network to slow down (the gradient will bounce around the optimum value in some cases).

* elu - exponential linear unit
  - This is an activation function that takes ReLU (and its variants) and corrects the discontinuity of the function 
    around 0.
  - It converges better than ReLU and its variants but is more expensive to compute due to the exponential function:
  ```js
  elu(z) = {
    z             if z >= 0,
    a * (e^z - 1) if z < 0
  }
  ```

Batch Normalization
-------------------
* The methods we looked at so far (activation functions and better weight initialization) work well for reducing 
  the vanishing/exploding gradients in the beginning of the training, but they can't guarantee that the
  problem won't return in later stages of the training.

* Batch normalization works by taking the result of each layer (before or after the activation function) and 
  standardizing it such that: 
  - It has a mean of 0
  - It then scales and shifts the results using parameters that it learns. 
  - In other words, it learns the scale and mean of each layer's inputs.

* How does batch normalization work?
  - For each mini batch, for each layer, it computes:
    * The mean of every input in the mini-batch:
    ```js
    u_b = (1/m_b) * ∑ x_i for i = 1 to m_b
    ```
      - m_b is the number of instances in the mini-batch
      - u is a vector of means for each input (each input can be seen as a vector, the mean is of each of these vectors).
        It contains one mean per feature in the mini-batch.
    * The standard deviation of each input in the mini-batch:
    ```js
    std_b = (1/m_b) * ∑ (x_i - u_b)^2
    ```
      - std_b is the vector of standard deviations. It contains one std per feature in the mini-batch
    * Then the zero-mean inputs:
    ```js
    x_hat_i = (x_i - u_b) / sqrt(std_b^2 - eps)
    ```
      - x_hat_i is the vector of zero-centered and normalized inputs for instance i.
    * Then the rescaled and shifted inputs:
    ```js
    z_i = y * x_hat_i + b
    ```
      - y is the learned scale
      - b is the learned shift
      - z_i is the output of the BN operation. It is a rescaled and shifted version of the inputs.

* For example, assume we enter layer l1 with the following mini-batch of 3 samples with 3 features each
  ```js
  [
    [1,2,3],
    [4,5,6],
    [4,8,9]
  ]
  ```
  - The mean and std per feature are computed:
  ```js
  u_b = [(1+4+4)/3, (2+5+8)/3, (3+6+9)/3] = [3,5,6]
  std_b = [(1-3)^2 + (4-3)^2 + (4-3)^2, (2-5)^2 + (5-5)^2 + (8-5)^2, (3-6)^2 + (6-6)^2 + (9-6)^2]
        = [6/3,18/3,18/3] = [2,6,6]
  
  // Normalize each input feature using mean and std (not shown...). After this the normalized batch has a 
  // mean of zero and std of 1. 

  // Scale and shift each input feature (not shown...). In reality, not all data follows a normal distribution.
  // If we let the network scale and shift the batch as needed, it'll learn the correct distribution of the 
  // data. In other words, we leave flexibility using learned parameters to learn the data properly
  ```

* What about batch normalization during inference?
  - During inference we probably won't have enough instances to compute all those values. Using less instances
    will make these values stochastic.
  - To solve this, the BN algorithm uses the values learned during training for new examples during inference.
  
* Batch normalization helps in MANY areas of DNN:
  - It solves the vanishing/exploding gradient problem throughout training.
  - It solves the need to use sophisticated weight initialization techniques (for vanishing gradients)
  - It even allows using the tanh and sigmoid activations without vanishing gradients
  - It speeds up training (convergence on an optimal solution)
  - It adds regularization and reduces overfitting
  - If added to the first input layer, it reduces the need to normalize inputs

* Batch normalization makes each epoch longer (since there more computations) but you'd need less epochs.
  - All in all, it's still faster with BN than without.






