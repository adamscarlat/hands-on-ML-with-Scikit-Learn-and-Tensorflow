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

Faster convergence during training
----------------------------------
* Below are methods to speed up training convergence. This is done by handling vanishing/exploding gradients
  (speed up is a side effect of it) and by applying different techniques:
  - Smart weight initialization
  - Using better activation functions
  - Batch normalization
  - Gradient clipping
  - Transfer learning
  - Unsupervised pre-training
  - Faster gradient descent optimizers

* Below are details about each of these techniques.

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
  ```
  - Normalize each input feature using mean and std (not shown...). After this the normalized batch has a 
    mean of zero and std of 1. 
  - Scale and shift each input feature (not shown...). In reality, not all data follows a normal distribution.
    If we let the network scale and shift the batch as needed, it'll learn the correct distribution of the 
    data. In other words, we leave flexibility using learned parameters to learn the data properly.
  
* How many new learnable parameters do we add when using BN?
    - Assume a single layer which takes as input a (m,n) matrix.
    - Each one of the n features will require a mean and std (not really learned, but still new params), scale 
      and shift. `Therefore n * 4 new parameters per layer`.

* What about batch normalization during inference?
  - During inference we probably won't have enough instances to compute all those values. Using less instances
    will make these values stochastic.
  - To solve this, the BN algorithm uses the values learned during training for new examples during inference.
  
* Batch normalization helps in MANY areas of DNN:
  - Solves the vanishing/exploding gradient problem throughout training.
  - Solves the need to use sophisticated weight initialization techniques (for vanishing gradients)
  - Even allows using the tanh and sigmoid activations without vanishing gradients
  - Speeds up training (convergence on an optimal solution)
  - Adds regularization and reduces overfitting
  - Added to the first input layer, it reduces the need to normalize inputs

* Batch normalization makes each epoch longer (since there more computations) but you'd need less epochs.
  - All in all, it's still faster with BN than without.


Gradient Clipping
------------------
* A technique used mostly in RNNs, where BN is tricky. It clips the gradients during backpropagation so that
  they never exceed some threshold.

* For example, we can choose to clip gradients to never be outside of the range [-1,1].
  - Note that it can change a gradient vector's direction. For example, [0.9, 100] is a vector that points
    much more in the second axis direction.
  - After clipping it, it becomes [0.9, 1], which is almost diagonal.

* To not lose the direction of the gradient, we can clip the normalized gradient such that its direction is
  preserved:
  - For example, [0.9, 100]:
    * [0.9 / (100.09), 100 / 100.09] = [0.0089, 0.9999]

* Preserving the gradient's direction is not guaranteed to generate better results. You'll need to explore
  both options.

Reusing pre-trained layers
--------------------------
* We can reuse the lower layers of an existing DNN to avoid training a large DNN from scratch in a process called
  transfer learning.

* When we do transfer learning: 
  - We take a trained DNN
  - Remove the top layers
  - Freeze the layers that are left, meaning their parameters won't change when we reuse this DNN
  - Add new top layers 
  - Retrain the DNN for a new task

* For example, assume that we want to build a DNN to classify car types.
  - We can find a trained DNN that was used for classifying different objects (e.g. cars, people, trees, etc.)
  - We remove the top layers and freeze a stack of its lower layers.
  - We add new top layers, including an output layer with the correct number of output neurons (e.g. number
    of car types we're trying to classify)
  - We retrain the network using a labeled dataset of cars and their model

* Transfer learning works best if your input has the same shape as the original network's input.
  - If it doesn't, you'd need to preprocess the DNN's input to match.

* How many top layers should I remove for my new task?
  - Output layer is probably a for sure remove (you label something different than what the original DNN was 
    labeling).
  - The closer the task is to the original DNN's task, you should remove less top layers.
  - The more training data you have, the more top layers you can unfreeze.
  - You'd want to use a low learning rate with the original DNNs top layers to not wreck their fine-tuned weights.
    In other words, since they are already trained and fine-tuned, you want to be gentle in how much you change 
    them.
  - If you can't get good performance:
    * If you have a lot of training data, replace the top layers with new ones and/or add new hidden layers.
    * If you don't have a lot of training data, remove the top hidden layers and unfreeze new top layers, 
      essentially, simplifying the DNN.

Unsupervised Pretraining (using GANs and auto-encoders)
-------------------------------------------------------
* Suppose that you can't find a good base model for transfer learning and you don't have a lot of labeled data. You 
  can still manage to train a model using this technique.

* Assume that you have a lot of unlabeled samples and a little of labeled samples:
  - Take an unsupervised model such as a GAN or an auto-encoder.
  - Train the unlabeled and labeled data using this unsupervised model. 
    * For example, an auto-encoder task can be to learn how to compress the data to a lower dimension and 
      then reconstructing it to the original form.
  - Take the lower layers of these unsupervised models after fitting and use them as the lower layers in a 
    new model. 
  - Add the output layer for your task on top. This will be a supervised learning task.
  - Train the network using whatever labeled examples that you have.

* The idea is because we have very little labeled data and much more unlabeled data, we can't use a supervised
  learning task.
  - However, we can still utilize the unlabeled data to learn from. The unsupervised model's layers pick up
    representations of the data that can be used later in the supervised model.


Pretraining on an auxillary task
--------------------------------
* Another useful technique for when we don't have much labeled data.

* We train a model on a task that's similar to the one we want to train a model for and reuse the layers of this
  model for our task.

* Example 1: face recognition
  - Suppose that you want to train a model for face detection. You have only a few images of your employees faces,
    not enough to train a model to recognize them.
  - Instead, take a face dataset from online (of random faces). Train a model to detect if two faces are the same 
    in this dataset (this model requires two inputs).
  - Reuse this models lower layers and retrain on the few images that you have.

Faster Optimizers
----------------- 
* We've seen a single way to do gradient descent (weight updates with a learning rate). There are different ways that 
  are more involved that can lead to better results.

* Momentum
  - Regular gradient descent updates the weights according to the learning rate and the slope of the gradient at 
    a given point.
    * It does not take into account the previous gradients.
  - Momentum optimization cares about previous gradients when updating the current ones. This gives it the effect
    of a momentum - like a balling ball rolling down a hill picking up speed (accelerating).
  - It subtracts the local gradient from the momentum vector m and uses m to update the weights:
    ```js
    m_1 = b * m_0 - step * grad
    weights = weights + m_1
    ```
  - Why is it a faster optimizer?
    * Consider a situation where the algorithm reached a plateau, an area where the gradient is constant.
    * Using the regular GD optimizer, we'd deduct the weight updates directly from the weights:
      ```js
      w1 = w0 - step * grad
      ```
    * The weight updates will get reduced significantly and eventually will stop changing (if the GD does not
      escape the plateau).
    * With the momentum approach, we have vector m which absorbs the weight update. It contains sort of a memory
      to the previous updates in that vector. This means that if our previous point was a steep part of a curve,
      its momentum is captured in vector m which affects it when it gets to the plateau.
  - In summary, momentum can help the optimizer to escape local minima.
    * It's great for deep NN where the values in the lower layers of the network are vastly different than the 
      ones in later layers.
    * One drawback of using momentum is that it introduces another hyperparameter (b, see above). A good value
      for it is 0.9.

* Nesterov accelerated gradient
  - This is a tweak to the momentum optimizer. Instead of taking the gradient at the point of the weights, we 
    take it at the points of the direction of the momentum:
    ```js
    m_1 = b * m_0 - step * grad(weights_0 + b * m_0)
    ```
    * Adding b*m_0 to the weights changes their direction in the direction of the momentum, which always 
      points towards the minimum.
  - Using this tweak, we can reach convergence much faster.

* AdaGrad
  - Corrects the direction of the gradient towards the global optimum:
  ```js
  s_1 = s_0 + grad(weights_0) x grad(weights_0)
  weights_1 = weights_0 - step * grad(weights_0) / sqrt(s_1 + eps)
  ```
  - `grad(weights_0) x grad(weights_0)` - element wise multiplication which is basically the square of the gradients.
    * This has the effect of amplifying partial derivatives that are steeper. For example, if the gradient is steep
      along the ith dimension, this steepness will get emphasized more than other dimensions due to the square.
  - The second step is almost like regular gradient descent only that the weights are scaled down by a factor of
    `sqrt(s + eps)`.
  - AdaGrad pushes the gradient more at the direction of its steepest weights. It leads to the function not taking
    into account partial derivatives that make the gradient lose focus.
  - **DO NOT USE** AdaGrad for deep networks. It stops before the global optimum is reached due to the scaling down of
    the weight updates and the learning rate.

* RMSProp
  - Tweak to the AdaGrad optimizer. Instead of using all the gradients since the beginning of training, it considers
    only the ones from recent iterations.
  - This prevents the optimizer from slowing down before reaching global optimum:
  ```js
  s_1 = p * s_0 + (1 - p) * grad(weights_0) x grad(weights_0)
  weights_1 = weights_0 - step * grad(weights_0) / sqrt(s_1 + eps)
  ```
  - It uses the decay rate p, which reduces the amplification
  - Outperforms AdaGrad almost always.

* Adam
  - Adaptive Moment Estimation
  - Combines the ideas of all optimizers shown above (combines momentum and RMSProp):
  ```js
  m_1 = b1 * m_0 - (1 - b1) * grad(weights_0)
  s_1 = b2 * s_0 + (1 - b2) * grad(weights_0) x grad(weights_0)
  m_hat = m_1 / (1 - b1 ^ t)
  s_hat = s_1 / (1 - b2 ^ t)
  weights_1 = weights_0 + step * m_hat / sqrt(s_hat + eps)
  ```
  - `t` represents the iteration number



Learning Rate Schedules
-----------------------
* Setting the learning rate too high and we'll overshoot the global minima, setting it too low and the algorithm
  won't converge.

* To find the optimal learning rate, we can plot a learning curve:
  - We choose a number of increasing learning rates
  - For each one, we train the model and record the loss per epoch
  - We plot the different learning rates and choose the best one 

* One thing that was noticed is that the learning rate can be high in the beginning and then reduced. This gets the
  benefit of both approaches - faster convergence and hitting the global minima.

* There are different strategies (called learning rate schedules) to tweak the learning rate during training:
  - Power scheduling 
    * Learning rate decreases as a function of the iteration
  - Exponential scheduling
    * Learning drops by a factor of 10 every s steps
  - Piecewise constant scheduling
    * Reducing the learning rate by a set amount after a set number of epochs
  - Performance scheduling
    * Reduce the learning rate by a factor when the error on the validation set stops dropping

Regularization
--------------
* Deep neural networks have sometimes millions of parameters. This gives them a very large degree of freedom to fit
  complex datasets. However, it also makes them prone to overfitting.

* We already saw two very good techniques that regularize a NN:
  - Early stopping
  - Batch normalization (regularization here is a positive side effect)

* l1 and l2 regularization:
  - These work in the same way they work in other models.
  - l2 can constrain the network weights and work best in cases where there are correlated features.
    * l2 regularization is not a good match for the Adam optimizer. This optimizer already has weight decay
      and adding this regularizer can cause "over regularization".
    * l2 works with the momentum and nestrov optimizer.
  - l1 can remove (make 0) weights for features that are not important and is good for sparse datasets.

* Dropout
  - At each training step (not epoch, but for every batch iteration), every neuron (including input neurons
    but excluding output neurons), has probability p of being temporarily "turned off".
    * The turned off neuron is ignored during this step. `It outputs 0.`
    * It can come back alive in next step.
  - p is called the `dropout rate`
    * It's typically set to 20-30% in RNNs and to 40-50% in CNNs
  - Dropping neurons out only happens during training
  - This works because the learning "spreads" across the layer
    * In every layers, you have neurons that are more active than others. Their weight updates are larger.
    * If they get turned off, there is no choice but for other neurons to adapt.
  - Another way to understand why dropout works:
    * Since at every step we turn off neurons randomly, we essentially train 2^N different networks, where N
      is the total number of droppable neurons (power has base 2 since a sampled neuron can either be a part 
      of the network or not).
    * What happens is that we get an ensemble of different networks where the result is an average of all those
      networks.
  - In practice, you'd want to apply dropout to neurons in the top 3 layers (excluding output layer of course).
  - During testing and inference we need to adjust because of dropout. To understand why:
    * Assume we have a p=0.75. This means that on average 25% of the neurons are active during training.
    * Once we're done training, we remove dropout and now each neuron is connected to x4 neurons (during training
      this neurons were dead so that neuron wasn't considered connected to them).
    * What happens is that input neuron's activation is training on a "diluted" function of the weights. During
      testing, we reactivate all neurons and the input neurons start receiving activations that are much larger
      than what they were trained on - this is called `activation mismatch`.
    * To fix it, we need to scale the activations down by a factor of (1-p). For example, with a p=0.75, we 
      scale the weights down by a factor of 1-0.75 = 0.25 during test.
      - Another way to achieve the same thing is by scaling the dropped out weights up during training by dividing
        them by the keep probability. Then during test we don't need to do anything (how keras does it).

  - When using dropout, comparing the training and validation loss during training can be misleading.
    * Dropout is only applied to the training set, which can make the validation loss of the training worse
      or equal to the validation loss (where no dropout is applied).
    * This can give the impression that the model has a training loss equal to the validation loss where in 
      reality, if we removed dropout, we'd see that the training loss is much lower, hence the model is 
      overfitting.
    * To get a better picture, evaluate the model after training on the training set (without dropout) and 
      compare the loss to the validation set. 
      - If you see overfitting then, increase the dropout rate and retrain
      - If you see underfitting then, reduce the dropout rate and retarin


Summary and Practical Guidelines
--------------------------------
* With so many different initial weights, optimizers and regularization techniques which ones do we choose?

* The following is a best practice configuration that seems to work with most networks:
  - Kernel initializer: 
    * He initialization
  - Activation function:
    * ReLU for shallow
    * Swish for deep
  - Normalization:
    * None for shallow
    * Batch norm for deep
  - Regularization:
    * Early stopping
    * Weight decay (l2)
    * Dropout
  - Optimizer
    * Nesterov
    * AdamW
  - Learning rate schedule
    * Performance scheduling

