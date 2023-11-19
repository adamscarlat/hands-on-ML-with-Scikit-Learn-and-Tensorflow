Generative Adversarial Networks (GANs)
--------------------------------------
* The idea of GANs is simple, have two NN compete against each other in hope that this competition will push them to excel.

* A GAN is composed of 2 NN:
  - Generator
    * Takes in random, latent inputs from a Gaussian distribution.
    * Outputs some data, typically an image.
    * The idea is very similar to the decoder in a variational autoencoder (takes in codings that are samples of a Gaussian
      distribution and outputs an image). The training method is different though.
  - Discriminator
    * Takes either a fake image from the generator or a real image from the training set, and has to guess whether the input image
    is fake or real.

* Note that the generator and the discriminator have opposite goals.
  - The generator attempts to generate images that are real enough so that they fool the discriminator. The worse the discriminator is doing, 
    the better te generator is doing.
  - The discriminator must be able to discern fake from real. So if the generator is doing poorly, the discriminator is doing well.

* Because the objectives of the NNs are different, they cannot be trained like a regular NN. Instead, training is divided into two phases:
  - In the first phase, we train the discriminator. A batch of real images from the training set and an equal number of fake images from the generator are passed to the discriminator.
    * The discriminator is trained on this batch for one step using
      binary cross-entropy for loss.
    * Only the weights of the discriminator are updated during this phase.
    * The discriminator's task is straight forward - it learns to classify fake vs real.
  - In the second phase we train the generator. We start again by generating a batch of fake images.
    * We pass them (only the fake ones) to the discriminator and set the label of all of them to 1 (real). In other words, we fool the discriminator, 
      making it believe (wrongly) that the fake images are real.
    * The weights of the discriminator are frozen during this step and we only update the weights of the generator.
    * This way we improve the generator in producing better images. When the discriminator correctly identifies an image as fake, the generator can 
      improve its weights.

* We set the label of these fake images to 1 (real). When the discriminator (not trainable at this phase) labels an image as fake, 
  the generator can improve the weights accordingly - what needs to change so that the discriminator will say it's a real image.

* Mode collapse
  - One of the downsides of a GAN network is situation called "mode collaps". 
  - In this scenario, the generator gets good in generating fake images of a single class and the discriminator gets good in discriminating that one class. 
  - The network forgets how to generate/discriminate other classes.
  - There are different ways to overcome this scenario (see experience replay and mini-batch discrimination).


Deep Convolutional GANs (DCGANs)
-----------------------
* DCGANs have a few guidelines that make them work:
  - Strided conv layers in the discriminator and transposed conv layers in the generator.
  - Use batch normalization in both the generator and the discriminator, except in the generator’s output layer and the discriminator’s input layer.
  - Remove fully connected hidden layers for deeper architectures.
  - Use ReLU activation in the generator for all layers except the output layer, which should use tanh
  - Use leaky ReLU activation in the discriminator for all layers

* 