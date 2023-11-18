Autoencoders
------------
* Autoencoders have several uses:
  - Dimensionality reduction
  - Feature detectors used for unsupervised learning
  - Generative models

* Key idea: patterns help storing information more efficiently
  - For example, which of these two sequences is easier to memorize:
    * 40, 27, 25, 36, 81, 57, 10, 73, 19, 68
    * 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14
  - Once you notice the pattern in the second sequence, you need to memorize fewer elements.

* An autoencoder looks at inputs, converts them to an efficient latent representation and then spits out
  something that hopefully looks very close to the inputs.
  - The latent representation is the equivalent of the pattern learned by the autoencoder
  - The output is the autoencoder's memory of the inuput

* An autoencoder is built from:
  - Encoder
    * Converts inputs to a latent representation
  - Decoder
    * Converts the internal representation to an output

* The encoder input size must equal the decoder output size in an autoencoder
  - The outputs are also called `reconstructions` and the loss function measures the `reconstruction loss`.

* The latent representation has a lower dimension than the input data (and the output since it equals the input dimension).
  - The autoencoder must find a way to learn the most important features of the data because of it. It must 
    find a way to regenerate the data from less.

Autoencoder for dimensionality reduction
----------------------------------------
* An autoencoder that uses only linear activation functions with a cost function that is MSE, ends up performing PCA.

* Autoencoders are not as good as other dimensionality reduction algorithms, their advantage is that they can handle
  larger datasets with more features better.
  - A common technique for visualizationis to first use autoencoders to reduce the dimensionality to a good level 
    (without losing too much information).
  - Then use a better dimensionality reduction algorithm to reduce the dimensionality to 2D or 3D.

* The hidden layer has the dimensionality we want to reduce to.

Stacked Autoencoders
--------------------
* Adding more hidden layers to an autoencoder makes it possible for it to learn more complex codings.
  - Adding too many hidden layers makes it easy for the autoencoder to learn the mapping between
    the inputs and outputs.
  - Essentially, it'll just learn how to map a pixel onto itself, which will make it not learn the 
    important features of the input data.

* Stacked autoencoders are usually symmetrical
  - Input layer of n dimensions
  - Hidden layer 1 with q dimensions
  - Hidden layer 2 with y dimensions
  - Hidden layer 3 with q dimensions
  - Output layer with n dimensions

Unsupervised Pretraining Using Autoencoders
-------------------------------------------
* If you have a large, unlabeled dataset, you can first train a stacked autoencoder using all the data, then reuse
  the lower layers to create a NN for your actual task and train it using the labeled data.
  - The idea is similar to using lower layers of a deep NN in transfer learning, we're adjusting the model
    to learn low level patterns from the data by mapping it back to itself.

* In phase 1, we train the autoencoder on all the data. The output is a decoder that reconstructs the inputs.

* In phase 2, we build a new NN. We take the weights of the hidden layers of the auto encoder and use them
  in a model where the output is a dense softmax layer.

Convolutional Autoencoders
--------------------------
* The autoencoders we've seen so far are not meant to work with high-res images. For that, same as with regular 
  ML models, we better use convolutional autoencoders.

* The encoder in this case is a regular CNN composed of convolutional layers and pooling layers.
  - It stacked similarly to a CNN - it reduces the spatial dimension of the image and increases the depth 
    (the feature maps).

* The decoder, unlike a regular CNN, does the reverse - it upscales the image and reduces the depth back
  to the original dimension.
  - To do this, it uses transpose convolutional layers or upsampling layers with conv layers.

Denoising Autoencoders
----------------------
* Another applications of autoencoders is to denoise inputs.

* We artificially add noise to inputs during training and the AE tries to reconstruct the original input that's noiseless.

* We can implement it by taking a regular, stacked autoencoder
 and add to it a Dropout layer or a GaussianNoise layer.
 - The idea is that the inputs are normal, the noise gets added by the encoder and the decoder tries to reconstruct the image without the noise.
 - The dropout layer makes learning the original input more difficult and is supposed to emulate noise by way of regularization.
 - See the example in the notebook when a dropout layer is applied to the MNIST images.


Variational Autoencoders
------------------------
* Generative autoencoders - they can generate new instances that look like they were sampled from the training data.
  - Unlike the other AEs that we saw, the variational AEs outputs are partly determined by chance (probabilistic outputs).

* These AEs follow a similar process to Bayesian inference
  - They update a known probability distribution (priori) with new data, from Bayes theorem.
  - The updated distribution is called posterior.

* The encoder of a VAE produces the mean (μ) and standard deviation (σ) of the codings. During training, the model samples from the unit Gaussian 
  distribution  and uses the reparameterization trick to obtain a sample from the codings distribution using the mean and standard deviation provided by the encoder. The decoder then takes this sample and tries to reconstruct the input using a reconstruction loss.
  - The VAE is trained to balance the reconstruction loss with a regularization term that encourages the codings to follow a unit Gaussian distribution.

* The variational AE learns how to map vectors sampled from a gaussian distribution back to their original inputs.
  - The variational AE has a cost function that's a sum of two losses. The first the MSE (same as before). It uses it to map outputs to inputs.
  - The second is the `latent loss`, that pushes the codings to look like points from a Gaussian distribution (part of the encoder).
  This loss is the KL divergence between the target distribution and (Gaussian) and the actual distribution of the codings (latent).
  - The KL divergence measures how much a distribution diverges from another distribution. If we minimize it, we can reduce the divergence and converge the two distributions.