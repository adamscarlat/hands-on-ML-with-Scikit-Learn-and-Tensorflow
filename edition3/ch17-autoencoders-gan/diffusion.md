Diffusion Models
-----------------
* This is also a way to generate images that outperforms GANs.

* The idea is to first (in the forward pass) iteratively drown an image with Gaussian noise.
  - At each iteration we add more noise to the image in a linear fashion.

* We do this while also training a model to learn the reverse process from a fully noised image back to an image that resembles
the original image.

* The idea is similar to GANs - the model learns to map from Gaussian space to an image. The learning process is different though.


* A dataset for a diffusion model consists of images that went thru a
process to add gaussian noise and a label which is the noise that 
was added to these images.
  - The model learns to predict that noise vector and then we can subtract it from the input to get an image.
  - During inference, we pass Gaussian noise to the model, get back noise, subtract it from the noise and get a generated image.
  - Inference has to be iterative as well, we need to reduce the noise and call the model multiple times to generate an image.


