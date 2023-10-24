Convolutional Neural Networks
-----------------------------
* CNNs are a type of deep neural network specialized in computer vision tasks. It's based on the 
  architecture of the visual cortex in our brains.

* Biological neurons in the brain respond to specific patterns (e.g. a curve, a diagonal, etc.). The signal 
  from all those neurons flows up and powers neurons that detect more complex features (such as a circle, square, etc.).

* Why do regular, fully connected deep neural networks (DNNs) not perform well in computer vision tasks?
  - `High dimensionality`: Images with many pixels (e.g. over 100x100) require a lot of parameters to capture the 
    information in the image. This would result in a model that is difficult to train, prone to overfitting, and 
    computationally expensive.
    * For example, a 100x100 image has 10,000 input neurons when flattened. If the next hidden layer has only 
      a 1000 neurons (which is too little as is), it'll require 10 million parameters just for this layer.
  - `Lack of Spatial Awareness`: DNNs treat the input data as a flat vector, disregarding the spatial 
    structure and relationships within images. This means they don't take into account the proximity of pixels or the significance of local patterns, which is crucial for understanding images.


Convolutional Layers
--------------------

* Convolutions
  - Mathematical convolution is an operation that combines two sets of data to create a third set of data. 
    It's commonly used in various fields, including signal processing, image processing, and mathematics.
  - Imagine you have two sequences of numbers, let's call them A and B. To perform a convolution, you slide one of 
    these  sequences (for example, B) over the other (A) and at each position, you multiply the overlapping values and then add up the results. The position where you are currently multiplying and adding is called the "current position."
  - For example, 
    * Sequence A: [1, 2, 3]
    * Sequence B: [0.5, 0.5]
    * To compute the convolution of A and B, we slide B over A and calculate the multiplication and addition at each 
      position. We'll denote the current position with "x."
      - C[0] = (1 * 0.5) = 0.5
      - C[1] = (1 * 0.5) + (2 * 0.5) = 1.5
      - C[2] = (2 * 0.5) + (1 * 0.5) = 1.5

* In a convolutional layer, neurons are not connected to all neurons from the previous layer, but only to neurons
  that are a part of their `receptive field` - that is the connections between a neuron in layer l to neurons
  in layer l-1.
  - Unlike a dense layer in a DNN, in a conv layer is not 1D - It's 2D.
  - A neuron in conv layer l at position (i,j) is connected to the layer l-1 outputs of neurons in:
    * Rows `i` to `i + fh -1`
    * Columns `j` to `j + fw - 1`
    * fh and fw are the height and width of the receptive field.
    * For example, assume fh=3 and fw=2. 
      - neuron (2,4) in layer 2 is connected to neurons in range [2, 2+3-1], [4,4+2-1] -> [2,4], [4,5]
      - so neuron (2,4) in layer 2 is connected to layer 1 neurons: 
        (2,4), (2,5)
        (3,4), (3,5)
        (4,4), (4,5)
    * The layers must have the same height and width for this to work. If they don't, we pad the layers with zeros.

* It's possible to connect a large input layer to a small next layer as well by spacing out the receptive fields.
  - We call this spacing `stride`. It can be different horizontally and vertically.
  - A neuron in conv layer l at position (i,j) is connected to the layer l-1 outputs of neurons in:
    * Rows `i × sh` to `i × sh + fh – 1` 
    * Cols `j x sw` to `j x sw + fw - 1`
      - Where sh and sw are the vertical and horizontal strides
    * For example, assume sh, sw = 2 and fh, fw = 3.
      - Neuron (2,4) in layer 2 is connected to neurons in range [2 x 2, 2 x 2 + 3 - 1], [4 x 2, 4 x 2 + 3 - 1] -> [4,6],[8,10]
      - Neuron (2,5) (next neuron over) is connected to neurons in range:
        [2 x 2, 2 x 2 + 3 - 1], [5 x 2, 5 x 2 + 3 - 1] -> [4,6],[10,12]
      - Notice that the next neuron in layer l skipped some neurons in layer l-1 because the stride is greater than 1.

* Filters
  - We've seen that in a conv layer, a neuron in layer l is connected to a 2D receptive field in layer l-1. 
  - This receptive field can be defined as a matrix of specific size that "slides" on the 2D input according
    to the stride defined.
  - Now instead of a single matrix, we have many and we call them `filters`. Each filter slides over the input 
    in the same way (according to slide and receptive field size). 
  - Each filter learns to detect different patterns in the input. For example, horizontal lines, vertical, curves, etc.
    * Each filter gets activated by different patterns in the input.
  - The weights of each filter change through the learning process to detect a specific pattern.
  - As we go higher in the CNN, these filters learn more complex patterns.
  - The filter convolved with the input creates a `feature map`
  - Given an input size, filter size, stride and padding, we can compute the size of the feature map:
    ```js
    Feature Map Size = [(Input Size - Filter Size + 2 * Padding) / Stride] + 1
    ```
    * For example, 
      - Input size - (10 x 10)
      - Filter size - (3 x 3)
      - Stride - 2
      - Padding - 1
      ```js
      [(10 - 3 + 2 * 1) / 2] + 1 = 5.5 -> 6
      ```
      - Result is (6x6)

* Stacking Filters
  - We've seen how a neuron in layer l is connected to a 2D frame in layer l-1 called a filter (or a receptive field).
  - This filter slides over the input and gets activated by specific patterns.
  - Now instead of a single filter, we actually slide a stack of them, each will get trained to detect a different 
    pattern.
  - Each filter in the stack has its own set of weights.
  - Each time we apply a filter by sliding it, the same set of weights is used for all positions on the input image.
    * For example, in layer l-1, neuron (i,j) and neuron (i+5,j+5) will share the same weights with filter k.
  - This is called parameter sharing and it allows the filter to learn a pattern no matter its position in the 
    input layer.
    * Once the filter got trained enough to successfully recognize a pattern in one part of the image, it can 
      recognize it in another part of the image.

* Each time we slide a filter over a 2D subsection of the input, we do:
  - Element-wise multiplication of the filter's weights with the input's 2D subsection.
  - Summation of the result
  - The result is an output neuron in the filter
  - The overall result of applying a filter to an input is called a `feature map`

* Convolution on multiple channels
  - When the input layer has multiple channels (e.g. 3 channels for input RGB or multiple feature maps of a conv layer),
    we need to have a filter per channel.
    * For example, an input image with 3 channels (RGB) and a conv layer with 32 filters will actually have 32 x 3 = 96 
      filters.
  - Each channel filter does the convolution as mentioned above and then we sum their results.
    * Continuing the example above, filter 1 will have 3 channel filters (each its own weight matrix).
    * Each channel filter convolves the input to create a feature map 
    * We then sum the feature maps together to get a single feature map
    * The overall result will be 32 feature maps

* Summary:
  - In a convolutional layer, we have a stack of `filters`. Each filter can be thought of as a small weight matrix
    that can learn specific patterns
  - `Each of the filters slides across the input layer` (which is also 2D) according to stride and padding.
  - `The filters convolve with the input`. 
    * This means that at each slide of the filter, an element-wise multiplication followed by summation is performed.
    * The result is a single numerical value
  - The convolution of each filter with the input results in a `feature map`
    * We also do a non-linear activation on the result.
  - Each 2D filter generates its own feature map when it interacts with a 2D input layer. The feature map's size will 
    be:
    ```js
    Feature Map Size = [(Input Size - Filter Size + 2 * Padding) / Stride] + 1
    ```

* Summary in a mathematical form:
- To compute `z_ijk`: the output neuron in filter map k at position (i,j):
```js
z_ijk = 
sum_u(
  sum_v(
    sum_kt(
      x_ijk_t * w_uvktk
    )
  )
)

a_ijk = ReLU(z_ijk)
```
- The inner sum is for all filters k. It multiplies `x_ijk_t` - the neuron at layer l-1 at position (i,j) channel k_t (the 
  input layer can have multiple layers as well), with `w_uvktk` - the weight at filter k position (u,v).
  * (i,j) are computed according to the stride:
  `i = i × s_h + u`
  `j = j × s_w + u`
- The center sum is over the input width v
- The outer sum is over the input length u

* Convolutional layer - memory requirements
  - During training we must hold all activations of all layers for each batch (since we need them for the backprop).
    * If not enough memory, reduce the batch size.
  - During inference, we can dispose the previous layers activations.


Pooling Layers
---------------
* Pooling layers subsample their input - they reduce the dimensions of their inputs.
  - This helps with memory and CPU usage and also helps with overfitting.

* Similar to conv layers, a pooling layer will have a 2D size, padding and stride.
  - Unlike a conv layer, a pooling filter has no weights. 
  - Instead, it does a very simple aggregation function over its connected input neurons
  - For example, the most common pooling layer is a 2x2 max pooling layer.
    * It slides over the results of a conv layer (over each feature map) and outputs the max input out of 
      4 (size 2x2) that it currently slides over.

* A pooling layer introduces a level of invariance to small translations
  - This means that feature maps that activate on similar pattern (e.g. straight line shifted right and left in 
    another), will be the same after going through a pooling layer.
  - A pooling layer makes the network less sensitive to small shifts or movements in the input data. Max pooling retains the 
    most important features by selecting the maximum value within a small window, which means that objects or patterns can be recognized even if they are slightly shifted in the input. 
  - This property helps the network maintain robustness to small positional variations, making it more effective in tasks like 
    image recognition where object positions may vary.

* Keep in mind that pooling also has downsides
  - It reduces information in a lossy way. This may not be desirable for some tasks such as sematic segmentation.

* There are different types of pooling layers
  - For example, there is an average pooling layer (takes the average instead of max).
  - It has been shown that max pooling layer is the best. It's fast to compute and it propagates the cleanest
    signal forward.

* The number of outputs of a pooling layer is equal to the number of input channels
  - The only thing that changes is that the size of each channel decreases.
  - For example, the output of a conv layer is (64, 114, 32) - 32 feature maps, each 64x114
    * Feeding it to a pooling layer of size 2x2 with a stride of 2 (no padding) we get back
      [(64 - 2) / 2] + 1 = 32
      [(114 - 2) / 2] + 1 = 57
      (32, 57, 32) - 32 reduced feature maps each of size 57x32

CNN Architectures
-----------------
* A CNN is typically composed of convolutional layers followed by a ReLU activation followed by a pooling layer.
  - The layers make the image smaller and deeper (more feature-maps) as it goes through the network.
  - At the top of the network, there is typically a fully connected layer (+ReLU) and the final layer 
    outputs a prediction (e.g a dense layer with a softmax activation for multiclass of a sigmoid for binary, etc.).

* It makes sense to double the number of filters in every conv layer as we go into the CNN.
  - Since the first layers in a CNN pick up simple patterns (lines, curves, etc.), there aren't as many 
    of these patterns as there are combinations of them.
  - The higher layers have more filters since they learn to activate on combinations of simpler patterns (e.g 
    square, circle and even faces in higher layers).

* ResNet
  - An advanced CNN architecture that won the 1000 image challenge in 2015.
  - Uses the concept of `skip connections`
    * The signal feeding into the layer is also added to the output of the layer.
      `Output = F(Input) + Input`
    * This forces the network to learn the residual - instead of just learning what should be the transformation
      F(Input) that generates Output, it learns `What can be added to the Input in order to generate Output`.
      - It learns F(Input) not as a function from scratch, but as an addition to the inputs.
  - Residual units also have the benefit of reducing the vanishing gradient problem and increasing generalization.

    

Data Augmentation
-----------------
* When it comes to images, adding augmented version of images to the train set can help with overfitting.
  - In general, increasing the size of a dataset helps with overfitting.
  - Augmentation can also help with an unbalanced dataset

* Successful augmentation techniques for images include:
  - Shift, rotate and resizing of every image in the train set
  - Change lighting conditions (change contrast)

* It makes the model more tolerant to variations in the image.



Classification and Localization
-------------------------------
* This type of task deals with finding a bounding box around an object in an image.

* This task can be broken down to a regression task that outputs 4 values:
  - Horizontal and vertical coordinates
  - Height and width 

* We can use transfer learning on one of the CNN models that Keras offers.

* To measure the success of a predicted bounding box, we shouldn't compute the loss on the 4 regression values
  as it's not providing enough information to the model.
  - Instead, we compute the `Intersection over Union (IoU)` which is the ratio of the intersection of the 
    area of the predicted and real boxes over the area of their union.
  - In other words, this ratio is higher when the intersection of the predicted vs actual boxes is closer to 1.

Detecting Multiple Objects
--------------------------
* The approach above works well when we need to detect a single object in an image along with its bounding box.

* What do we do when we need to classify multiple objects in a single image?

* Sliding CNN
  - This simple approach involves sliding the CNN along the image and classifying at each location.
  - Then we do "non-max suppression", we get rid of all boxes that overlap with the discovered object boxes.
  - This approach works well but it's slow in practice (need to run the CNN many times per prediction).

* Fully convolutional network
  - Here the idea is to do semantic segmentation - classify every pixel in an image according to the object it
    belongs to.
  - The idea here is instead of having a top Dense layer after the last conv layer, we use another conv layer.
    * For example, assume that the last conv layer in a CNN outputs 100 feature maps of size 7x7
    * It's connected to a dense layer with 10 neurons.
    * Since the dense layer is fully connected we have a flatten layer before it which takes the feature
      maps and turns them into 100 x 7 x 7 = 4900 neurons. Each of those is fully connected to the 10
      neurons of the dense layer so we get 10 x 4900 weight matrix.
  - Now assume that instead of a dense layer with 10 neurons, we add a convolutional layer with 10 filters of 
    size 7x7 and a stride of 1:
    * Since the filter size is the same as input size, the stride is 1 and there is no padding, we're getting 
      back 10 feature maps of size 1x1:
      ```js
      feature map size = [(input size - filter size + 2 * padding) / stride] + 1
      [(7-7 + 2 * 0) / 1] + 1 = 1
      ```
    * If we check the value of each of these neurons we'll see that it's equal to the value of the neurons
      in the dense layer.
  
  - By using a conv layer instead of a dense layer, `we can use any input size`. To see why, consider the same network
    we used in the example above (the FCN one). Instead of a 224x224 image, we'll use a 448x448 image.
    * Now our last conv layer (10 filters of 7x7, stride=1, no padding) generates 10 feature maps that are 8x8.
    * Before, when we got 10 feature maps of 1x1 each, it was easy to see how this mapped to the classes in the 
      dense layer.
    * Now, what does 10 8x8 feature maps mean? it's the same as sliding the original CNN over the image using
      8 steps per row and 8 steps per column. 
      - The FCN took that into account automatically.
      - Each cell in the 8x8x10 stack corresponds to the 10 classes at that cell's location.
      - If you do the math, you'd see that sliding the original CNN over a 448x448 image would yield 64 
        slides, each producing 10 classes.

  - When you think about it, it makes sense. 
    * The original CNN can do a single 10 class prediction on a 224x224 image.
    * When we increase the image size to 448x448, we need to slide the CNN over it in order to cover the whole 
      thing.
    * By using constant filter sizes no matter the input size, we're bound to get larger feature maps at the
      last layer.
      - For the original image, we got 7x7 feature maps (that's why we get 1x1 feature maps after using a 7x7
        filter on them).
      - For the larger image, we got 14x14 feature maps (twice as large since the image is twice as large).
      - Now the 7x7 filters have room to slide over the inputs and convolve. They will slide over the input
        generating 10 8x8 feature maps.
  
  - The last step is to take that results and map it to the original image size
    * It's done by a process called "upsampling". Not clear to me how that works. See Semantic Segmentation

  - A famous version of this technique is called YOLO - you only look once. It's an efficient and accurate 
    way to detect object in images.

Semantic Segmentation
---------------------
* In semantic segmentation, each pixel is classified according to the class of the object it belongs to.

* The main challenge with SS and CNNs is that CNNs can classify complex patterns but without their location
  information in the image.
  - A regular CNN will know that there is a car in the image, but it won't know where.

* To handle this issue, An FCN is used and at the end `upsampling` is performed. Essentially, we bring back
  the feature maps to their original size.

* Using SS, if there are similar objects of the same type near each other they will get clumped together.
  - For example, parked bicycles on the side of the road will get clumped.
  - `Instance Segmentation` is a technique that builds on SS and can distinguish objects of the same class.




