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
    Feature Map Size = [(Input Size - Filter Size + Stride * Padding) / Stride] + 1
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
    Feature Map Size = [(Input Size - Filter Size + Stride * Padding) / Stride] + 1
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



