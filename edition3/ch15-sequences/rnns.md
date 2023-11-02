Processing Sequences Using RNNs and CNNs
----------------------------------------
* Recurrent Neural Networks (RNNs) can analyze time series data such as:
  - Number of daily users on a website
  - Hourly temperature in an area
  - Daily power consumption
  - etc...

* RNNs can work on sequences of arbitrary lengths rather than fixed size inputs. For example:
  - Sentences, documents or audio.

Recurrent Neurons and Layers
----------------------------
* So far, all neuron types we saw were ones where the activations flow in one directions - from input to output.
  - Recurrent neurons have a connections forward (regular) and another connections pointing backward.

* Each recurrent neuron at time t takes two inputs:
  - x(t): the input at time t
  - y(t-1): the output of the neuron from the previous time step

* Keep in mind that both inputs are vectors when we consider an entire layer of recurrent neurons.
  - This means that for each layer we have two weight matrices: `W_x` and `W_y`
  - We can represent the entire input into the layer as `x(t)` and `y(t-1)`
  - We can represent the output of a recurrent layer with the following equation:
  ```js
  y(t) = ReLU(x(t) @ W_x + y(t-1) @ W_y + b) 
  ```
  - ReLU is just one option for an activation function. It can be any.
  - `y(t)` is an `m x n` matrix containing the layer's output at time step t for each instance in the
    mini-batch (m).

* Note the recurrence relation in that equation:
  - y(t) is a function of x(t) and y(t-1)
  - y(t-1) is a function of x(t-1) and y(t-2)
  - y(t-2) is a function of x(t-2) and y(t-3)
    * and so on...
  - This makes y(t) a function of all inputs since time t=0 (X_0, X_1, ... , x(t))

Memory Cells
------------
* The purpose of using output from time t-1 in t at every layer is to maintain a memory or a dependence of 
  previous outputs.
  - If you think about a regular NN, each input into the network changes the weights independently and the accumulation
    of weight updates from all inputs make the network learn. However, each layer in the network sees each input 
    as independent.
  - In an RNN, the layers take into account all previous outputs. This allows the network to pick up patterns through
    time, or in case of a sequence, from previous items in the sequence.

* For example, assume we have a network that can guess the next word in a sentence.
  - Our training data is organized as follows:
    * Sample i: "I walked into the room, and there was a huge TV"
  - We feed the input into the RNN one word at a time:
    * t=0 (I)
      - Here the network receives 0 as previous state
    * t=1 (I walked)
      - Here the network gets the output of the previous state
    * t=2 (I walked into)
      - Here the network gets the output of the previous state

* At each time t > 0, the network receives the state of the previous time steps. This is considered the network's memory.
  - In practice, using regular RNNs the network after 10 steps starts forgetting as the memory gets diluted.

Inputs and Output Sequences (types of RNNs)
---------------------------
* RNNs can take inputs and produce outputs in various forms:

  - `Sequence to Sequence`
    * The RNN takes a sequence of inputs (e.g daily power consumption) and it produces a sequence of outputs:
      - Inputs: x_1, x_2, ..., x_m 
      - Outputs: y_1, y_2, ..., y_m
    * At each time step we feed the next input. Since the cells maintain and share state, the network "remembers"
      the previous states. 
    * The RNN is trained to predict the next item in the sequence.

  - `Sequence to Vector`
    * The RNN takes a sequence of inputs. It ignores all outputs except the last one.
    * For example, the input is a movie review and the output is a sentiment score (0 hate, 1 love)

  - `Vector to Sequence`
    * Network takes the same input vector at each time step and let it output a sequence.
    * For example, input can be an image and the output can be a caption for that image. At each time step
      we give it the same image and it outputs a word at that time step. The network takes into account
      the input image and the previous output every time.
      - Image: dogs playing
      - Output t0: dogs
      - Image: dogs playing
        Output t1: playing

  - `Encoder decoder`
    * The encoder is a sequence to vector network
    * The decoder is a vector to sequence network
    * Can be used for translation (more on this in chapter 6)

Training RNNs
-------------
```
WHEN WE LOOK AT AN RNN UNROLLED THROUGH TIME, EACH TIME STEP IS AN ENTIRE FORWARD PASS ON ONE PART OF THE INPUT SEQUENCE.
```
  - Each time step produces an output 
  - The hidden connections between time steps show passing of previous time step's output to the current time step


* To train an RNN we need to unroll the network through time. This is also known as `backpropagation through time`
  (BPTT).
  - Unrolling through time is specific to a single input sequence processed by the network. It represents the 
    unfolding of the recurrent neural network (RNN) operations in a sequential manner for a particular input sequence.
  - For example, assume a seq to seq network. I feed it the sentence: "I walked into the room". We feed it word for word
    and at each time step we get an output:
    At time step t=1, you feed "I"
    At t=2, you feed "walked."
    At t=3, you feed "into."
    At t=4, you feed "the."
    At t=5, you feed "room."
  - At each time step, the RNN processes the current token, updates its hidden state using the previous hidden state 
    and the current token, and produces an output.
  - When seeing an unrolled RNN it's important to remember that it's the same network at different time steps. This
    means that it's the same weight matrices.

* Just like a regular NN, we start with a feed forward pass through the unrolled network.
  - At the end of the sequence input, the output sequence is evaluated by a loss function.
  - The type of loss function depends on the type of RNN used (see the 4 types above). 
    * For example, in a seq to seq RNN, the loss function takes into account all outputs from previous time steps
      and compares them to all actual values (the sequence tokens):
      ```js
      loss(y_0, y_1, ..., y(t) ; y_hat_0, y_hat_1, ..., y_hat_t)
      ```
    * With a seq to vector network only the last output is of interest:
      ```js
      loss(y(t), y_hat_t)
      ```
  - Usually we define the `unroll length` which is how many previous states do we want to take into account in
    the loss function. For example, we can consider only the past 3:
      loss(y(t-2), y(t-1), y(t) ; y_hat_t-2, y_hat_t-1, y_hat_t )

  - The gradients of the loss function are then propagated backwards through the network. In the last example above,
    only the last 3 outputs affect the gradient calculations.
    * Remember that it's the same weight matrices in each time step. When we see the unrolled network, we're
      seeing it over multiple time steps, each with a different input and a different hidden state.
    * At the end of each input sequence we compute the loss and gradients.
    * We update the weights at the end of the batch but there will be multiple gradient updates per sequence.
      - We need to compute the gradients 3 times - one for each output (y(t-2), y(t-1), y(t))
      - Then we sum them up (that's our gradient update) for the input sequence
      - We continue processing input sequences in the batch
      - At the end of the batch we update our weights

Stacking RNN cells into a layer
-------------------------------
* All the examples above discuss a single RNN cell but we can also build a stacked layer of multiple RNNs (units).

* In this case, each one of the RNN cells receive the input and produce an output and a hidden state.
  - For example, assume an RNN layer with 32 units (`tf.keras.layers.SimpleRNN(32, input_shape=[None, 1])`)
  - At time step x(0), each one of the 32 units receives the input and h(0) - the initial hidden state. 
  - Each one of the units is its own RNN cell with weights. Each one produces an output
  - At time step x(1), the input is passed to each one of the 32 units along with each unit's previous hidden state.

* Adding more units into an RNN layer increases it's memory and ability to learn.

* Computations in a single RNN layer can be done in parallel.

* To condense the output of an RNN layer, we can use a dense layer with the desired size.

Stacking RNNs layers into a network
-----------------------------------
* Further extending the RNN network, we can stack these multi unit layers on top of each other.

* In this case (continuing the example above), the 32 hidden states of the first RNN layer are passed as inputs
  to the next RNN layer (in the same time step).
  - In the next time step, we use the result of the respective layer as hidden state.
  - Results of RNN layer n from step 0, are passed to RNN layer n in step 1.

* Adding more RNN layers allows the model learn hierarchical representations of the data.
  - Similar to CNNs, the lower layers pick up on simpler patterns while the higher layers build on them.

Time Series Forecasting
-----------------------
* When working with time series forecasting, it's helpful to create a baseline by doing `naive forecasting` - just copy
  the previous time step into the current one.
  - To forecast time step t, we simply copy time step t-1.
  - When working with seasonal data, it's helpful to copy the data from t-n when creating the naive forecast. n in this case
    is the period of seasonality observed in the data. For example, in the ridership dataset, it's 7 days.
  - When this works and the time series follows the naive baseline well - the time steps from n periods ago have correlated
    values with the current time steps, we say that the time series is `autocorrelated`
  
* Stationary vs non-stationary time series
  - `Stationary`
    * A stationary time series is one whose statistical properties, such as the mean, variance, and autocorrelation, 
      remain constant over time.
    * Stationary time series data is desirable for modeling because it allows for more accurate and reliable 
      statistical analysis, such as forecasting and hypothesis testing. Many time series models, like autoregressive integrated moving average (ARIMA), are designed for stationary data.
  - `Non stationary`
    * A non-stationary time series is one that exhibits one or more of the following characteristics:
      - Trend (upwards or downwards), seasonality, changing variance
    * Non-stationary time series data poses challenges for modeling and analysis because its statistical properties c
      hange with time. Traditional time series models are designed for stationary data and may not work well with non-stationary data.

* `Differencing` is the process of taking the difference between the values from current time step and the ones from n 
  days ago.
  - It's another view for the quality of the naive baseline. You want to see values close to 0 as it indicates that the 
    difference between now and n days ago is minimal.
  - Differencing is a technique to transform a non-stationary time series into a stationary one.
  - For example, a first order differencing of this time series [3,5,7,9,11] yields [2,2,2,2]
  - For example, a second order differencing (taking the differencing twice) of this time series [1,4,9,16,25,36]
    yields [3,5,7,9,11] (first differencing) and then [2,2,2,2] (second differencing).
    * The reason is because this series is quadratic whereas the first one is linear.

* `MAPE` - Mean Average Percentage Error
  - This is a metric for measuring the predictions of a time series model. It's formula:
    ```js
    ape = (actual - prediction) / actual 
    ```
  - Note that we do it for every time step. Then we compute the average of this value by averaging it:
    ```js
    mape = (∑ ape) / n
    ```

Time Series Forecasting - ARMA / ARIMA / SARIMA
------------------------------
* `ARMA` - Autoregressive moving average
  - Computes forecasts using a simple weighted sum of lagged values and corrects these forecasts by adding a 
    moving average:
  ```js
  y_hat(t) = ∑ a_i * y(t-i) + ∑ r_i * e(t-1) for i = 1 to p and q
  where e(t) = y(t) - y_hat(t)
  ```
  * y_hat(t) - forecast at time t
  * y(t) - value at time t
  * The first sum is from i=1 to p. It defines how far back the model should look. The parameters a_i are learned.
  * The second sum is from i=1 to q. 

* This model assumes that the time series is stationary. If not, use differencing.
  - To account for it, this model has several variations. For example, ARIMA and SARIMA.

* ARIMA models can also serve as a "non ML" baseline to an ML model.

Preparing Data for ML Models
----------------------------
* We'll start by using 8 week (56 day) windows for training. This means sliding a 56 day window over the data, taking 
  55 steps for training and the last, 1 value as a label.

Using an RNN cell to predict a sequence
----------------------------------------
* Given an RNN cell as part of a larger network, we can predict the next step in a sequence of elements.

* The Keras SimpleRNN cell works by processing an entire input sequence, step by step before moving to the 
  next layer of the network.
  - Assume that we're using a 2 layer network (RNN, Dense) and had input sequences of 10 time steps each. For each input 
    sequence, the RNN cell processes it step by step (x(0) to x(9)). Only when it's done, it passes the result to the next layer (Dense).
  - If the RNN cell's `return_sequences=False` (default), then it'll return only the last time step result (y_hat(9)). 
    If `return_sequences=True`, then it'll return the results of all time steps (y_hat(0) to y_hat(9)).

* The bottom line, an RNN cell has to process an entire sequence step by step. The output of the cell can be 
  the entire sequence of outputs or a vector (see all 4 types above).

* Notice that the SimpleRNN cell can also be a layer if we specify units > 1. In this case, we'll have n cells
  working on the input sequence concurrently (each RNN cell processing 1 step at a time and the n cells work 
  concurrently).
  - At the end, either each RNN cell in the layer outputs the whole sequence or the last step. 
  - Using multiple RNN cells in a layer increases the degrees of freedom that the network has.

Forecasting Multiple Time Steps Using RNN
-----------------------------------------
* If we want to forecast more than just the next time step (for example, the next 14), we have 2 options:
  - Using a single time step forecast trained model
    * We use the model to predict the next time step, add it to the input and predict the next. We repeat this
      14 times to get the next 14 time steps.
    * The problem with this approach - if there is an error in the first predictions, it'll carry to the next 
      ones as well. Therefore, this can only be used for a small number of steps.

  - Using an RNN that can compute the next 14 values in one shot. It's still a `sequence to vector` model, but one
    that outputs a 14 element vector instead of 1.
    * To do this, we need to prep the data accordingly. We need to change the targets to be vectors containing
      the next 14 values (see notebook).


Forecasting Multiple Time Steps Using a Sequence-to-Sequence Model
------------------------------------------------------------------
* The RNN approach above to forecast multiple time steps at once used a sequence to vector model. It mapped the 
  hidden outputs of each of the 32 RNN cells into a 14 item vector which represented the result.

* Instead, we can train the model to forecast the next 14 values at each and every time step. This will work as follows:
  - At time step 0, the model will output a vector containing steps 1 to 14.
  - At time step 1, the model will output a vector containing steps 2 to 15.
  - And so on.

* As we can see, the targets are sequences of consecutive windows, shifted by 1 time step at each time step.
  - This means that each input sequence needs to have a sequence of windows as output.