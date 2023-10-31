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
  - x_t: the input at time t
  - y_t-1: the output of the neuron from the previous time step

* Keep in mind that both inputs are vectors when we consider an entire layer of recurrent neurons.
  - This means that for each layer we have two weight matrices: `W_x` and `W_y`
  - We can represent the entire input into the layer as `X_t` and `Y_t-1`
  - We can represent the output of a recurrent layer with the following equation:
  ```js
  Y_t = ReLU(X_t @ W_x + Y_t-1 @ W_y + b) 
  ```
  - ReLU is just one option for an activation function. It can be any.
  - `Y_t` is an `m x n` matrix containing the layer's output at time step t for each instance in the
    mini-batch (m).

* Note the recurrence relation in that equation:
  - Y_t is a function of X_t and Y_t-1
  - Y_t-1 is a function of X_t-1 and Y_t-2
  - Y_t-2 is a function of X_t-2 and Y_t-3
    * and so on...
  - This makes Y_t a function of all inputs since time t=0 (X_0, X_1, ... , X_t)

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
      loss(y_0, y_1, ..., y_t ; y_hat_0, y_hat_1, ..., y_hat_t)
      ```
    * With a seq to vector network only the last output is of interest:
      ```js
      loss(y_t, y_hat_t)
      ```
  - Usually we define the `unroll length` which is how many previous states do we want to take into account in
    the loss function. For example, we can consider only the past 3:
      loss(y_t-2, y_t-1, y_t ; y_hat_t-2, y_hat_t-1, y_hat_t )

  - The gradients of the loss function are then propagated backwards through the network. In the last example above,
    only the last 3 outputs affect the gradient calculations.
    * Remember that it's the same weight matrices in each time step. When we see the unrolled network, we're
      seeing it over multiple time steps, each with a different input and a different hidden state.
    * At the end of each input sequence we compute the loss and gradients.
    * We update the weights at the end of the batch but there will be multiple gradient updates per sequence.
      - We need to compute the gradients 3 times - one for each output (y_t-2, y_t-1, y_t)
      - Then we sum them up (that's our gradient update) for the input sequence
      - We continue processing input sequences in the batch
      - At the end of the batch we update our weights

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

* 