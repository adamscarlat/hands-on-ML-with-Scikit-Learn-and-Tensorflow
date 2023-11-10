Handling Long Sequences
-----------------------
* When we have long sequences (more steps per sequence), we start seeing two issues:
  - Unstable gradients
  - Loss of memory for items in the sequence

Unstable Gradients
------------------
* When the input sequences to RNNs are long, the back propagation through time (BPTT) needs to consider more
  time steps when it computes gradients.
  - To understand why longer input sequences create unstable gradients, consider a simple sequence-to-vector 
    RNN cell which outputs only the last hidden state y_hat(t).
  - When computing the loss of this RNN cell we do:
    ```js
    loss(y_t, y_hat(t))
    ```
  - The gradient of this loss has to take into account the weights and biasses (which are the same for all time steps) and 
    also how every previous hidden state affected the result. All of the previous time steps are taken into account 
    in the chain rule. This leads to an unstable gradient if we have many previous time steps.
  - In an RNN which outputs longer sequences, we repeat this process and compute gradient updates for each y_hat. Then
    we aggregate them.

* When choosing activation functions for RNNs, we prefer a saturating activation function (one where the derivative is
  0 for very large or very small inputs).
  - The reason is that RNNs have the problem of an exploding gradient (rather than a vanishing one). 
    * I am not sure why. The book says that it's related to the fact that a single RNN cell has the same weights for
      all time steps.
  - Some popular saturating activation functions are: tanh and sigmoid.

* Batch normalization has been shown to not help much with RNNs.

* Layer normalization (LN) has been shown to work well with RNNs to fix unstable gradients.
  - With LN, at the end of each time step, after we get the output at time t (using the input x(t) and previous state
    y_hat(t-1)), we normalize the output.

Tackling the Short-Term Memory Problem
---------------------------------------
* Another big issue with RNNs is that when they process steps in a sequence, they lose track of the states from 
  the beginning of the sequence.
  - This causes issues when the input sequences are longer, usually more than 10 items.
  - This issue is a symptom of the vanishing gradient problem. To understand why, consider how the gradients in an
    RNN cell are updated:
    * The RNN cell takes as input each time step x(t) and previous state y(t-1). 
    * At the end of the sequence gradients are computed. The longer the sequence, the more probable it will be
      for the gradients to be unstable at the lower steps. These steps are related to the first items of the 
      sequence. This is directly related to the chain rule which destabilizes the gradient values for long
      chains.
    * Therefore, items in the beginning of the sequence will affect the training less than items in the end of
      the sequence.

LSTM Cells
----------
* Long Short Term Memory (LSTM) is a specialized RNN cell that can handle the long term memory loss.
  - In contrast to the RNN cell, which has a single path for past memories, the LSTM cell has two paths,
    one for short term memories and one for long term.

* The anatomy of an LSTM cell:
  - Inputs:
    * c(t-1)
      - Long term state.
    * h(t-1)
      - Short term state.
    * x(t)
      - Input at time step t
  - Outputs:
    * c(t)
    * h(t)
    * y_hat(t)
  - Gates:
    * g(t)
      - General purpose gate. This has a set of weights for the input x(t) and a set of weights for the 
        previous short term state h(t-1). 
      - It does something similar to the simple RNN cell:
        ```js
        g(t) = tanh(x(t) @ W_g + h(t-1) @ W_g + b_g)
        ```
      - Main difference between an LSTM cell and a simple RNN cell is that g(t) is not sent out as the current cell
        output as in a simple RNN cell. Instead, it's multiplied element-wise with the gate output i(t) and added to 
        the c(t-1) (see below).
    * i(t)
      - Input gate. Regulates the result of the g(t) gate:
        ```js
        i(t) = logistic(x(t) @ W_i + h(t-1) @ W_i + b_i)
        ```
      - Because it's using the logistic function, its outputs are between 0 and 1. Outputs closer to 0 reduce 
        the results of g(t) and outputs closer to 1 keep the results of g(t).
      - This gate controls which parts of g(t) should be added to the long term state c(t-1).
    * f(t)
      - Forget gate. Does the same computation as the input gate but the output of it is element wise multiplied with the
        previous long term memory:
          ```js
          f(t) = logistic(x(t) @ W_f + h(t-1) @ W_f + b_f)
          ```
      - Similar to i(t), it regulates the long term memory c(t-1), reducing its values when the result of the logistic 
        function is closer to 0, keeping them when the result is closer to 1.
    * o(t)
      - Output gate. Controls which parts of the long term state should be added to h(t) and y(t), the other outputs 
        of the cell. It's computation is similar to the other 3 logistic gates but its result is used differently:
        ```js
        o(t) = logistic(x(t) @ W_o + h(t-1) @ W_o + b_o)
        ```

* Gate computations:
  - First we compute the forget gate f(t) and regularize the long term memory c(t-1) that's inputted into the cell:
    ```js
    c_temp = f(t) x c(t-1)
    ```
  - Next we regularize the general cell input g(t) using i(t) and add the new memories to the long term memory:
    ```js
    c(t) = g(t) x i(t) + c_temp
    ```
  - Next we add the new long term memory c(t) to the cell's output and hidden output while regularizing with the o(t)
    gate:
    ```js
    h(t), y_hat(t) = tanh(c(t)) x o(t)
    ```

* Notice that every time we take the sigmoid and multiply, we regularize how much memory (short or long, depending on the 
  gate) we want to keep.

* What in the heck?
  - Since each ones of these gates has its own set of weights, each gate can get specialized in a different task 
    during training. For example, the input gate can recognize which input should be stored in long term memory.
  - Regarding the unstable gradients:
    * LSTM decouples cell memory (typically denoted by c) and hidden layer/output (typically denoted by h), 
      and only do additive updates to c, which makes memories in c more stable. Thus the gradient flows through c 
      is kept and hard to vanish (therefore the overall gradient is hard to vanish). However, other paths may cause gradient explosion.

* A variant of the LSTM cell is the GRU cell. It's a simpler version of LSTM which performs just as well of not better.

* These variants of RNN cells can process longer sequences of 100 or more time steps.