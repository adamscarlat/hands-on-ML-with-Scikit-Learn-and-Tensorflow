Generating Text Using Character RNN
-----------------------------------
* In the first example we generate text by predicting the next character given an initial seed of characters.

* First step is to vectorize the text into numerical form. For this we'll index each character using a number.
  - We use Keras `TextVectorization` layer for that.

* Then we create a labeled dataset by sliding a window over the vectorized text:
  - A training example can be: "to be or not to b"
  - With a label: "o be or not to be"

* The model we use in the example has 3 layers:
  - Embedding, to reduce the dimensionality of the data from 39 (unique tokens) to 16
  - GRU, learns the next character recursively based on past tokens
  - Dense, outputs probability for the 39 unique tokens

* After we train the model we generate text by iteratively appending the prediction (next char) to the seed
  and predicting over and over again.
  - We also use the concept of `temperature`. This parameter is between 0-1. Higher values will cause the model
    to take the less probable values (and vise versa)
  - To get the temperature:
    * Compute log of probabilities
    * Divide by the temperature value
    * Use the logits as a distribution to draw samples 
  - Dividing by temperatures closer to 1 smooths the distribution, making have closer probabilities. Dividing by 
    values closer to 0 peaks the distribution, making already higher values more probable.

Stateful RNNs
-------------
* All the RNN models we've seen so far do not maintain state between sequences.
  - When the model gets an input sequence, the hidden state is all zeros.
  - The model processes the sequence and updates the hidden state after each element
  - The hidden state from previous step becomes another input along with the current element.
  - At the end of the sequence, before moving to the next sequence, the hidden state is reset to 0

* Stateless RNNs are useful when the sequences are independent of each other.

* With a stateful RNN, we can keep the hidden state from sequence t and use it as an input for sequence t+1.
  - This allows the RNN to have much better long term memory.

* When using a stateful RNN the sequences must be contiguous - sequence t+1 must start exactly where sequence t ended.
  - This means that the window size cannot be 1 anymore (no overlapping sequences like we had above).
  - Also, no shuffling. The order of the sequences in a batch matters.
  - Keep the batch size at 1 and maintain the hidden state between batches.

* 