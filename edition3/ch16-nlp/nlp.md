Generating Text Using Character RNN
-----------------------------------
* In the first example we generate text by predicting the next character given an initial seed of characters.

* First step is to vectorize the text into numerical form. For this we'll index each character using a number.
  - We use Keras `TextVectorization` layer for that.

* Then we create a labeled dataset by sliding a window over the vectorized text:
  - A training example can be: "to be or not to b"
  - With a label: "o be or not to be"

* The model we use in the example has 3 layers:
  - `Embedding`, to map each character from an integer in range (0-38) to a vector of size 16.
    * This helps the network learn relations between character better since the embedded vector can capture
      more semantic meaning than a single number representing the token.
    * For example, the input with shape (1,8): [2,3,2,6,4,22,3,32], will turn into a matrix of size (8,16).
      Each one of the tokens now maps to a 16 element embedded vector.
  - `GRU`, learns the next character recursively based on past tokens
  - `Dense`, outputs probability for the 39 unique tokens

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
  - Reset the state at the end of each epoch (an entire pass over the dataset). This allows the network to start
    fresh and pick up new patterns at the start of a new epoch.

* With a stateful RNN, we need to let the model know the size of the batch (why?)
  - This creates a limitation. After training, the model will only be able to process such batch sizes.
  - To overcome this limitation (if needed), copy the model's weights to a stateless RNN model.

Sentiment Analysis (word level RNN)
-----------------------------------
* The example of textual sentiment analysis using RNNs is a good introduction to tokenization of text at the word 
  level (up until now, we saw only character level tokenization).

* When splitting the text by words, we need to consider the following:
  - Not all languages use spaces as a word separator (e.g German).
  - There are cases in which two words go together (e.g New England)
  - There are cases in which a single word combines multiple words (e.g #ILoveMachineLearning)
  - If these special cases need to be taken into consideration, there are different implementations (e.g Byte 
    Pair Encoding - BPE) implemented in Tensorflow Text and Hugging Face.

* The IMDB dataset we're using is in english and can be parsed using spaces.
  - We'll limit the vectorization to the most frequent 1000 words. This will reduce the number of parameters
    the model needs to learn.


Handling sequences of different lengths
---------------------------------------
* When using the `TextVectorization` layer in Keras to transform text into numerical form (each word is mapped to
  and integer), it also makes all mapped sequences the same length.
  - It takes the longest sequence as a limit and pads all other sequences with zeros to match the limit length.
  - This causes an RNN cell to pick up many incorrect patterns and messes up the cell's memory.

* The way to overcome this is by using `masking` - telling the model to ignore padding while learning.
  - In the embedding layer, we add a parameter called `mask_zero=True`. 
  - This creates a boolean vector (before embedding), where any element that is equal zero gets a False.
    * Remember that the input to an embedding layer is a sequence of integers (e.g [2, 5, 1, 8, 3, 0, 4]) and the output
      is a matrix in which each one of the integers is mapped to an embedded vector.
    * For example, if an embedded layer has dimensions - input_dim: 10, output_dim: 5, the vector above would generate
      a (7,5) matrix.
    * The mask generated would have dimensions (1,7) and would look like this: [1, 1, 1, 1, 1, 0, 1]
  - This mask vector is then passed to subsequent layers and they use it to ignore 0 elements.
    * For example, when a GRU layer checks the mask vector and sees that a certain timestep is zero, it ignores
      it and copies the previous timestep.


Using pre-trained embedded layers
---------------------------------
* We used an embedded layer from scratch in our text based models so far. This layer learned semantic meaning between
  words (and characters) by mapping each one into a vector.
  - Higher-dimensional embedding vectors have more capacity to capture nuanced semantic information. 
    Words can have complex relationships with one another, and higher-dimensional vectors provide more degrees of freedom to encode these relationships accurately.

* We trained an embedding layer on limited textual data (e.g 25K movie reviews), hoping that the model captures semantic
  meaning of all words in the english language.
  - For example, "Amazing" is closer to "Great" than it is to "Bad".

* Since words generally carry the same semantic meaning regardless of the context, it's possible to re-use someone elses
  trained embedded layer in your model.
  - For example, Word2Vec is a model with embedded layers we can reuse. It was trained on a much bigger text corpus.
  - The limitation of this approach is that the context is not taken into account. For example, the word "right" can 
    be related to "left or right" and also to "right or wrong". When we reuse a pretrained word embedding, the 
    context of the input is not taken into account.

* To address this limitation, instead of reusing a pretrained embedded layer, we reuse a part of a pretrained language
  model.
  - These pretrained LLM parts take into account context and embeddings.
  - See example of adding the universal sentence encoder into our model as a layer.



Attention
---------