An Encoder-Decoder Network for Neural Machine Translation
----------------------------------------------------------
* Encoder-decoder are RNN models that are a type of sequence-to-sequence model.

* To understand encoder-decoder networks we'll look at a model that can translate from english to spanish.
  - Notice that not all english and their spanish translations sentences have the same length (let's go == vamanos)

* In a machine translation encoder-decoder network:
  - We embed all sequences (encoder and decoder) before using them. When I mention sequences below, these are 
    embedded vectors of the sequences.
  - The encoder is fed an english sentence as input
  - The decoder outputs spanish translations

* The encoder is the part of the model that uses recursive cells (e.g LSTM) to take an input sequence and encode
  it into a `context vector`. 
  - Think of it as an LSTM cell that processes the input sequence, item by item and outputs a sequence at the end
  - The final sequence will serve as input to the decoder. It serves as the initial state of the decoder's LSTM
    cells.

* The decoder decodes the context vector. The output of the encoder - the context vector - is fed into a new 
  set of LSTM cells.
  - In addition to the context vector, the decoder is inputted the spanish sentence as a target.
  - After that, there is a fully connected layer (Dense) which has the softmax activation. The number of 
    units in this layer is equal to the number of words in Spanish. It outputs the probability of the next 
    word.
  - Now comes the trick:
    * During training: we take the next correct word and input it as the next item to the decoder.
      - Plugging in the known words and stopping at the known phrase length is called `teacher forcing`.
    * During inference: we take the highest probability output from the decoder's Dense layer and send it as the 
      next input to the decoder's LSTM cell.
  - Repeat until <EOS> token is inputted (training) or predicted (inference) or max output length is reached (inference).

* By decoupling the encoder and decoder, we can have different lengths.

* When we test the english to spanish enc-dec translation model, we see that it performs ok on short sentences.
  - When we increase the sentence length, the translation becomes incorrect at the later parts of the sentence.
  - We can improve this by increasing the dataset size and the model's complexity, however, there is a more
    effective method - bidirectional RNNs

Bidirectional RNNs
------------------
* All the RNN based cells we've seen so far take the previous state and current input to make a sequence (or vector)
  predictions. It doesn't look into the future.
  - This type of RNNs doesn't fit all seq to seq tasks. For example, it's useful when we need context. Consider the 
    following sentences: "the right arm", "the right person".
  - In both the word "right" means two different things and we can understand it because we read the whole sentence
    and only then translate.

* A bidirectional RNN has two recurrent layers operating on the same input sequence:
  - On RNN layer reads the input sequence from left to right 
  - The other from right to left
  - Then we combine their outputs at each time step (usually by concatenation).

* When we use keras Bidirectional layer to wrap an RNN based cell, keras takes care of the implementation of the
  bidirectional cell (including concatenation) automatically.
  - It creates a clone of the RNN cell in the reverse direction and concatenates both directions step outputs
  - One thing we need to consider is that the cell now outputs 4 sequences instead of 2:
    * Short term, long term of the forward cell
    * Short term, long term of the reverse cell
  - We can't pass these 4 states to the decoder that expects only 2 and we can't make the decoder bidirectional
    since it has the task of translating a sentence. If the decoder can peek into the future, it won't infer well.
    * To deal with it, we simply reduce them into 2 states by concatenating them.

Beam Search
-----------
* This is a technique to increase the performance of an RNN seq-2-seq model without further training. We do it during 
  inference.

* When we generate a sequence for translation, we always take highest probability word predicted by the model, append it 
  to the translated sentence and use it as input to the model (along with the original english sentence).

* Instead of simply taking the highest probability next token, we can take the top `k` next tokens and generate 
  multiple possible translations simultaneously. Then we compute the conditional probability of each one and 
  choose the highest. For example:
  - English: "I like soccer"
  - 3 top predictions (k=3):
    * "me": 75%
    * "a": 3%
    * "como": 1%
  - Next 3 predictions (appended):
    * me
      - gustan: 27%
      - gusta: 24%
      - encanta: 12%
    * a
      - mi: 0.5%
      ...
    ...
  - At the end we have a tree with many branches. We compute the conditional probability of each branch and take the
    highest one:
    * me gustan los jugadores = 0.27 * 0.1 * 0.01 = 0.00027
    * me gusta el futbol = 0.24 * 0.1 * 0.6 = 0.0144
    ...

* Notice the crucial part - even tho the next probable word after "me" was "gustan" that was incorrect, we managed to 
  course correct by having parallel generated sequences and computing the conditional probabilities.

Attention
---------
* In our encoder-decoder model, if we try to translate long sequences, we'll start seeing it breaking down.
  - The encoder's RNN based cells unrolls an entire sequence into a single context vector that is then passed to the
    decoder.
  - Words that are inputted early on are forgotten. For example, "dont eat the delicious looking and smelling pizza".
    In this sentence, if the word "don't" is forgotten, the entire sentence loses its meaning.

* Currently we're sending a single context vector from the encoder to the decoder. This context vector is the final
  hidden state of the encoder. 
  - The main idea of attention, is to send a context vector to the decoder at each step of the sequence. Each step in
    the coder would have direct access to the step outputs from the encoder.
  - In the example sentence above, we'd send the output of the RNN cell to the decoderat: dont, dont eat, 
    dont eat the, dont eat the delicious, etc...
  - These outputs are sent to the decoder.
  - This is the attention - the decoder places attention on the relevant word at each timestep.

* To set the hidden steps of the encoder as inputs to the decoder steps:
  - We take output step i from the encoder and do a similarity score (cosine) to the output of step i from the 
    decoder.
    * For example, Let's go => vamanos
    * "Let's" is embedded and passed thru the encoder at step 1. It's compared to the "SOS" token after it passed 
      thru the decoder
    * Then similarity with "go" and the SOS token
  - Now we use the similarity scores to scale the results of the softmax function. 
  
* In Keras, we can simply add an attention layer after we define the decoder outputs. We pass to it the encoder outputs
  and the decoder outputs. Then we connect to it the final, softmax, Dense layer.
  - Since this layer needs to match encoder steps with decoder steps, we need to make sure that our encoder outputs
    all time steps (return_sequences=True)

