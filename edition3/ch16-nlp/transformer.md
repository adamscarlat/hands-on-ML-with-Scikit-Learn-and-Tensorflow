Transformers
------------
* Sources:
  - StatQuest: https://www.youtube.com/watch?v=zxQyTK8quyY&t=11s
  - AI Hacker: https://www.youtube.com/watch?v=4Bdc55j80l8

* Transformers are built on top of the concept of encoder-decoders with attention
  - Unlike the encoder-decoders we've seen, this NN architecture does not use any RNN cells. 
  - This allows it to learn longer sequences without the issues or vanishing/exploding gradients which
    affect the long-term memory of the model.

* The job of the encoder is to encode each word in an input sequence into a vector which captures its:
  - Meaning
  - Relationship to the rest of the sentence (context)
  - For example, in the sentence, "I like soccer" the word "like" has a different meaning than in the 
    sentence "I did it like I did the other one".

* The decoder’s role is to gradually transform each word representation in the trans‐ lated sentence into a 
  word representation of the next word in the translation. 
  - This is done given the encoder outputs as inputs.
  - For example, in the translated sentence "me gusta el fútbol", the word representation of the word “el” will 
    end up transformed into a representation of the word “fútbol”. Similarly, the representation of the word “fútbol” will be transformed into a representation of the EOS token.

* Since we're not using RNN based cells, which require a sequence to be processed word for word sequentially, we can 
  process the sequences in parallel. 
  - How can the model learn context and causality if all words a processed independently? by using self-attention (see
    below for details).

Details
-------
* We start with word embeddings for our vocabulary, same as with the encoder-decoders.

* Now we add an `embedding position` sequence to each embedding
  - It's unique for each word embedding. We add it to the embedded vector.
  - It helps the transformer keep track of word order.
  - We didn't need to use this when we used RNNs in the encoder-decoder since the RNNs were processed sequentially
    and every steps output was passed as input to the next one. With transformers, we don't use RNNs and often 
    parallelize the training. Therefore, we need a way to maintain the positional information of the input sequence
    elements.
  - This is a vector that represents the position of a word in the sequence. Think of this vector as a position
    identifier of words in sequences. For example:
    * "Squatch eats pizza"
    * "Pizza eats squatch"
    * Each word here gets an embedded vector. Each vector gets added a positional encoding vector which is unique
      per position. In the example above, assume that the embedding for pizza is [0.9, 0.2]. In the first sentence
      the word pizza is third, so we add a positional vector for the third index. In the second sentence the 
      position is first, so we add a positional vector for the first index. This makes the embedded vector change
      according to the position of the word in the sentence.
  - We use a variation of the sine and cosine functions to introduce positional vectors. These vectors save positional
    information that is absolute (e.g token "pizza" is 3 spaces from the beginning) and relative (e.g token pizza is 
    1 space from "eats").

* Another concept introduced in transformers is `self-attention`
  - This is a mechanism that creates similarity scores for all pairs of words in a sentence.
    * For example, "the pizza came out of the oven and it tasted good"
    * The word "it" talks about the pizza.
    * The self-attention process does a similarity score between all pairs of words in a sentence (e.g [the,pizza], 
      [the, came], [the, out], ..., [it, pizza], ..., [tasted, good])
  - To get the self attention for words in a sequence:
    * Starting from the embedded vectors + their positional encoding, we multiply these positionally embedded vectors
      by a new set of weights. This is called a `query` vector.
    * We multiply the query vector by all the other positionally embedded vectors in the sequence. In this context,
      these vectors are `key` vectors. For example, take the sentence "let's go"
      - "lets" has a positionally embedded vector: [0.2, 0.9]
      - "go" has a positionally embedded vector: [0.3, 0.4]
      - First we create a query for "lets" - [0.2, 0.9] * `weights` = [0.55, 0.67]. 
      - And for "go" - [0.3, 0.4] * `weights` = [0.22, 0.34].
      - Now we multiply lets with "lets" and "lets" with "go".
    * We take each of the scalar results and run it through a softmax function. This normalizes the scalar results. Now
      they are values between 0-1, representing the probabilities that each word is closer to its paired word.
    * We take the probability vector produced by the softmax function and multiply it by the positionally embedded vector
      of the word "lets". These are considered the `values`
      - This part scales the positionally embedded vector with its similarity to other words in the sequence.
    * This new vector has a lot more information in it now:
      - First, it's an embedded vector, representing a word in a geometric, high-dimensional space.
      - It also has positional encoding, capturing its position in the sequence
      - Then it has similarity information to all other words in the sequence captured in it. This is the 
        capturing of the elusive context in the word embeddings.

* Up until now, we talked about the first half of a transformer - the encoder. To generate the output, we 
  proceed to the decoder.
  - The decoder represents the unit that builds the output sequence based on the encoded input sequence.
  - For example, it can be machine translation (same as in the enc-dec example). Or it can be text generation,
    where an input sequence produces an output sequence (training data is set up differently).
  - We'll use the machine translation example. Assume that the input sequence is "let's go" and the output sequence
    is "vamos"

* In the decoder, we process the "to be decoded" sequence similarly to the encoder. We pass it through a self-attention
  process to generate vectors that are:
  - Embedded
  - Positionally encoded
  - Contain context information using a self-attention mechanism

* Now we need to connect the encoder and the decoder self-attended vectors.
  - We do a process of attention between the decoder self-attended vectors (as the query values) to the encoder 
    self-attended values (the key values).
    * Unlike the encoder's self-attention which did a comparison between the query word and all words in the sequence,
      the decoder's self-attention only looks at the words before the current query as keys.
    * We take the self-attended decoder vector, pass them (again) thru a set of weights and multiply each 
      pair of words (between the decoder and encoder this time). We take the softmax and use it to scale
      the vectors.
  - Now we have the `encoder-decoder attention` values. Each word in the decoder sequence is a vector that
    has:
    * Self-attention
    * Attention to the encoder outputs.
  - We still need to take those vectors and get a single output word. Therefore, our next layer is a fully-connected
    layer with as many neuron as our vocabulary.
    * Here we use softmax and argmax to choose the next word.

* Multi-head attention
  - If we put all the steps necessary for attention in a single unit (e.g dot product between queries/keys and the softmax
    function), we can have multiple attention units in a single layer.
  - Each unit will have its own Q,K,V matrices that will allow learning different contexts separately. 
    * For example, in the sentence "I like soccer". One attention head can learn that "like" means "to be fond of"
    * Another attention head can learn that it's a verb in this context.


Decoder-Only Transformers (ChatGPT specific transformer)
--------------------------------------------------------
* Sources:
  - StatQuest: https://www.youtube.com/watch?v=bQ5BoolX9Ag

* The decoder-only transformer starts off using the same embedding and positional embedding steps (as described
  above). 
  - It then moves on to do `masked self-attention` - the same process of attention described in the transformer
    part above. It only computes attention scores for current word i and words with index j <= i (words that came
    before i or i itself).
  - For example: "What is statquest?"
    * What
      - self attention is only with itself
    * What, is
      - "is" has self-attention with itself and with "what"
    * What, is, statquest
      - And so on...

* Now that we have the masked self-attended vectors for each word in the sequence, our decoder only transformer tries
  to guess the next word in the sequence.
  - The actual next word is used as the label.


Hugging Face
------------
* An ML platform that offers open-source tools for NLP, vision and more. At its core there is a transformers
  library that is ready for use.

* To install (the transformers and datasets libraries of Hugging Face):
```bash
pip install transformers
pip install datasets
```