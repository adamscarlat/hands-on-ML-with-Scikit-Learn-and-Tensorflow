Transformers
------------
* We start with word embeddings for our vocabulary, same as with the encoder-decoders.

* Now we add an `embedding position` sequence to each embedding
  - It's unique for each word embedding. We add it to the embedded vector.
  - It helps the transformer keep track of word order.
  - This is a vector that represents the position of a word in the sequence. Think of this vector as a position
    identifier of words in sequences. For example:
    * "Squatch eats pizza"
    * "Pizza eats squatch"
    * Each word here gets an embedded vector. Each vector gets added a positional encoding vector which is unique
      per position. In the example above, assume that the embedding for pizza is [0.9, 0.2]. In the first sentence
      the word pizza is third, so we add a positional vector for the third index. In the second sentence the 
      position is first, so we add a positional vector for the first index. This makes the embedded vector change
      according to the position of the word in the sentence.

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
      of the word "lets".
      - This part scales the positionally embedded vector with its similarity to other words in the sequence.
    * This new vector has a lot more information in it now:
      - First, it's an embedded vector, representing a word in a geometric, high-dimensional space.
      - It also has positional encoding, capturing its position in the sequence
      - Then it has similarity information to all other words in the sequence captured in it 