Model Training and Deployment
-----------------------------

TF Serving 
----------
* TF Serving is a dedicated model server. Helps packaging and serving your trained model while also taking care of model versioning, 
  retraining, scaling and more.

* It's best to include all preprocessing (e.g Scaling) as layers so that we don't make mismatches between model versions and their preprocessing steps later on.
  - Also, we want to save the models in a directory structure where each model version gets a subfolder.

* It's best to run tf serving from a container.

* TF Serving has 2 communication methods:
  - REST API
    * Easy to use
    * Uses text and most of our input is numeric so that becomes inefficient. 
      - For example, the string "15.889988761234" requires 120 bits (8 bits per utf-8 character of that sort)
      - If we were to send it as a numeric value (possible with gRPC) we would have needed 32 bits (float) or
        64 bits (double)
      - The REST API is good for small amount of data where real-time latency is not required
  - gRPC API
    * The tensorflow serving library provides the necessary gRPC constructs (e.g protocol buffers and clients) to use.

* Updating model version
  - When we retrain the model and increment the version subfolder (e.g 0001 -> 0002), tf serving will automatically
    start using the newer model version.
  - It's important to note that tf serving will keep both models in memory for a short amount of time until the new 
    model is loaded and working and all pending request to the older model version have been resolved.
    * When working wil large models this can cause out of memory issues. 
    * See about the option configuring this to unload the first model before loading the new model if needed.
  - If the new model version doesn't work well, simply remove the directory and tf serving will go back to the previous
    model version.


Cloud Deployment Using Vertex AI
--------------------------------
* Vertex AI is GCP's ML/AI service for model training and deployment (equivalent to AWS SageMaker).

* Amongst many features, we can upload the saved model to VertexAI and have GCP serve, load-balance and scale the model
  server as needed.
  - We start by uploading the model to a GCP bucket
  - Then create a VertexAI model upload. Here we specify:
    * The bucket where the model is stored
    * The docker image to server the model
  - Then create an endpoint for the model

* Vertex AI offers a `batch_predict` API for larger prediction batches
  - To use it, we put the instances we want an inference for in a file and send it to the batch API.
  - This happens asynchronously and GCP spawns new nodes for these batch jobs so it can take a second for 
    the setup of a batch job to run.

Running a tensorflow model in the browser (no server)
-----------------------------------------------------
* In case you need to run a model for inference directly in the browser, you can use tfjs.
  - This library loads a tfLITE model (see book on how to create those).

* Some reasons why you would want to do this:
  - Latency 
  - Privacy 

* tfjs also offers training a model in the browser.
  - It integrates with WebGL to get access to GPU cards.
  - This can be used to fine-tune a model locally with user data.

Using GPUs to Speed Up Computations
-----------------------------------  
* Reasons for setting up your own GPU training instance instead of using a service like Vertex AI, SageMaker or
  Kaggle:
  - Data privacy
  - Financial

* A GPU card for NLP and image processing should have at least 10 GBs.

* Tensorflow uses CUDA and cuDNN to control the GPU card.

* Setting up GPU for Mac M1/M2 (without conda):
  - Source: https://medium.com/@sorenlind/tensorflow-with-gpu-support-on-apple-silicon-mac-with-homebrew-and-without-conda-miniforge-915b2f15425b
```bash
brew install hdf5
pip install tensorflow-macos
pip install tensorflow-metal
```

* We can choose devices (CPU/GPU) when doing different tasks:
  - Generally, you want to place data preprocessing on the CPU
  - Neural network operations on the GPU

* By default all operations will be placed on the first GPU device.
  - We can choose which operations are placed on the CPU or other GPU devices.

* Tensorflow takes care of any parallelization of the computation graph itself.
  - TensorFlow identifies independent subgraphs within the computation graph and allocates them to 
    available devices for parallel execution.
  - This is a lower level of parallelism that is handled by tensorflow. It is different than distributed training where
    a model can be split to work on multiple devices within a single machine or multiple machines. Or even have 
    multiple models run in parallel over different subsets of the data (see next section).

* By default, tensorflow parallelizes any operations that can be parallelized (they have no dependency).
  - When sending those operations to the CPU:
    * `Inter-op queue` is used to split operations across cores 
    * `Intra-op queue` is used to split a single operation across cores.
  - With the GPU, cuDNN takes care of splitting the work and the data across GPUs

Training Models Across Multiple Devices
---------------------------------------
* The native parallelism discussed above is only for a single machine with multiple devices (GPUs and CPUs).
  - Tensorflow decides how to parallelize the computation graph for you.
  - In summary, TensorFlow will try to automatically parallelize computation across devices on a single machine, 
    but the use of distribution strategies provides additional control and optimization options, especially for 
    more complex scenarios.

* Tensorflow also offers a way to control how the parallelism works and how to distribute the work across multiple
  machines.
  - For example, we can split a single model across machines (Model Parallelism).
  - Or we can create clones of a model and give each a subset of the data to train on (Data Parallelism).