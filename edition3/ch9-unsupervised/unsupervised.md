Unsupervised Learning
---------------------

Types of unsupervised learning
------------------------------
* Clustering
  - Grouping similar instances into clusters
  - Great for customer segmentation, recommender systems, search engines,
    image segmentation, semi-supervised learning, dimensionality reduction

* Anomaly detection
  - Learning what's "normal" and then using the that to detect abnormal instances (anomalies).
  - Great for fraud detection, defective product detection, id new trends in time series,
    removing outliers from datasets before training.

* Density estimation
  - Estimating the probability density function (PDF) of the dataset in order to detect
    anomalies.

Clustering Algorithms
---------------------

* In clustering algorithms, we assign instances to groups of similar instances. We may not know what these clusters
  represent (their label if they have one), but we sure know that the instances in each cluster are similar.
  - For example, while hiking you stumble upon a certain flower. You see more of it along the hike. You don't
    know what that flower is, but you can identify more of the same.

* Clustering applications
  - Customer segmentation
    * Segment your customers based on their purchases, activity and demographic information.
    * Use these clusters to adapt products and marketing campaigns 
  - Data analysis
    * When analyzing a new dataset, it's helpful to cluster the data and analyze each cluster separately.
  - Dimensionality reduction
    * After clustering a dataset into k cluster, we can reduce the dimensionality of the dataset by replacing
      each instance's features with its affinity to each cluster.
    * Each instance now has k features.
  - Feature engineering
    * Instead of reducing the data to k features, we can use 1 or all of these k features in addition
      to the existing features of each instance.
    * We did this in ch2. First we clustered the data according to latitude and longitude to get centers.
      Then we added to each instance the center it's the closest to. This was a way to bin the large space
      of all lats and longs in the dataset.
  - Anomaly detection
    * Any instance which is not close to any cluster (by threshold), can be considered an anomaly.
    * For example, we can detect anomalous user traffic in a web application.
  - Semi-supervised learning
    * Assuming that only a part of the dataset is labeled, we can cluster the dataset and propagate the
      labels to instances in the same cluster.
  - Search engines
    * Cluster the data and when a new instance is inputted, see to which cluster it resembles the most
      and return instances from that cluster.

* The definition for what makes a cluster depends on the model. Some models will define a cluster by a center point
  (called a centroid), others will look for continuous regions of densely packed instances. 

K-Means
-------
* K means works by finding each cluster's center and assigning each instance to the closest cluster.

* K means is sensitive to feature scales

* K means can cluster instances in two methods:
  - Hard clustering
    * Labeling each instance with the cluster it belongs to
  - Soft clustering
    * Give each instance a score per cluster. The score can be the distance from the clusters center.
    * This can be used as a dimensionality reduction technique:
      - Start with a dataset of shape: (m,n)
      - Choose k such that k < n
      - Apply k means and get the distance of each instance from each cluster center
      - New dataset has shape: (m,k)
    * The new k features can be used as the entire dataset for dim reduction, or they can be added to the original
      dataset as a way of feature engineering.

* The k means algorithm
  - Start with k random cluster centers 
  - Measure the distance between each point in the dataset to each center
  - Label each point with closest center id
  - For each group of points get the average over all axis - that's the new center
  - Go to step 2 
  - Continue until the centers stop moving

* The k means algorithm has the potential to converge to a local minimum.
  - To overcome this, we can run the model multiple times, each with a different random k initialization centers
    and choose the best solution.
    * In sklearn's KMeans this is the `n_init` parameter.
  - The best solution is measured by `inertia` - the sum of squared distances between the instances and their closest
    centroids.

* An improvement to the k means algorithm called kmeans++ is able to better avoid local minima by adding an 
  extra step to the initialization.
  - It starts by selecting random centers that are distant from each other
  - This makes the algorithm less likely to converge on a local minima
  - The extra computation at the beginning of the process is worth it since we can drastically lower the n_init
    parameter and overall save time.

* Elbow method - Coarse way for finding the optimal number of clusters 
  - If we go by inertia alone, we won't find an optimal solution. This is because the inertia is a metric that will
    continue decreasing as we add more centers.
    * Thinking about it, the more centers there are, the closer each instance to a center.
    * With high k values, you end up with what's supposed to single clusters broken into multiple clusters.
  - A better approach is to plot a function of the inertia with respect to k and take the number of k at the 
    inflection point (the elbow) of the plot.
  

* Silhouette score - precise way for finding the optimal number of clusters
  - For each point in the dataset, we measure:
  ```js
  s = (b-a) / max(a,b)
  ```
  - Where `a` is the mean distance between the point to points in its own cluster
  - Where `b` is the mean distance between the point to points in the closest cluster to the one where the point
    was assigned. The closest cluster is chosen by computing `b` to all other clusters and taking the one that
    returns the minimum `b`
  - This score returns a number between -1 to +1. 
    * -1 means that the point was assigned to the wrong cluster (since `b` is lower than `a`, we know there is a 
      cluster out there that's closer to the point).
    * 0 means that the point is on the border of a cluster
    * +1 means that the point is in the right cluster
  - We compute the silhouette score for all instances and take the average (?). Meaning that a greater silhouette score
    is better.
  - This metric is more informative than the elbow one (which uses inertia) since higher number of k won't 
    return a higher silhouette score.
    * Thinking about it many small clusters produce lower inertia (lower sum of squared distance to each center).
    * But it won't generate higher silhouette score. The distances of the score will get smaller as well. In other words,
      this metric rewards larger clusters that capture their instances well.

* K means limits
  - As we saw, it can easily converge to a local minima so multiple runs using randomized starting centroids is required.
  - K means has issues clustering data when the clusters have different densities, sizes or non-spherical shapes.


Using clustering for image segmentation
---------------------------------------
* Image segmentation has several variants:
  - Color segmentation
    * Pixels of the same color get assigned to the same segment
    * It's rough and simple but sufficient for many applications (e.g. satellite image analysis)
  - Semantic segmentation
    * Pixels of the same object get assigned to the same segment
    * Requires a model that can "understand" the image. Results in a more precise segmentation
  - Instance segmentation
    * An extension of semantic segmentation. If there are multiple instances of the same object, they each 
      get their own segment.
    * For example, multiple "pedestrians" are understood by the model to be multiple instances of a pedestrian.

* K means allows us to do color segmentation. The more complex types require a deep neural network.

Using clustering for semi-supervised learning
----------------------------------------------
* In this case we start with a dataset which is not labeled. For the rest of this section, assume an image
  dataset of 1400 images.

* First we run K means on the dataset with a k=50 (this is a hyperparameter of the training pipeline)
  - We get back a matrix of shape (1400,50)
  - It represents the distance of each image to one of the 50 clusters

* Next we want to get a representative image for each of the 50 clusters
  - That's a single image per one of the 50 clusters which is the closest to the cluster center
  - We end up with an array of 50 indices to images closest to the cluster centers
  - These are the representative images we'll use to label the rest of the dataset

* Next we display these images (or in the case of other types of data, understand what label they belong to).
  - We label them manually into an array
  - Now we have 50 labeled images - `representative_array`

* Next we want to propagate these labels to images in the same cluster
  - For each cluster i:
    * We get the indices of all images in cluster i
    * We label these with the label that's in `representative_array[i]`
  - At this point we have the dataset labeled and can run training

* Another step for improving performance is to remove outliers
  - We want to remove all images that are the furthest from the cluster center for each cluster.
  - For each cluster i:
    * Calculate the cutoff distance of a chosen percentile (e.g. what's the 99th percentile cutoff; the cutoff after
      which there are only 1% of instances)
    * Replace these instances distance with -1
  - Remove all instances who's distance is -1 
  - Retrain model


* Sklearn has classes that do the above process:
  - LabelSpreading
  - LabelPropagation
  - SelfTrainingClassifier
    * Works differently but achieves the same purpose


Density-based spatial clustering of applications with noise - DBSCAN
--------------------------------------------------------------------
* An algorithm that finds clusters by defining regions of high density

* How it works:
  - Start by finding instances that have neighbor instances within a small (epsilon) distance.
    * These are called the epsilon neighborhood of the instance.
  - If the instance has a `min_samples` of neighboring instances in it's epsilon neighborhood it's
    considered a core instance.
  - All instances in the neighborhood of a core instance belong to the same cluster. This neighborhood
    may contain other core instances, all forming a single cluster.
  - Instances that are not core instances or that do not have one in their neighborhood are considered
    an anomaly.

* DBSCAN has no predict method, however, we can take its results and train a classification model.


Gaussian Mixture Models (GMMs)
------------------------------
* A probabilistic model which assumes that instances in the data were generated from a mixture of several
  Gaussian distributions with unknown parameters.
  - All instances generated from a single Gaussian distribution form a cluster.

* Covariance matrix 
  - Calculates the mean distance between any variable to the mean of another variable
    * In the same way that a distribution of a single independent variable as a standard deviation which 
      measures the distance between that variable to the mean, a covariance matrix works when there are
      more than 1 independent variables. 
    * It measures the distance between each variable to the mean of itself (std) and other variables
  - For example, assume a multivariate distribution with variables X1 and X2:
    * We can compute a 2x2 covariance matrix as follows:
      ```js
      Cov(X1, X1) = (1/n) * Σ((X1_i - μ_X1)^2)
      Cov(X2, X2) = (1/n) * Σ((X2_i - μ_X2)^2)
      Cov(X1, X2) = (1/n) * Σ((X1_i - μ_X) * (X2_i - μ_X2))
      Cov(X2, X1) = (1/n) * Σ((X2_i - μ_X2) * (X1_i - μ_X1)

      // The covariance matrix
      Cov(X1, X1), Cov(X1, X2)
      Cov(X2, X1), Cov(X2, X2)
      ```
      - Notice that the diagonal of this matrix is the standard deviation of each variable.
      - Notice that the non-diagonal cells are a product of the stds of each variable.
      - The covariance matrix indicates whether the two variables tend to move together or in opposite directions.
        * A positive covariance of two variables indicates that as one grows the other does as well, a negative
          covariance indicates the opposite (negative relation).

Simple GMM
----------
* In this variant of GMM, the data is assumed to have been generated through this probabilistic process:
  - Each instance `xi` is assumed to have been sampled from one of `k` gaussian distributions.
    * The probability to sample `xi` from distribution `j` is the distribution's weight - `wj`.
  - This distribution has a mean `mj` and a covariance matrix `Cj`. The location of `xi` is sampled 
    from this distribution as follows - `xi <- (mj, Cj)`

* To use this model, we start by taking a dataset and estimating:
  - Number of Gaussian distributions k that make up the dataset (defined by user)
  - Weights of the k distributions (learned by model)
  - Mean and covariance matrix of the k distributions (learned by model) 

* Once the model fits the data to the k distributions we defined, we can take new instances and get the 
  probability that they belong to one of the distributions.
  - This is done by testing the new instance against each of the k distributions:
  ```js
  // Bayes theorem
  P(k | xi) = (P(xi | k) * P(k)) / P(xi)
  ```
  - `P(k | xi)` - probability to get distribution k given xi
  - `P(xi | k)` - probability to get xi given distribution k. To get this we use the PDF of dist k
  - `P(k)` - probability of dist k, this is the weight of dist k
  - `P(xi)` - probability to get xi out of all k distributions. Calculated as the sum of `P(data_point | k) * P(k)`
    over all k.

* Since we're building gaussian distributions, we can also sample new instances from these distributions.

Using GMMs for anomaly detection
--------------------------------
* Applying GMMs for anomaly detection is simple, for any instance, we can get the density of the PDF it belongs
  to at its location.

* We define a density threshold under which an instance is considered an anomaly
  - For example, 2%. This means that if the instance falls in a density area where only 2% of the instances 
    fall, it's considered an anomaly
  - If we get many false positives (model said anomaly and it's not), we can lower the threshold
  - If we get many false negative (model missed anomalies), we can raise the threshold

* If the data has too many anomalies, the model will try to fit all of them and affect the fit.
  - To handle this, fit the model once, remove the outliers
  - Then fit the model again
  - This can also be used for `novelty detection`. In this technique, we try to understand if an instance is
    considered novel. 
    * This type of task assumes that the data has no outliers or that the novel data is different than the 
      dataset completely (even from the outliers).
    * For example, detecting a new type of fraud that was never seen before.

Finding the optimal number of clusters when using GMM
-----------------------------------------------------
* Unlike knn, we can't use the elbow (average distance of all points to a cluster) or silhouette score (distance
  between points to own cluster and neighbor cluster). It's unreliable since clusters are not always the same size.

* To get the optimal number of clusters, we can use the BIC calculation:
  ```js
  BIC = log(m)*p - 2*log(likelihood_function_max)
  ```
  - m: number of instances
  - p: number of parameters learned by the model
  - likelihood_function_max: maximized value of the likelihood function

* We compute the BIC of a GMM model for different k values. 
  - Lower BIC values indicate a better model

* Likelihood vs probability
  - Probability
    * How plausible that we get the value x, given a PDF function with known STD and mean
  - Likelihood
    * How plausible that we get STD and mean of certain values, given X.
    * Maximizing the likelihood function means - finding the sets of parameters that give us the maximum
      likelihood given X.
  - Summary
    * Likelihood - how well a particular model or hypothesis explains observed data
      - Fitting a model to the data.
    * Probability - how plausible is it that i'll see a certain  result given a known probability distribution 
      and its parameters
      - Getting an inference from a given model to a set of observations




