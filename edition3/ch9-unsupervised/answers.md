1. Clustering is an unsupervised task for grouping of similar data.
  - Data can be considered similar by measuring geometric distance between points in the data
    * For example, K means measures distance between points in the data to centers where the density
      of points is greater.
  - Data can be considered similar if we assume that it was sampled from a Gaussian distribution (or a set of them).
    * For example, GMM assume that the data was sampled from k distributions and that data points are considered
      similar if they are from the same distribution from areas of high density.

2. Clustering has the following applications:
  - Anomaly detection
  - Customer segmentation
  - Recommender systems
  - Color segmentation (images)
  - Dimensionality reduction (data is reduced to distance to each cluster center)
  - Semi supervised learning (label propagation)

3. When using K means we can choose an optimal number of cluster using:
  - The elbow method
    * Measures the inertia (average distance between all points to cluster centers).
    * The lower the better.
    * Will continue to decrease as we increase the number of clusters, therefore, not as precise. To use it,
      you want to identify a sharp drop in inertia as you increase the number of clusters.
  - The silhouette method
    * Measures the mean distance between each point to points in its own cluster and nearest cluster.
    * Takes the difference of that distance and divides by the greater value.
    * In english, it measures how close each point is to its own cluster vs its nearest cluster
    * If the number of clusters is optimal, this score will be higher since the difference between the distance
      of points to their own cluster and other cluster will be larger.
    * This score is robust to adding more clusters since it takes into account a relative distance not just to its
      own cluster, but a closest neighbor cluster as well.

4. When dealing with non-labeled or semi labeled datasets, we can use clustering (unlabeled technique) to propagate
  the labels to all instances.
  - First we do clustering over the data and get back a fit to some k clusters
  - Then for each of the k clusters, we take the k nearest instances to their cluster center. These are the 
    representative instances.
  - We figure out their k labels (this requires a manual process where a human reviews and understands the label
    of each of these instances). We label each k 
    * For better results, we should use k > num_unique_labels
    * This way multiple clusters can represent a single label if needed
  - Now we propagate these labels to all instances in the same cluster
    * For better results, we first remove the outliers in each cluster
    * We can do that by removing instances that are in the 1% or lower (cluster center distance)

5.Clustering algorithms for large datasets
  - K means and BIRCH

6. Active learning is the process of adding a human to the training loop
  - Assume we have an unlabeled dataset
  - During training, we use label propagation (as described above) to propagate labels
  - We fit the model using these artificial labels
  - Then we validate the model. When we do inference for an instance and get a low confidence score, we send 
    it to a human for checking. 
    * If the label is incorrect, we fix it 

7. Anomaly detection learns to identifies the outliers in a dataset. It can then say if a new instance is an 
  outlier.
  Novelty detection assumes a dataset with no outliers. It then learns to identify instances that do not belong
  to the dataset at all. Can be used to identify new types of fraud, that were never seen before.

8. A Gaussian Mixture is a model that clusters the data using multiple Gaussian distributions.
  - Given a number of Gaussian distributions, the model fits the datapoints to each one of the distributions
  - The model does a maximum likelihood estimation - given the data, what are the optimal parameters and distributions
    that best describe this data.
  - It can then tell us, given a data point, which distribution it belongs to out of the k and what's the density
    at that point (of that distribution).
  - Can be used for anomaly detection and clustering

9. AIC and BIC
  - The lower their score the better the fit.
  - Both models do a maximum likelihood estimation given the data




