Dimensionality Reduction
------------------------
* Curse of dimensionality - having too many features makes learning slow and also make it difficult
  to find optimal solutions.

* Reducing dimensionality is a technique for reducing the number of dimension a dataset has.
  - For example, in the MNIST dataset, the pixels on the border are always white and they do not contribute 
    any information. Moreover, two neighboring pixels are always related and can potentially be merged 
    into a single pixel (by taking their mean).

* In addition to improving training time and quality, dimensionality reduction can also help with visualizing 
  high-dimensional dataset.

The curse of dimensionality
---------------------------
* In a 1x1 2D surface, any point has a 0.4% chance to being along the border. In a 10,000 dimensional surface, any point
  has 99.99999% chance to being along the border.
  - The intuitive reason for it is that there are a lot more borders to be close to.
  - In the context of ML it's telling us that the higher the space, each data point becomes an extreme in one or
    more dimensions.

* Another fact, the average distance between two random points in a 2D surface is 0.52. In a 3D surface it's 0.66.
  In a 10,000 dimensional hypercube it's 408.25.
  - The higher the dimensionality the more distant the points are in that space.
  - In the context of ML it's telling us that the higher the dimensionality of a dataset, the more sparse it is.
    New instances will have less chance to be close to training instances and the extrapolation a model has to
    do is greater. 
  - A model is more prone to overfitting in higher dimensional datasets since the there is so much more space in 
    a higher dimensional dataset. The model learns well the parts of the space that are in the training set, but
    the rest of the space is not learned well.

* Two solutions for dealing with a high dimensional dataset:
  - Increase the training set
    * This increases the density of the high dimensional space.
    * It's a great approach, the problem with it is - the higher the dimension, the amount of training instances
      required to fill the space grows exponentially. Also, more training instances are not usually readily available.
  - Reduce the dimensionality of the dataset.
    * What this chapter is all about...

Main approaches to dimensionality reduction
-------------------------------------------

Projection
----------
* In reality, points of a dataset are not spread uniformly across the space. Instead, they cluster around a lower
  subspace of the high-dimensional space.
  - For example, in a 3D dataset, points will be close (above, below and around) a 2D subspace (plane).

* We can use this fact and project the points to the subspace, reducing the dimensionality of them.

* Projection is a special type of linear transformation which (like any linear transformation) maps vectors
  from one space to another. The main difference is that projection maps the vectors but also maintains
  two properties while doing so:
  - The vectors in the subspace retain their direction
  - As a result, the vectors in the subspace are orthogonal to the original vectors
  - The projection process is idempotent. Applying it twice to the same matrix yields the same result:
  ```js
  P(P(X)) = P(X)
  ```

Manifold Learning
-----------------
* Not all datasets in high dimension hold the properties described above. There are cases in which points are not
  close to a given subspace to be projected to (see swiss roll dataset).
  - In those cases, simple projection to a lower subspace will cause too much information about the data to be lost.

* A d-dimensional manifold is a part of an n-dimensional space (where d > n) that locally resembles a d-dimensional 
  hyperplane.
  - In manifold learning, the dim reduction algorithm learns how to find the manifold on which the data lies.


PCA - Principle Component Analysis
----------------------------------
* A dim reduction algorithm that works by projecting points into a hyperplane. It identifies the hyperplane that lies
  closest to the data and projects the data onto it.

* Preserving the variance
  - When projecting information to a hyperplane, we are bound to lose some of it (less space to represent the data).
  - The idea is to find the hyperplane that will allow us to retain as much information.
  - More concretely, we need to preserve as much variance of the data as possible.

* Principle components
  - PCA finds the axis that accounts to the largest amount of variance in the training set.
  - It also finds a second axis, orthogonal to the first one, that accounts for the largest amount of the 
    remaining variance. 
    * In a 2D space, you have a main line and a single orthogonal line
    * In a 3D space, you have a main line, an orthogonal line to it and another that's orthogonal to both
      previous axes.
    * In an n-dimensional space, you have n lines where the n-1 line is orthogonal to the nth line, the 
      n-2 line is orthogonal to n and n-1 line and so on.
  - The ith axis is called the ith principal component of the data.

* Finding principal components
  - Using singular matrix decomposition we can find those PCs:
    ```js
    X = U @ E @ Vt
    ```
    - A matrix X can be decomposed into 3 matrices U,E,Vt.
    - Vt contains the unit vectors that define all the principal components we're looking for.

* Projecting down to d dimensions
  - Once we found the principal components, we can reduce the dimensionality down to d dimension by projecting 
    it onto the hyperplane defined by the first d principal components.
  - This ensures that we project the points to a hyperplane and preserve the most amount of variance.
  - Projecting the data using the PCs can be done by matrix multiplication:
  ```js
  X_proj = X @ W_d
  ```
  - X is the original dataset
  - W_d is a matrix containing the first d principal components

* Variance ratio
  - The percentage of variance that lies along each PC
  - For example, if the matrix Vt contains 3 PCs (for a 3 dim space), an example variance ratio could be:
    [0.76, 0.15, 0.09]
  * Which tells us that 76% of the variance is on the first PC, 15% on the second and 9% on the third.

* Instead of choosing the number of dimensions to reduce down to, it's better to choose the number of dimensions
  that add up to a sufficiently large portion of the variance (e.g. 95%).
  - An exception to this rule would be if you were reducing dims for visualization purposes.

PCA for Compression
-------------------
* Notice that when we reduced the MNIST dataset, it went from 784 to 154 dimensions and we lost 5% of the variance.
  - That's an ~80% compression with ~5% information loss.

* We can go back from the reduced dataset to the original one by doing the inverse of the PCA transform.
  - We won't get back the original dataset due to the variance lost, but it'll be close.
  - The mean squared distance between the points of the reconstructed data to the original data is referred to 
    as the `reconstruction error`.


Incremental PCA
---------------
* One issue with all the PCA computations we're seeing, is that they require the entire training set to fit in 
  memory. Incremental PCA solves this issue.

* With incremental PCA we can fit the PCA model in mini-batches.


Random Projection
-----------------
* For datasets with many features (e.g. images), PCA can be very slow if we try to use it to reduce dimensionality.

* Random projection is another technique which performs better.

* It uses an equation called Johnson-Lindenstrauss equation which tells us:
  - Given the size of the dataset - `m`
  - And given a distance between any two points that you would like to have as a threshold
    (in the reduced space) - `eps`
  - It outputs the number of dimensions you can reduce to without going above the distance specified:
  ```js
  d >= 4 * log(m) / [(0.5 * eps^2) - (0.3 * eps^3)]
  ```

* Once we have `d`, we generate a random matrix `P` where each item is sampled from a Gaussian distribution.
  - The dims of it are (d, n) - where n is the number of features of the original dataset
  - We divide it by sqrt(d)
  - Now we can multiply it by the dataset to get the reduced dataset.

* This process is called Gaussian Random Projection
  - There is no fitting at all since the number of dimensions to reduce to is known by using a closed form 
    formula to compute it.

* sklearn's `GaussianRandomProjection` does exactly that process
  


  
