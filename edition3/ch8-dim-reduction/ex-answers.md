1. Dimensionality reduction
* Motivations:
  - Reduce complexity of high dimensional dataset (reduce overfitting)
  - Reduce training time of high dimensional datasets (less data to process)
  - Make high dimensional dataset fit in memory
  - Use for visualization of high dimensional datasets (by reducing them to 2D or 3D)
* Drawbacks:
  - Information loss 
  - Transformed features are hard to interpret
  - Adds complexity 

2. Curse of dimensionality specifies how data becomes more difficult to understand in high dimensions.
  - It shows us how as the number of dimensions increases, instances become further away from each other
    making models overfit to the data they have as new examples are not likely to be close to any given point.
  - In addition, as the number of dimensions increases, there are more instances closer to a dimension, making
    them outliers along those dimensions.

3. Yes, a dataset that has its dimensions reduced can be transformed back to its original form with some 
  information loss. The loss is relative to the loss of variance when the dataset was reduced.

4. No, PCA works best with linear datasets. With non-linear datasets the variance lost will be larger 
  (much information loss) when reducing the data if there are no useless dimensions (swiss roll). If there
  are useless dimensions in a non-linear dataset, PCA will reduce them perfectly.

5. Depends on the data. It could be as many dimensions c1,c2,...cn such that the preserved variance from each 
  of these dimensions c1_var, c2_var,...,cn_var is summed to 0.95:
  
  sum(ci_var) = 0.95 for i=1 to n.

  - for a 100 items dataset, the reduced number of dimensions is a number between 1 to 95.
    * If the instances are perfectly aligned, you could project them down to a line (1D)
    * If the instances are random, it can be that you'll need 95 dimensions.

6. Regular PCA - when datasets can fit in memory and the number of dimensions is relatively low (roughly up to 10K?)
  Incremental PCA - when datasets cannot fit in memory or for online tasks.
  Random PCA - when datasets fit in memory and you want a faster approach than regular PCA

7. To evaluate performance of PCA, you could use randomized hyperparameter search with the PCA's n_components as one
  of the hyperparameters. This will use cross validation and find the best number of PCA components to use.

8. Yes, chaining dim reduction models can help


