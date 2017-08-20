scikit provides the option to write custom data transformers in addition to its existing ones. in order
to match scikit learn pipeline, the class must have fit, transform and fit_transform methods.

it is possible for custom transformers to inherit from the below base classes to get additional functionality:
    * BaseEstimator - inheriting from this class provides automatic hyperparameter tuning
    * TransformMixin - inheriting from this class provides fit_transform

parameters that are supplied via the constructor are considered hyperparameters 