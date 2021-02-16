# Bayesian Optimization Experiments


## Installation

These experiments use ProBO, which requires Python 3.6+. To install all dependencies for
development, `cd` into this directory, and run:
```
$ pip install -r requirements/requirements_dev.txt
```


## Examples

### Branin40 with SklearnPenn
A first example is tuning a 40 dimensional Branin function using SklearnPenn, a
probabilistic neural network ensemble model implemented with scikit-learn.

In the [`branin40_sklearnpenn_modeldef.ipynb`](branin40_sklearnpenn_modeldef.ipynb)
notebook, the SklearnPenn model is first defined, then BO details are set up (including
starting with 80 uniform random observations), then BO is run for a couple iterations.

The [`branin40_sklearnpenn.ipynb`](branin40_sklearnpenn.ipynb) notebook is the
same, except the SklearnPenn model is imported from [`penn_sklearn.py`](penn_sklearn.py).


### Ackley10 with SklearnPenn
A second example is tuning a 10 dimensional Ackley function using SklearnPenn, a
probabilistic neural network ensemble model implemented with scikit-learn.

In the [`ackley10_sklearnpenn.ipynb`](ackley10_sklearnpenn.ipynb) notebook, the
SklearnPenn model is imported from [`penn_sklearn.py`](penn_sklearn.py), then BO details
are set up (including starting with 80 uniform random observations), then BO is run for
a couple iterations.
