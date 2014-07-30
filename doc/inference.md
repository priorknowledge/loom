# Loom's Inference Method

Loom uses subsample-annealed MCMC to infer a cross-categorization.
Specifically, loom uses 5 different inference kernels, interleaved:

* categorization of row into groups, within each kind
* categorization of features in to kinds
* feature hyperaparameters
* "clustering" hyperparameters for the Pitman-Yor categorization of rows
* "topology" hyperparameters for the Pitman-Yor categorization of features

We describe each method in detail.

## Category inference

Single-site Gibbs sampling.

## Kind inference

Block Algorithm 8.

## Hyperparameter inference

Coordinate-wise Grid Gibbs for most models.

Auxiliary kernel for Dirichlet-Process-Discrete.

## Subsample Annealing

Loom uses subsample annealing to improve mixing with large datasets.
Subsample annealing is much like single-site Gibbs sampling,
but progressively adds data while doing single-site Gibbs sampling on its
current subsample of data.
