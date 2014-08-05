# Adapting and Extending Loom

[Probabilistic Model](#model)
[Inference Algorithm](#inference)
[Sparsity](#sparsity)
[Dataflow](#dataflow)

## Loom's Model: Cross Cat <a name="model"/>

## Loom's Inference Algorithm <a name="inference"/>

Loom uses subsample-annealed MCMC to infer a cross-categorization.
Specifically, loom uses 5 different inference kernels, interleaved:

* categorization of row into groups, within each kind
* categorization of features in to kinds
* feature hyperaparameters
* "clustering" hyperparameters for the Pitman-Yor categorization of rows
* "topology" hyperparameters for the Pitman-Yor categorization of features

We describe each method in detail.

### Category inference

Single-site Gibbs sampling.

### Kind inference

Block Algorithm 8.

### Hyperparameter inference

Coordinate-wise Grid Gibbs for most models.

Auxiliary kernel for Dirichlet-Process-Discrete.

### Subsample Annealing

Loom uses subsample annealing to improve mixing with large datasets.
Subsample annealing is much like single-site Gibbs sampling,
but progressively adds data while doing single-site Gibbs sampling on its
current subsample of data.

## Sparsity <a name="sparsity"/>

Loom can efficiently handle two types of sparsity in data:

1. Sparsely observed data, e.g., missing fields in forms
2. Sparsely nonzero data, when a column takes a single value most of the time.

Sparsely observed data is handled transparently.
To hadle sparsely nonzero data, loom performs two initial passes over the dataset to (i) find a tare row, and (ii) sparsify the data WRT the tare row.
These pre-passes are performed by default in `loom.tasks.infer`.

The loom inference engine actually supports multiple tare rows, even though
the automatic `tare` process can only produce a single tare row.
Multiple tare rows are useful e.g. in datasets with multiple text fields,
both of which are blown out into many boolean `bb` columns, but each of which
may be independently unobserved / missing.
In this case, you can create the tare rows and sparsify with a custom script.

## Dataflow <a name="dataflow"/>

![Dataflow](dataflow.png)

