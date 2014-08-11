# Adapting and Extending Loom

* [Probabilistic Model](#model)
* [Inference Algorithm](#inference)
* [Sparse Data](#sparsity)
* [Component Architecture](#components)
* [Data Files](#files)
* [Dataflow](#dataflow)
* [Parallelization](#parallel)
* [Developer Tools](#tools)


## Loom's Model: Cross Cat <a name="model"/>

See [mansinghka2009cross, shafto2011probabilistic](/doc/references.bib).


## Loom's Inference Algorithm <a name="inference"/>

Loom uses subsample-annealed MCMC to infer a cross-categorization.
Specifically, loom interleaves 5 different inference kernels to learn:

* categorization of row into groups, within each kind
* categorization of features in to kinds
* feature hyperparameters
* "clustering" hyperparameters for the Pitman-Yor categorization of rows
* "topology" hyperparameters for the Pitman-Yor categorization of features

We describe each inference kernel in detail.

### Category inference

Single-site Gibbs sampling.

The mathematics of the category kernel is simple, since the underlying
collapsed Gibbs math is outsourced to the distributions library.
In pseudocode, the category kernel adds or removes each row from
the `Mixture` sufficient statistics and `Assignments` data:

    for action in annealing_schedule:
        if action == ADD:
            row = rows_to_add.next()                        # unzip + parse
            for kindid, kind in enumerate(cross_cat.kinds):
                value = row.data[kindid]                    # split
                scores = mixture.score(value)               # add: score
                groupid = sample_discrete(scores)           # add: sample
                kind.mixture.add_row(groupid, value)        # add: push
                assignments[kindid].push(groupid)           # add: update
        else:
            row = rows_to_remove.next()                     # unzip + parse
            for kindid, kind in enumerate(cross_cat.kinds):
                value = row.data[kindid]                    # split
                groupid = assignments[kindid].pop()         # remove: pop
                kind.mixture.remove_row(groupid, value)     # remove: update

where `rows_to_add` and `rows_to_remove` are cycling iterators on the shuffled
dataset (a gzipped protobuf stream),
and `assignments` is a list of FIFO queues, one per kind.

Loom parallelizes the category kernel over multiple threads
by streaming rows through a shared concurrent partially-lock-free ring buffer,
or `Pipeline` in C++.
The shared ring buffer is divided into three phases separated by
moving barriers.
Workers in each phase independently process rows within that phase,
and only block when they hit a barrier.
The three phases are:

1. <b>Unzip (2 threads).</b>
   The annealing schedule determines whether to add or remove each row.
   When adding, a row is read from the `rows_to_add` read head
   at the front of Loom's moving window;
   when removing a row is read from the `rows_to_remove` read head
   at the back of Loom's moving window.
   Since this each of these heads is bound to a resource (the zipfile),
   we can parallelize over at most two threads: add and remove.
   Thus we do as little work as possible in this step,
   deferring parsing and splitting.

   <b>Constraints:</b>
   Each row is either added or removed, but not both.
   Add tasks must be performed sequentially.
   Remove tasks must be performed sequentially.
   Add and remove tasks are independent.

2. <b>Parse and Split (~6 threads).</b>
   The next phase is to deserialize the raw unzipped bytes into a protobuf `Row`
   structure, and then split the full row into partial rows, one per kind.
   Since this transformation is purely functional,
   arbitrarily many threads can be allocated.
   We have found 6 threads to be a good balance
   between downstream work starvation and context switching.
   Thread count is configured by `config['kernels']['cat']['parser_threads']`
   and `config['kernels']['kind']['parser_threads']`.

   <b>Constraints:</b>
   Each row must be parsed+split exactly once.
   Rows can be processed in any order.

3. <b>Gibbs add/remove (one thread per kind)</b>.
   The final phase embodies the Gibbs kernel, and depends on whether the current
   row should be added or removed.
   When adding, we score the row;
   sample a category assignment `groupid`;
   push the assignment on the assignment queue;
   and update sufficient statistics.
   When removing, we pop an assignment off the assignment queue;
   and update sufficient statistics.
   This phase is parallelizable per-kind,
   so that very-wide highly-factored datasets parallelize well.
   The bottleneck in the entire kernel is typically the add/remove thread
   for the largest kind (which has to do the most work).

   <b>Constraints:</b>
   Each row must be processed by each kind.
   Within each kind, rows must be processed sequentially.
   Tasks in different kinds are independent.

![Parallel Cat/Kind Kernel](parallel-kernels.png)

The shared ring buffer is sized to balance
context switching against cache pressure:
too few rows leads to worker threads blocking and context switching;
too few rows can cause the buffer to spill out of cache into main memory.
The default buffer size is 255 (= 256 - 1 sentinel).
The buffer size is configured with
`config['kernels']['cat']['row_queue_capacity']` and 
`config['kernels']['kind']['row_queue_capacity']`. 


### Kind inference

Block Algorithm 8.

First note that what is called the `KindKernel` in C++ is actually
a combined category + kind kernel.
The streaming version of the kind kernel needs to manage the category kernel's
operation, and replicates all of the `CatKernel`'s functionality.
This section describes the purely Kind part.

Loom's kind kernel uses a block-wise adaptation of Radford Neal's celebrated
Algorithm 8 kernel for nonconjugate Gibbs sampling
[neal2000markov](/doc/references.bib).
The block algorithm 8 kernel first sequentially builds
a number of ephemeral kinds
and sufficient statistics for each (kind,feature) pair,
including both real and ephemeral kinds.
After sufficient statistics are collected,
the kernel computes the likelihoods of all (kind,feature) assignments,
an entirely data-parallel operation.
Finally the kernel randomly Gibbs-reassigns features to kinds
using the table of assignment likelihoods.
Unlike Neal's algorithm 8, the block algorithm 8 does not
resample ephemeral kinds after each feature reassignment;
indeed Loom's streaming view of data requires batch kind proposal.
In pseudocode:

    kind_proposer.start_building_proposals()                    # kind kernel
    for action in annealing_schedule:
        if action == ADD:
            row = rows_to_add.next()
            for kindid, kind in enumerate(cross_cat.kinds):
                value = row.data[kindid]
                scores = mixture.score(value)
                groupid = sample_discrete(scores)
                kind.mixture.add_row(groupid, value)
                assignments[kindid].push(groupid)
                kind_proposer[kindid].add_row(groupid, value)   # kind kernel
        else:
            row = rows_to_remove.next()
            for kindid, kind in enumerate(cross_cat.kinds):
                value = row.data[kindid]
                groupid = assignments[kindid].pop()
                kind.mixture.remove_row(groupid, value)
                kind_proposer[kindid].remove(groupid)           # kind kernel

        # kind kernel
        if kind_proposer.proposals_are_ready():
            kindi_proposer.compute_assignment_likelihoods()
            for i in range(config['kernels']['kind']['iterations']):
                for feature in features:
                    kind_proposer.gibbs_reassign(feature)       # see below
            kind_proposer.start_building_proposals()

where the `gibbs_reassign` function performs a single-site Gibbs move,
in pseudocode

    class KindProposer:
        ...
        def gibbs_reassign(self, feature):
            self.remove_feature(feature)
            scores = self.clustering_prior() \
                   + self.assignment_likelihoods[feature]
            kind = sample_discrete(scores)
            self.add_feature(feature, kind)

Because this innermost operation is so cheap
(costing about one `fast_exp` call per (kind,feature) pair),
Loom can run many iterations, defaulting to 100,
increasing the proposal acceptance rate per unit of compute time.

The block algorithm 8 kernel is correct only when the number of ephemeral kinds is larger than the number of features;
otherwise the hypotheses of many-kinds-with-few-features are unduly penalized.
In practice, proposals are cheap so we set `config['kernels']['kind']['ephemeral_kind_count'] = 32` by default,
and never run out of ephemeral kinds.

Loom parallelizes the kind kernel in the same way it parallelizes the cat
kernel with two differences.
First, the kind kernel has more kinds (the ephemeral kinds), and is thus
more amenable to parallelization.
Second, the kind kernel performs an extra step of accumulating sufficient statistics in proposed kinds.
Each real kind and each ephemeral kind must update sufficient statistics for each feature.
But in contrast to the cached `ProductMixture` used in category inference,
the `KindProposer`'s `ProductMixture` does not cache scores, and is thus very cheap.

### Hyperparameter inference

Coordinate-wise Grid Gibbs for most models.

Auxiliary kernel for Dirichlet-Process-Discrete.

### Subsample Annealing

Loom uses subsample annealing to improve mixing with large datasets.
Subsample annealing is much like single-site Gibbs sampling,
but progressively adds data while doing single-site Gibbs sampling on its
current subsample of data.


## Sparse Data <a name="sparsity"/>

Loom efficiently handles two types of sparsity in data:

1. Sparsely observed data, e.g., missing fields in forms
2. Sparsely nonzero data, when a column takes a single value most of the time.
   (for boolean/categorical/count valued data, but not real-valued)

### Sparsely Observed Data

Sparsely observed data is handled by ingesting csv data into streams of packed
`ProductValue` protobuf messages.
During inference, loom only looks at the observed columns when computing scores
and updating sufficient statistics.

See [`loom.format.import_rows`](/loom/format.py)
and [`loom.format.export_rows`](/loom/format.py) for implementation.

### Sparsely Nonzero Data

Sparsely nonzero data is handled in by diffing rows against one or more
<b>tare rows</b>, resulting in `ProductValue.Diff` protobuf data structures.
During inference, loom only looks at the diff when scoring and updating
the kind kernel's sufficient statistics, but must look a all tare rows when
updating the cat kernel's sufficient statistics.

The typical application for tare rows is when text fields are blown out to
a large number of boolean present/absent fields, so that most words are missing from most text fields.
Loom has full support for the case when there is a single text field,
or when there are multiple text fields that are always observed; in these cases
a single tare row suffices.
Multiple tare rows are required in the more complicated setting of multiple
text fields which are independently observed (e.g. text field `A` is present
and `B` missing in row `, but `A` is absent and `B` present in row 2, and both
present in row 3).

Loom automatically searches for a single tare row and diffs the data as part of `loom.tasks.ingest`
These two initial passes over the dataset are implemented as
`loom.runner.tare` and loom.runner.sparsify`, resp.
See [differ.hpp](/src/differ.hpp),[.cc](/src/differ.cc) for implementation.

The loom inference engine fully supports multiple tare rows, even though
the automatic `tare` process can only produce a single tare row.
In this case, you can create the tare rows and sparsify with a custom script.

#### Example

Consider sparsifying a single row of a dataset with five boolean features.

1.  Original CSV Row.
    Features 0, 1, 3, and 4 are observed; feature 2 is unobserved.

        false,true,,false,false

2.  The imported Row after `loom.format.import_rows`
    has a `DENSE` `diff.pos` field and an empty `NONE` `diff.neg` field.

        {
            id: 0,
            diff: {
                pos: {
                    observed: {
                        sparsity: DENSE,
                        dense: [true, true, false, true, true],
                        sparse: []
                    },
                    booleans: [false, true, false, false],
                    counts: [],
                    reals: [],
                },
                neg: {
                    observed: {sparsity: NONE, dense: [], sparse: []},
                    booleans: [],
                    counts: [],
                    reals: [],
                },
            }
        }

3.  After running `loom.runner.tare` on the entire dataset,
    we might find that features 0 and 1 are sparsely-nonzero,
    both with most-frequent value `false`.

    The tare value is:

        {
            observed: {
                sparsity: DENSE,
                dense: [true, true, false, false, false],
                sparse: []
            },
            booleans: [false, false],
            counts: [],
            reals: []
        }

4.  After running `loom.runner.sparsify`,
    the original row will be compressed to contain only differences from
    the tare row.

        {
            id: 0,
            diff: {
                pos: {
                    observed: {
                        sparsity: SPARSE,
                        dense: [],
                        sparse: [1,3,4]
                    },
                    booleans: [true, false, false],
                    counts: [],
                    reals: [],
                },
                neg: {
                    observed: {
                        sparsity: SPARSE,
                        dense: [],
                        sparse: [1]
                    },
                    booleans: [false],
                    counts: [],
                    reals: [],
                },
            }
        }

    * Feature 0 `false` agrees with the tare value `true`
    * Feature 1 `true` differs from the tare value `false`
    * Feature 3 was unobserved in both the example row and the tare value
    * Feature 4 `false` differs from the tare value of unobserved
    * Feature 5 `false` differs from the tare value of unobserved

## Component Architecture <a name="components"/>

Loom is organized as a collection of high-level python modules
wrapping a collection of C++ stand-alone utilities.

Within C++, the lowest-level data structures are mostly provided by
protocol buffers in [schema.proto](/src/schema.proto) (
notably `ProductValue`, `ProductValue::Diff`, and `Row`),
or by `Mixture` objects from the
[distributions](https://github.com/forcedotcom/distributions) library.
On top of these basic structures, loom builds `ProductModel` and `ProductMixture` data structures for Dirichlet Process inference.
The `CrossCat` structure holds a factorized collection of `ProductModel`,`ProductMixture` pairs, one pair per kind.
Finally, the `Loom` object wraps a `CrossCat` object for hyperparameters and sufficient statistics, plus an `Assignments` object for row-category assignments.
During kind inference, the kind kernel builds a `KindProposer` object
that has a collection of ephemeral kinds for the block algorithm 8 kind kernel;
the `KindProposer` is analogous to the `CrossCat` object,
but with different caching strategies and with all kinds seeing all features.

The python-C++ binding layer is in [runner.py](/loom/runner.py)
where the top-level C++ executables are run via python subprocess.
In particular, loom does not use extensions/boost::python/cython to bind C++.


## Dataflow <a name="dataflow"/>

When debugging dataflow issues, it is handy to be able to look at files.
Loom provides a `cat` command that tries to decompress + parse + prettyprint
files based on their filename

    python -m loom cat FILENAME     # parses and pretty prints file

![Dataflow](dataflow.png)


## Parallelization <a name="parallelization"/>

Loom uses four techniques for parallelization:

1.  Parallelizing inference per-sample using python multiprocessing.
    `loom.tasks.infer` parallelizes inference tasks over
    multiple inference processes.
    In distribututed systems, `loom.tasks.infer_one` can be run per-machine.
    You can configure the number of workers with the `LOOM_THREADS`
    environment variable.
    See `parallel_map` [util.py](/loom/util.py) for the abstraction and
    [tasks.py](/loom/tasks.py) for usage.

2.  Parallelizing hyperparameter kernels per-feature using openmp.
    You can configure this with `config.kernels.hyper.parallel`.
    See `HyperKernel::run` in
    [hyperkernel.cc](/src/hyper_kernel.cc) for implementation.

3.  Parallelizing the kind and category kernels
    using a shared concurrent partially-lock-free ring buffer.
    See the [inference section](#inference) above for details.
    See `Pipeline` in [pipeline.hpp](/src/pipeline.hpp) for the abstraction and
    [cat_pipeline.hpp](/src/cat_pipeline.hpp)|[.cc](/src/cat_pipeline.cc) and
    [kind_pipeline.hpp](/src/kind_pipeline.hpp)|[.cc](/src/kind_pipeline.cc)
    for usage.

4.  Vectorizing low-level math using SIMD operations.
    This is outsourced to the
    [distributions](https://github.com/forcedotcom/distributions) library.

In addition, loom uses openmp to parallelize other simple operations like
loading files and precomputing computation caches.


## Developer Tools <a name="tools"/>

### Debugging

You can inspect most of loom's intermediate files of these files with

    python -m loom cat FILENAME     # parses and pretty prints file

You can watch log files with

    python -m loom watch /path/to/infer_log.pbs

When debugging C++ executables run through `loom.runner`,
you can turn on debug mode usually with a `debug=true` parameter,
and replicate the command that `loom.runner.check_call` prints to stdout.
If temporary files are missing, try setting the environment variable

    LOOM_CLEANUP_ON_ERROR=0

When debugging multi-threaded python code, sometimes messages are difficult to
read (e.g. when sifting through 32 threads' error messages).
In this case, try setting the environment variable

    LOOM_THREADS=1

### Testing

The simplest unit tests are accessible by

    make test
    make small-test  # equivalent to make test
    make big-test

These use `loom.datasets` to create synthetic datsets.
Each synthetic datset is accessible from the decorator
`loom.test.util.for_each_dataset`.

When hacking on inference kernels, posterior enumeration tests are much more
sensitive (and expensive). To see available tests, run

    python -m loom.test.posterior_enum

### Profiling, Benchmarking, and Debugging

Each of the high-level C++ executables is wrapped in a benchmarking jig.
To see available jigs, run

    python -m loom.benchmark

These are useful for debugging (set debug=true profile=none),
benchmarking (set debug=false profile=time) and profiling
(e.g. set debug=true profile=callgrind).
To see a list of pre-wrapped profilers, run

    python -m loom.benchmark profilers

The benchmark jigs each take a dataset name.
For debugging, small datasets work well, but for benchmarking, we recommend
using larger datasets or your own datasets.
Each jig depends on previous data, so, e.g.,
to profile inference with your own dataset, you'll need to

    python -m loom.datasets load my-data my-schema.json my-rows.csv
    python -m loom.benchmark ingest my-data
    python -m loom.benchmark tare my-data
    python -m loom.benchmark sparsify my-data
    python -m loom.benchmark init my-data
    python -m loom.benchmark shuffle my-data
    python -m loom.benchmark infer my-data profile=time  # to get a rough idea
    python -m loom.benchmark init-checkpoint my-data
    python -m loom.benchmark infer-checkpoint my-data profile=callgrind
    kcachegrind callgrind.out &  # to view profiling results
