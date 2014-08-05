# Using Loom

* [Overview](#overview)
* [Formatting Data](#format)
* [Running Inference](#infer)
* [Querying Results](#query)
* [Assessing Inference Quality](#quality)
* [Tuning Loom](#tuning)

## Overview <a name="overview"/>

Loom is organized as a collection of python modules
that wrap C++ stand-alone engine tools.
Using loom is very file-oriented, with the main tasks being purely-functional
transformations between files.

All the most common transforms are accessible both in python and at the
command line with syntax like

    python -m loom.<modulename> function argument1 key1=value1 key2=value2

To see help with a particular module, simply

    python -m loom.<modulename>

The most common transforms are available by the top level

    python -m loom

### Data Store

Loom stores all intermediate files in a tree rooted at `loom.store.STORE` which
defaults to `$LOOM/data` and can be overridden by setting the `LOOM_STORE`
environment variable.

See [Adapting Loom](/doc/adapting.md#dataflow) for detailed dataflow.

    python -m loom

### Data formatting, inference, and querying

    python -m loom.format               # data formatting commands
    python -m loom.runner               # inference & query commands

### Testing

    make test                           # simple unit tests
    make big-test                       # more expensive unit tests
    python -m loom.test.posterior_enum  # more expensive correctness tests
    python -m loom.benchmark            # speed tests

## Input format <a name="format"/>

Loom currently supports the feature types

| Name | Data Type             | Example Values | Probabilistic Model
|------|-----------------------|----------------|---------------------------
| bb   | booleans              | 0,1,true,false | Beta-Bernoulli
| dd   | categorical up to 256 | Monday, June   | Dirichlet-Discrete
| dpd  | unbounded categorical | CRM, 90210     | Dirichlet Process Discrete
| gp   | counts                | 0, 1, 2, 3, 4  | Gamma-Poisson
| nich | real numbers          | -100.0, 1e-4   | Normal-Inverse-Chi-Squared

## Loom data formats

Loom injests data in various gzipped files in .csv, protobuf (.pb) messages
and protobuf streams (.pbs).
See `src/schema.proto` for protobuf message formats.

Loom stores files under `$LOOM_STORE`, defaulting to `loom/data/`.
The store is organized as

    $LOOM_STORE/
    $LOOM_STORE/my-dataset/
    $LOOM_STORE/my-dataset/data/        # directory for starting dataset
    $LOOM_STORE/my-dataset/infer/       # directory for inference
    $LOOM_STORE/my-dataset/.../         # directories for various operations

Each directory has a standard set of files

    rows.csv.gz | rows_csv/*.csv.gz     # one or more input data files
    schema.json                         # schema to interpret csv data
    version.txt                         # loom version when importing data
    encoding.json.gz                    # csv <-> protobuf encoding definition
    rows.pbs.gz                         # stream of data rows
    tares.pbs.gz                        # list of tare rows
    diffs.pbs.gz                        # compressed rows stream
    config.pb.gz                        # inference configuration
    samples/sample.000000/              # per-sample data for sample 0
        init.pb.gz                      # initial model parameters etc.
        shuffled.pbs.gz                 # shuffled, compressed rows
        model_in.pb.gz                  # initial model parameters etc.
        model.pb.gz                     # learned model parameters
        groups/mixture.000000.pbs.gz    # sufficient statistics for kind 0
        groups/mixture.000001.pbs.gz    # sufficient statistics for kind 1
        ...
        assign.pbs.gz                   # stream of inferred group assignments
        infer_log.pbs                   # stream of log messages
        checkpoint.pb.gz                # checkpointed inference state

You can inspect any of these files with

    python -m loom cat FILENAME

And watch log files with

    python -m loom watch /path/to/infer_log.pbs

In single-pass inference rows can be streamed in via stdin
and assignments can be streamed out via stdout.
