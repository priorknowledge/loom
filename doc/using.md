# Using Loom

* [Overview](#overview)
* [Formatting Data](#format)
* [Running Inference](#infer)
* [Querying Results](#query)
* [Assessing Inference Quality](#quality)
* [Tuning Loom](#tuning)

## Overview <a name="overview"/>

Loom is organized as a collection of python modules that wrap C++ stand-alone engine tools.
Using loom is very file-oriented, with the main tasks being purely-functional
transformations between files.
The highest-level tasks are in the `loom.tasks` module.

<!--
FIXME what are the main tasks? are all of the common operations contained in loom.tasks? If not, should they be?
-->

The most common operations are accessible both in python and at the
command line with syntax like

    python -m loom.<modulename> function argument1 key1=value1 key2=value2

To see help with a particular module, simply

    python -m loom.<modulename>    # lists available commands

The most common transforms are available by the top level

    python -m loom    # lists available commands

Loom stores all intermediate files in a tree rooted at `loom.store.STORE` which
defaults to `$LOOM/data` and can be overridden by setting the `LOOM_STORE`
environment variable.

See [Adapting Loom](/doc/adapting.md#dataflow) for detailed dataflow.

## Input format <a name="format"/>

Loom inputs a pair of files `schema.csv` and `rows.csv` via `loom.tasks.transform`,
which creates a stricter schema and table with basic types suitable for `loom.tasks.ingest`.
The `schema.csv` accepted by `loom.tasks.transform` should document
all relevant columns in `rows.csv`.

An example `schema.csv`:

| Feature Name | Type                   |
|--------------|------------------------|
| full name    | id                     |
| start date   | optional\_date         |
| age          | real                   |
| zipcode      | unbounded\_categorical |
| description  | text                   |
| misc unused  |                        |

Loom currently supports the following fluent feature types in `loom.tasks.transform`:

| Fluent Type            | Example Values                                 | Transforms To        |
|------------------------|------------------------------------------------|----------------------|
| boolean                | '0', '1', 'true', 'false'                      | bb                   |
| categorical            | 'Monday', 'June'                               | dd                   |
| unbounded\_categorical | 'CRM', '90210'                                 | dpd                  |
| count                  | '0', '1', '2', '3', '4'                        | gp                   |
| real                   | '-100.0', '1e-4'                               | nich                 |
| sparse\_real           | '0', '0', '0', '0', '123456.78', '0', '0', '0' | bb + nich            |
| date                   | '2014-03-31', '10pm, August 1, 1979'           | many nich + many dpd |
| text                   | 'This is a text feature.', 'Hello World!'      | many bb              |
| tags                   | '', 'big_data machine_learning platform'       | many bb              |
| optional\_(TYPE)       | '', ...examples of TYPE...                     | bb + TYPE            |

Text fields typically transform to 100-1000 boolean features
corresponding to word presence of words that occur in at least 1% of documents.
Date fields transform to absolute, cyclic and relative features,
so it is expensive to have lots of date fields.
An input schema with N date fields will transform to features:

    N date = N nich         # absolute date
           + N(N-1)/2 nich  # relative dates for each pair
           + 4 N dpd        # month + day-of-month + day-of-week + hour-of-day

Loom currently supports the following basic feature models in `loom.tasks.ingest`:

| Name | Basic Type            | Example Values | Probabilistic Model        | Relative Cost
|------|-----------------------|----------------|----------------------------|--------------
| bb   | boolean               | 0,1,true,false | Beta-Bernoulli             | 1
| dd   | categorical up to 256 | Monday, June   | Dirichlet-Discrete         | 1.5
| dpd  | unbounded categorical | CRM, 90210     | Dirichlet-Process-Discrete | 3
| gp   | count                 | 0, 1, 2, 3, 4  | Gamma-Poisson              | 50
| nich | real number           | -100.0, 1e-4   | Normal-Inverse-Chi-Squared | 20

## Loom file formats

Loom ingests data in various gzipped files in .csv, protobuf (.pb) messages
and protobuf streams (.pbs).
See `src/schema.proto` for protobuf message formats.

The `loom.store` module gives programmatic access to the loom store.
Loom stores files under `$LOOM_STORE`, defaulting to `loom/data/`.
The store is organized as

    $LOOM_STORE/
    $LOOM_STORE/my-dataset/
    $LOOM_STORE/my-dataset/ingest/      # directory for starting dataset
    $LOOM_STORE/my-dataset/samples/     # directory for inference
    $LOOM_STORE/my-dataset/.../         # directories for various operations

Each directory has a standard set of files

    ingest/                             # ingested data
      version.txt                       # loom version when importing data
      transforms.pickle.gz              # feature transforms
      schema.json | schema.json.gz      # schema to interpret csv data
      rows.csv.gz | rows_csv/*.csv.gz   # one or more input data files
      rowids.csv.gz                     # internal <-> external id mapping
      encoding.json.gz                  # csv <-> protobuf encoding definition
      rows.pbs.gz                       # stream of data rows
      schema_row.pb.gz                  # example row to serve as schema
      tares.pbs.gz                      # list of tare rows
      diffs.pbs.gz                      # compressed rows stream
    samples/                            # inferred samples
      sample.0/                         # per-sample data for sample 0
        config.pb.gz                    # inference configuration
        init.pb.gz                      # initial model parameters etc.
        shuffled.pbs.gz                 # shuffled, compressed rows
        model.pb.gz                     # learned model parameters
        groups/                         # sufficient statistics
          mixture.0.pbs.gz              # sufficient statistics for kind 0
          mixture.1.pbs.gz              # sufficient statistics for kind 1
          ...
        assign.pbs.gz                   # stream of inferred group assignments
        infer_log.pbs                   # stream of log messages
        checkpoint.pb.gz                # checkpointed inference state
      sample.1/                         # per-sample data for sample 1
        ...
    query/                              # query server data
      config.pb.gz                      # query configuration
      query_log.pbs                     # stream of log messages

You can inspect any of these files with

    python -m loom cat FILENAME         # parse + prettyprint

And watch log files with

    python -m loom watch /path/to/infer_log.pbs

In single-pass inference rows can be streamed in via stdin
and assignments can be streamed out via stdout.

## Querying Results <a name="query"/>

Loom supports flexible, interactive querying of inference results. These queries are divided between 
low-level operations, implemented in [loom.runner.query](/loom/runner.py), and higher-level operations, in [loom.preql](/loom/preql.py).

The **low-level primitives**, `sample` and `score`, are written in C++. They can be accessed as a 
transformation of protobuf messages via the `loom.runner.query` function. See `src/schema.proto` 
for the query message format.

The `loom.query` module provides a convenient way to create a persistent query server with both protobuf and python interfaces. 

<!--
* `sample` FIXME explain

* `score` FIXME explain
-->

**Higher level operations** are supported via the `loom.preql` module. Assuming that you have completed an 
inference run named `iris`, you can create a query server using the following convenience method in `loom.tasks`:

    import loom.tasks
    server = loom.tasks.query('iris')

* `related` returns scores between 0 and 1 representing the strength of the predictive relationship between two columns or groups of columns. A score of 1 indicates that loom has found very strong evidence for a relationship between the columns, and a score of 0 means that no evidence was found; intermediate values represent more graded levels of confidence.

    print server.relate(['class'])

* `predict` is a very flexible operation that returns a simulated values for one or more unknown columns, 
given fixed values for a different subset of columns. This flexibility is possible because loom learns a 
joint model of the data. Standard classification and regression tasks are therefore one special case, in which 
the value of one "dependent" or "target" column is predicted given values for all of the other columns (known as "independent variables", "predictors", "regressors", or "features").

  Some examples of `predict` usage are as follows:

  * Fix all but one column and predict that column.
  
    <!-- FIXME explanation and example usage -->
    
  * Fix some columns, predict some other columns.
  
    <!-- FIXME explanation and example usage -->
    
  * Fix no columns, predict all columns.
  
    <!-- FIXME explanation and example usage -->

<!--

* `group` FIXME not implemented

* `similar` FIXME not implemented

-->
