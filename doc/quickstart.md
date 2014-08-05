# Quick Start

1.  install loom following the [Install Guide](/doc/installing.md).

2.  Format a dataset for ingestion. We will start with the
    [example taxi dataset](/examples/taxi).
    Loom needs two pieces of data to start: a csv file (or files) for data,
    and a schema indicating which feature models to use for each column.
    Loom currently supports the feature types

    | Name | Data Type             | Example Values | Probabilistic Model
    |------|-----------------------|----------------|---------------------------
    | bb   | booleans              | 0,1,true,false | Beta-Bernoulli
    | dd   | categorical up to 256 | Monday, June   | Dirichlet-Discrete
    | dpd  | unbounded categorical | CRM, 90210     | Dirichlet Process Discrete
    | gp   | counts                | 0, 1, 2, 3, 4  | Gamma-Poisson
    | nich | real numbers          | -100.0, 1e-4   | Normal-Inverse-Chi-Squared

    Here is the taxi example:
    [schema.json](/examples/taxi/schema.json),
    [example.csv](/examples/taxi/example.csv)
 
3.  Ingest data.

        cd $LOOM/examples/taxi
        python -m loom.tasks ingest quickstart schema.json example.csv

4.  Run inference.  This can take a long time on larger datasets.

        python -m loom.tasks infer quickstart

5.  Interactively query loom using the client library.

        python
        import loom.tasks
        with loom.tasks.query("quickstart") as server:
            related = server.relate(["feature1", "feature2", "feature3"])
            print related
