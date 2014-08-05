# Quick Start

1.  Install loom following the [Install Guide](/doc/installing.md).

2.  Prepare a dataset for ingestion.
    Loom needs two pieces of data to start: a csv file (or files) for data
    and a schema indicating which feature models to use for each column:

    We will start with the [example taxi dataset](/examples/taxi).
    Here are the files:
    [example.csv](/examples/taxi/example.csv),
    [schema.json](/examples/taxi/schema.json).
 
3.  Ingest data.  We will name our example "quickstart" below

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
