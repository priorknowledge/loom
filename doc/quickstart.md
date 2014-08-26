# Quick Start

1.  Install loom following the [Install Guide](/doc/installing.md).

2.  (optional) Set up a remote ipython notebook server

        ssh -A -L 8888:localhost:8888 <user>@<hostname>
        workon loom
        pip install ipython[all]
        pip install --upgrade pyzmq
        ipython notebook --no-browser --ip=0.0.0.0 --matplotlib=inline
    
    You should now be able to access the ipython notebook server at `http://localhost:8888`

3.  Prepare a dataset for ingestion.
    Loom needs two pieces of data to start: a csv file for data and
    a schema indicating which feature models to use for each column:

    We will start with the [example taxi dataset](/examples/taxi).
    Here are the files:
    [example.csv](/examples/taxi/example.csv),
    [schema.json](/examples/taxi/schema.json).
 
4.  Ingest data.  We will name our example "quickstart" below

        cd $LOOM/examples/taxi
        python -m loom.tasks ingest quickstart schema.json example.csv

5.  Run inference.  This can take a long time on larger datasets.

        python -m loom.tasks infer quickstart

6.  Interactively query loom using the client library.

        python
        import loom.tasks
        with loom.tasks.query("quickstart") as server:
            related = server.relate(["feature1", "feature2", "feature3"])
            print related
