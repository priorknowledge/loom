# Quick Start

1.  Install loom following the [Install Guide](/doc/installing.md) and activate the 
    virtualenv you created during the install process, for example using:

        workon loom

2.  (optional) Set up a remote ipython notebook server

        ssh -A -L 8888:localhost:8888 <user>@<hostname>
        workon loom
        pip install ipython[all]
        pip install --upgrade pyzmq
        ipython notebook --no-browser --ip=0.0.0.0 --matplotlib=inline
    
    You should now be able to access the ipython notebook server at `http://localhost:8888`

3.  There are three basic steps in the loom workflow: preparing and ingesting data, 
    running inference, and querying the results.

    To prepare a dataset for ingestion, loom needs two files: the data in a csv file,
    and a schema indicating which feature models to use for each column in the csv.

    The [taxi example](/examples/taxi) contains both of these, ready to go:
    [example.csv](/examples/taxi/example.csv),
    [schema.json](/examples/taxi/schema.json).
 
4.  Ingest the data. Loom will read in the csv and the schema, and translate them into [its
    own on-disk representation](/doc/using.md#loom-file-formats). 

    You must supply a name, which is how loom will refer to this analysis in subsequent 
    steps; we will use the name "quickstart".

        cd $LOOM/examples/taxi
        python -m loom.tasks ingest quickstart schema.json example.csv

5.  Run inference.  This step reads the ingested data from the previous step, and produces indexes
    that can be queried.

        python -m loom.tasks infer quickstart

6.  Interactively query loom using the client library. See [here](FIXME) for more information about the supported
    query operations.

        python
        import loom.tasks
        with loom.tasks.query("quickstart") as server:
            related = server.relate(["feature1", "feature2", "feature3"])
            print related
