# Using Loom

# Loom Interface

Loom is built of a low-level C++ layer,
and a high level python module to run the C++ layer.
The basic commands can be found via

    python -m loom

### Data formatting, inference, and querying

    python -m loom.format               # data formatting commands
    python -m loom.runner               # inference & query commands

### Testing

    make test                           # simple unit tests
    make big-test                       # more expensive unit tests
    python -m loom.test.posterior_enum  # more expensive correctness tests
    python -m loom.benchmark            # speed tests

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
