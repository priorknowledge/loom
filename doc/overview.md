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
    python -m loom.test.posterior_enum  # more expensive correctness tests
    python -m loom.benchmark            # speed tests

## Loom data formats

Loom injests data in various gzipped protobuf messages and streams.
See `loom/schema.proto` for message formats.

A typical inference problem consists of files:

    config.pb.gz                    # inference configuration
    rows.pbs.gz                     # stream of data rows
    model_in.pb.gz                  # initial model parameters etc.
    model_out.pb.gz                 # learned model parameters
    groups_out/mixture.000.pbs.gz   # sufficient statistics for kind 0
    groups_out/mixture.001.pbs.gz   # sufficient statistics for kind 1
    ...
    assign_out.pbs.gz               # stream of inferred group assignments
    log.pbs.gz                      # stream of log messages
    checkpoint.pb.gz                # checkpointed inference state

Rows can be streamed in via stdin,
and assignments can be streamed out via stdout.
