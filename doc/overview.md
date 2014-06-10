# Loom Interface

Loom is built of a low-level C++ layer,
and a high level python module to run the C++ layer.
The basic commands can be found via

    python -m loom

## Data formatting, inference, and querying

    python -m loom.format               # data formatting commands
    python -m loom.runner               # inference & query commands

## Testing

    make test                           # simple unit tests
    python -m loom.test.posterior_enum  # more expensive correctness tests
    python -m loom.benchmark            # speed tests
