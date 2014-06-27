# Installing Loom

Loom targets Ubuntu 12.04 systems and requires the
[distributions](https://github.com/forcedotcom/distributions) library.

Loom assumes distributions is installed in a standard location. You
may need to set `CMAKE_PREFIX_PATH` for loom to find distributions.

## virtualenv

Within a virtualenv, both distributions and loom assume a prefix of
`$VIRTUAL_ENV`. `make install` installs headers to
`$VIRTUAL_ENV/include`, libs to `VIRTUAL_ENV/lib`, and so on.

For distributions and loom to find these installed libraries at
runtime, `LD_LIBRARY_PATH` must include `$VIRTUAL_ENV/lib`. With
virtualenvwrapper, it's convenient to do this in a postactivate hook:

    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib' >> $VIRTUAL_ENV/bin/postactivate

## build

These steps assume you are running within a virtualenv.

Step 1: build distributions with protobuf support.

    git clone https://github.com/forcedotcom/distributions
    cd distributions
    DISTRIBUTIONS_USE_PROTOBUF=1 make install

Step 2: build loom

    git clone https://github.com/forcedotcom/loom
    cd loom
    ./requirements.sh
    make

## Ubuntu 14.04

In Ubuntu 14.04 the distributions.io library needs to be manally rebuilt:

Step 1.5:

    cd distributions
    make protobuf
    DISTRIBUTIONS_USE_PROTOBUF=1 make install
