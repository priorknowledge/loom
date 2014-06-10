## Installing Loom

Loom targets Ubuntu 12.04 systems and requires the
[distributions](https://github.com/forcedotcom/distributions) library.

Step 1: build distributions.

    export DISTRIBUTIONS_PATH=/desired/path/to/distributions
    git clone https://github.com/forcedotcom/distributions $DISTRIBUTIONS_PATH
    cd $DISTRIBUTIONS_PATH
    make

Step 2: build loom

    export LOOM_PATH=/desired/path/to/loom
    git clone https://github.com/forcedotcom/loom $LOOM_PATH
    cd $LOOM_PATH
    ./requirements.sh
    make
