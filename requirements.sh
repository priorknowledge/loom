#!/bin/sh

sudo apt-get install -y \
    make \
    cmake \
    g++ \
    protobuf-compiler \
    libprotobuf-dev \
    libgoogle-perftools-dev \
    libboost-python-dev \
    libeigen3-dev \
    python-setuptools \
    cython \
    python-numpy \
    python-scipy \
    graphviz \
    unzip \
    #

# install distributions separately
grep -v distributions requirements.txt | xargs pip install
