# Installing Loom + Distributions

Loom targets Ubuntu 12.04 and 14.04 systems and requires the
[distributions](https://github.com/forcedotcom/distributions) library.
This guide describes how to install both loom and distributions.

## Installing with virtualenvwrapper (recommended)

1. Make a new virtualenv named 'loom'.
    You can skip this step if you already have a virtualenv.

        sudo apt-get install virtualenvwrapper
        source ~/.bashrc
        mkvirtualenv --system-site-packages loom

2. Set environment variables.

        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib' >> $VIRTUAL_ENV/bin/postactivate
        echo 'export DISTRIBUTIONS_USE_PROTOBUF=1' >> $VIRTUAL_ENV/bin/postactivate
        workon loom

3. Clone the repos.

        git clone https://github.com/forcedotcom/distributions
        git clone https://github.com/priorknowledge/loom

4. Install required packages.

        sudo easy_install pip
        pip install -r distributions/requirements.txt
        source loom/requirements.sh

5. Build distributions.

        cd distributions
        make && make install
        cd ..

6. Build loom.

        cd loom
        make && make install
        make test               # optional, takes ~30 CPU minutes
        cd ..

Make sure to `workon loom` whenever you start a new bash session for looming.

## Installing globally for all users

If you prefer to avoid using virtualenvwrapper:

1.  Set environment variables.

        echo 'export DISTRIBUTIONS_USE_PROTOBUF=1' >> ~/.bashrc
        source ~/.bashrc

3. Build distributions and loom as above, but installing as root

    sudo make install       # instead of `make install`

## Custom Installation

Loom assumes distributions is installed in a standard location.
You may need to set `CMAKE_PREFIX_PATH` for loom to find distributions.

### virtualenv

Within a virtualenv, both distributions and loom assume a prefix of
`$VIRTUAL_ENV`. `make install` installs headers to
`$VIRTUAL_ENV/include`, libs to `$VIRTUAL_ENV/lib`, and so on.

For distributions and loom to find these installed libraries at
runtime, `LD_LIBRARY_PATH` must include `$VIRTUAL_ENV/lib`. With
virtualenvwrapper, it's convenient to do this in a postactivate hook:

    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib' >> $VIRTUAL_ENV/bin/postactivate
