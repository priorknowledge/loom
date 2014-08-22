# Installing Loom

Loom targets Ubuntu 12.04 and 14.04 systems and requires the
[distributions](https://github.com/forcedotcom/distributions) library.

## Install Distributions + Loom with virtualenvwrapper (recommended)

1. Make a virtualenv. We'll name our virtualenv 'loom'.

        easy_install pip
        pip install virtualenvwrapper
        source ~/.bashrc
        mkvirtualenv --system-site-packages loom
        workon loom

2. Set environment variables.

        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VIRTUAL_ENV/lib' >> $VIRTUAL_ENV/bin/postactivate
        echo 'export DISTRIBUTIONS_USE_PROTOBUF=1' >> ~/.bashrc
        workon loom

3. Build distributions.

        git clone https://github.com/forcedotcom/distributions
        cd distributions
        make
        make install
        cd ..

4. Build loom.

        git clone https://github.com/forcedotcom/loom
        cd loom
        ./requirements.sh       # installs apt packages
        make
        make install
        cd ..

5. Test that loom works (optional).

        cd loom
        make test       # Takes ~30 CPU minutes
        cd ..

Make sure to `workon loom` whenever you start a new bash session for looming.

## Install distributions + loom for all users

To avoid using virtualenvwrapper:

1.  Set environment variables.

        echo 'export DISTRIBUTIONS_USE_PROTOBUF=1' >> ~/.bashrc
        source ~/.bashrc

3. Build distributions and loom as above, but installing as root

    sudo make install

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
