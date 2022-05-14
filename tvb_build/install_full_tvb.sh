#!/usr/bin/env bash

# Use this script to install TVB from the main code sources repo, to the current python installation

# set up external dependencies that we maintain as distutils packages in externals/
cd ..

cd tvb_framework
python setup.py develop --no-deps --user
cd ..

cd tvb_library
python setup.py develop --user
cd ..

cd tvb_storage
python setup.py develop --user
cd ..

cd tvb_contrib
python setup.py develop --no-deps --user
cd ..

cd tvb_bin
python setup.py develop --user

cd ../tvb_build
python setup.py develop --no-deps --user
