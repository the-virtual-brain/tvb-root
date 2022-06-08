#!/usr/bin/env bash

# Use this script to install TVB from the main code sources repo, to the current python installation

# set up external dependencies that we maintain as distutils packages in externals/

cd ../tvb_framework
pip install -e . --no-deps --user

cd ../tvb_library
pip install -e . --user

cd ../tvb_storage
pip install -e . --user

cd ../tvb_contrib
pip install -e . --no-deps --user

cd ../tvb_bin
pip install -e . --user

cd ../tvb_build
pip install -e . --no-deps --user
