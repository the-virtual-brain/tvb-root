#!/usr/bin/env bash

# Use this script to install TVB from the main code sources repo, to the current python installation

# set up external dependencies that we maintain as distutils packages in externals/
cd ..

cd framework_tvb
python setup.py develop --no-deps
cd ..

cd scientific_library
python setup.py develop
cd ..

cd tvb_contrib
python setup.py develop --no-deps
cd ..

cd tvb_bin
python setup.py develop

cd ../tvb_build
python setup.py develop --no-deps
