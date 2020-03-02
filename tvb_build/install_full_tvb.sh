#!/usr/bin/env bash

# Use this script to install TVB from the main code sources repo, to the current python installation

# set up external dependencies that we maintain as distutils packages in externals/
cd ..

# Temporarily comment tvb-gdist install, as it has bugs.
#cd externals/tvb_gdist
#python setup.py install
#rm -rf build
#rm -rf dist
#rm -rf tvb_gdist.egg-info
#rm -f gdist.cpp
#cd ../..

cd framework_tvb
python setup.py develop --no-deps
cd ..

cd scientific_library
python setup.py develop
cd ..

if [[ -d "tvb_data" ]]; then
    cd tvb_data
    python setup.py develop
    cd ..
fi

cd tvb_bin
python setup.py develop

cd ../tvb_build
python setup.py develop --no-deps
