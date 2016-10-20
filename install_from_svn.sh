#!/usr/bin/env bash

# Use this script to install TVB's distutils packages, from the svn source, to the current python installation

# set up external dependencies that we maintain as distutils packages in externals/

cd externals/mplh5canvas
python setup.py install
rm -rf mplh5canvas.egg-info
rm -rf build
rm -rf dist
cd ../..

#cd externals/geodesic_distance
#python setup.py install
#rm -rf build
#rm -rf dist
#rm -rf gdist.egg-info
#rm -f gdist.cpp
#cd ../..

cd scientific_library
python setup.py develop
cd ..

cd framework_tvb
python setup.py develop
cd ..

cd tvb_data
python setup.py develop
cd ..

cd tvb_bin
python setup.py develop
cd ..
