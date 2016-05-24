#!/bin/bash

envname="tvb-run2"

echo '======= creating virtual env ====== '
echo
# create the virtual env exit this script on failure

conda create -y -n $envname numpy==1.8.1 python=2.7 || exit 1

source activate $envname || exit 2


echo '======= install dependencies ====== '
echo

# Install all dependencies in one command.
# Bad package dependencies will be reported and the install fails.
# This is better than using more conda create-s which will downgrade packages and break dependencies.

conda install -y scipy==0.14.0 numpy==1.8.1 networkx scikit-learn cython h5py==2.3.0 \
                 pip pil numexpr psutil coverage beautiful-soup lxml ipython ipython-notebook \
                 matplotlib==1.3.1 pytables==3.1.0 sqlalchemy==0.7.8 psycopg2

# Note that to resolve conflicts matplotlib==1.2.1 became matplotlib==1.3.1 and pytables==3.0 became pytables==3.1.0
# TODO: validate these updated dependencies

# this may be required on linux due to a conda bug with numpy 1.8
conda install -y libgfortran==1.0


echo '======= install pip dependencies ====== '
echo

# why do we depend on pypi gdist if we have it in externals?

pip install sqlalchemy==0.7.8 sqlalchemy-migrate==0.7.2 minixsv formencode genshi \
            simplejson mod_pywebsocket cherrypy cfflib nibabel gdist numexpr==2.4


echo '======= install externals ====== '
echo

cd ..
# assumming now in tvb top dir

cd externals/mplh5canvas || exit 3
python setup.py install
rm -rf mplh5canvas.egg-info
rm -rf build
rm -rf dist
cd ../..

cd externals/geodesic_distance
python setup.py install
rm -rf build
rm -rf dist
rm -rf gdist.egg-info
rm -f gdist.cpp
cd ../..


echo '======= install scientific library ====== '
echo

cd scientific_library
python setup.py develop
cd ..

echo '======= install framework ====== '
echo

cd framework_tvb
python setup.py develop
cd ..

echo '======= install extras ====== '
echo

cd tvb_data
python setup.py develop
cd ..

cd tvb_bin
python setup.py develop
cd ..
