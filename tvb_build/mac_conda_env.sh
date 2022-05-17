#!/bin/bash

#Install dependencies
conda update -n base -c defaults conda
conda env remove --name mac-distribution
conda create -y --name mac-distribution python=3 nomkl numba scipy numpy networkx scikit-learn cython pip numexpr psutil
conda install -y --name mac-distribution pytest pytest-cov pytest-benchmark pytest-mock pytest-xdist matplotlib-base
conda install -y --name mac-distribution psycopg2 pytables scikit-image==0.14.2 simplejson cherrypy docutils werkzeug
conda install -y --name mac-distribution -c conda-forge gevent pillow

/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip install --upgrade pip
/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip install h5py formencode cfflib flask==1.1.4 jinja2==2.11.3 nibabel sqlalchemy alembic allensdk # h5py and jinja2 versions are constarined by allensdk
/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip install tvb-gdist BeautifulSoup4 subprocess32 flask-restx python-keycloak mako pyAesCrypt pyunicore==0.6.0
/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip install pyobjc
/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip install biplist six
/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip uninstall python-magic
/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip install python-magic-bin==0.4.14
/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip install dmgbuild
/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip install jupyterlab
/WORK/anaconda3/anaconda3/envs/mac-distribution/bin/pip install syncrypto
