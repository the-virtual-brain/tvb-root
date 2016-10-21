#!/bin/bash

envname="tvb-run"

conda create -y -n $envname nomkl scipy==0.17.0 numpy==1.11.0 networkx scikit-learn cython h5py==2.3.0 pip pil numexpr psutil coverage beautiful-soup lxml ipython ipython-notebook

source activate $envname

conda install matplotlib==1.5.1 pytables==3.0 psycopg2 numba

pip install sqlalchemy==0.7.8 sqlalchemy-migrate==0.7.2 minixsv formencode genshi simplejson mod_pywebsocket cherrypy cfflib nibabel gdist

# The next ones are for Mac build env:
# pip install py2app docutils apscheduler pyobjc
# pip install --upgrade setuptools
# pip install --upgrade distribute
# Edit [anaconda-env]/Lib/python2.7/site-packages/macholib/MatchOGraph.py loader= into loader_path=

# After these run "sh install_from_svn.sh" or "python setup.py develop/install" from each of TVB packages

cd ../../..
python setup_extra.py develop

cd scientific_library
python setup.py develop

cd ../framework_tvb
python setup.py develop

conda uninstall pyside shiboken cairo

# [anaconda-env]/Lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc to Agg
