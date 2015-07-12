#!/bin/bash

envname="tvb-run"

conda create -y -n $envname scipy==0.14.0 numpy==1.8.1 networkx scikit-learn cython h5py==2.3.0 pip pil numexpr psutil coverage beautiful-soup lxml ipython ipython-notebook

source activate $envname

conda install matplotlib==1.2.1 pytables==3.0

conda install numpy==1.8.1 psycopg2

pip install sqlalchemy==0.7.8 sqlalchemy-migrate==0.7.2 minixsv formencode genshi simplejson mod_pywebsocket cherrypy cfflib nibabel gdist numexpr==2.4

cd ../../..
python setup_extra.py develop

cd scientific_library
python setup.py develop

cd ../framework_tvb
python setup.py develop

