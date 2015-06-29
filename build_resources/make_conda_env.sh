#!/bin/bash

if hash conda 2>/dev/null; then
	echo "using $(which conda)"
else
	echo "Please install the Miniconda Python distributions:"
	echo "\thttp://conda.pydata.org/miniconda.html"
	echo "or verify that the 'conda' executable is in the PATH environment"
	echo "variable before executing this script."
	exit 1;
fi

envname="$1"

if [ -z $envname ]; then
	echo "usage: make_conda_env.sh environ_name"
	exit 1;
fi

# build env with pre-built binaries from conda repos
conda create -y -n $envname matplotlib==1.2.1 scipy networkx scikit-learn cython h5py pip pil numexpr psutil coverage

# enter environ
source activate $envname

# add locally built or pure Python packages
pip install gdist sqlalchemy==0.7.8 sqlalchemy-migrate minixsv formencode genshi simplejson mod_pywebsocket cherrypy cfflib nibabel

# advise
echo "to use, execute 'source activate $envname'"
