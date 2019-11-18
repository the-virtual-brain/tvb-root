REM build env with pre-built binaries from conda repos
conda create -y -n tvb-run scipy numpy networkx scikit-learn cython h5py pip numexpr psutil ipython ipython-notebook

REM use environment
call activate tvb-run

conda install matplotlib pytables numba scikit-image pytest pytest-cov simplejson cherrypy sqlalchemy psycopg2 docutils sympy

REM make sure at least networkx 2.0 is installed
conda update networkx

REM add locally built or pure Python packages
pip install tvb-gdist formencode cfflib jinja2 nibabel sqlalchemy-migrate allensdk BeautifulSoup4 autograd


REM Now Install TVB packages in the correct order:
cd ..\framework_tvb
python setup.py develop

cd ..\scientific_library
python setup.py develop

cd ..\tvb_data
python setup.py develop

cd ..\tvb_bin
python setup.py develop

REM [anaconda-env]/Lib/site-packages/matplotlib/mpl-data/matplotlibrc to Agg