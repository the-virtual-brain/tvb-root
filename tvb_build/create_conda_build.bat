REM build env with pre-built binaries from conda repos
conda create -y -n tvb-run scipy==0.19.0 numpy==1.12.1 networkx scikit-learn cython h5py pip pil numexpr psutil coverage beautiful-soup lxml ipython ipython-notebook

REM use environment
call activate tvb-run

conda install matplotlib==1.5.1 pytables==3.0 numba

REM add locally built or pure Python packages
pip install gdist sqlalchemy==0.7.8 sqlalchemy-migrate==0.7.2 minixsv formencode genshi simplejson mod_pywebsocket cherrypy cfflib nibabel psycopg2 apscheduler 


#After these run "sh install_from_svn.sh" or "python setup.py develop/install" from each of TVB packages

cd ..\scientific_library
python setup.py develop

cd ..\framework_tvb
python setup.py develop

conda uninstall pyside

REM [anaconda-env]/Lib/site-packages/matplotlib/mpl-data/matplotlibrc to Agg