REM build env with pre-built binaries from conda repos
conda create -y -n tvb-run scipy==0.14.0 numpy==1.8.1 networkx scikit-learn cython h5py==2.3.0 pip pil numexpr psutil coverage beautiful-soup lxml ipython ipython-notebook

REM use environment
call activate tvb-run

conda install matplotlib==1.2.1 pytables==3.0 

conda install numpy==1.8.1

REM add locally built or pure Python packages
pip install gdist sqlalchemy==0.7.8 sqlalchemy-migrate==0.7.2 minixsv formencode genshi simplejson mod_pywebsocket cherrypy cfflib nibabel psycopg2 apscheduler 

cd ..
python setup_extra.py develop

cd scientific_library
python setup.py develop

cd ..\framework_tvb
python setup.py develop


conda uninstall pyside

# Upgrade to have numba on and the same packages as on Unix
conda install numba numpy==1.11.0 numexpr==2.5.2 scikit-learn==0.17.1 scipy==0.17.0

REM [anaconda-env]/Lib/site-packages/matplotlib/mpl-data/matplotlibrc to Agg