REM build env with pre-built binaries from conda repos
conda create -y -n tvb-build  matplotlib==1.1.1 scipy==0.13.2 networkx scikit-learn cython h5py pip pil numexpr psutil coverage

REM use environment
call activate tvb-build

REM add locally built or pure Python packages
pip install gdist sqlalchemy==0.7.8 sqlalchemy-migrate minixsv formencode genshi simplejson mod_pywebsocket cherrypy cfflib nibabel

REM exit environment
deactivate