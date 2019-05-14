FROM continuumio/miniconda

RUN apt-get -y update && apt-get -y install build-essential gcc
RUN apt-get -y install texlive-base texlive-formats-extra
RUN apt-get -y install postgresql

USER postgres
RUN service postgresql start && createdb -O postgres tvb-test && psql --command "ALTER USER postgres WITH PASSWORD 'postgres';"

USER root

RUN conda create -y --name tvb-docs python=2.7
RUN conda install -y --name tvb-docs numpy cython
RUN /opt/conda/envs/tvb-docs/bin/pip install sphinx==1.3.1 docutils==0.12

RUN conda create -y --name tvb-run python=2.7 nomkl scipy numpy networkx scikit-learn cython h5py pip numexpr psutil
RUN conda install -y --name tvb-run cython pytest pytest-cov ipython ipython-notebook
RUN conda install -y --name tvb-run psycopg2 sqlalchemy>=1.1.14
RUN conda install -y --name tvb-run matplotlib==2.1.0 pytables numba scikit-image simplejson cherrypy  docutils
RUN conda install -y --name tvb-run -c conda-forge subprocess32
RUN /opt/conda/envs/tvb-run/bin/pip install --upgrade pip
RUN /opt/conda/envs/tvb-run/bin/pip install tvb-gdist formencode cfflib genshi nibabel sqlalchemy-migrate allensdk BeautifulSoup4
RUN /opt/conda/envs/tvb-run/bin/pip install tvb-framework
RUN /opt/conda/envs/tvb-run/bin/pip uninstall -y tvb-framework tvb-library tvb-data
RUN conda uninstall -y --name tvb-run pyside shiboken cairo pycairo pywavelets

# Check why this is not created
RUN mkdir /root/.tvb-temp
RUN mkdir /root/.tvb-temp/logs