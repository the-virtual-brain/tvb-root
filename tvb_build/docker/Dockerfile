FROM continuumio/miniconda3

RUN apt-get -y update && apt-get -y install build-essential gcc
RUN apt-get -y install texlive-base texlive-formats-extra

RUN wget -q https://www.postgresql.org/media/keys/ACCC4CF8.asc -O - | apt-key add -
RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ stretch-pgdg main" >> /etc/apt/sources.list.d/pgdg.list'
RUN apt-get update
RUN apt-get -y install postgresql

USER postgres
RUN service postgresql start && createdb -O postgres tvb-test && psql --command "ALTER USER postgres WITH PASSWORD 'postgres';"

USER root

RUN conda update -n base -c defaults conda
RUN conda create -y --name tvb-docs python=3.7 nomkl numba scipy numpy networkx scikit-learn cython pip numexpr psutil
RUN conda install -y --name tvb-docs pytest pytest-cov pytest-benchmark matplotlib
RUN conda install -y --name tvb-docs psycopg2 pytables scikit-image==0.14.2 simplejson cherrypy docutils
RUN conda install -y --name tvb-docs -c conda-forge jupyterlab
RUN /opt/conda/envs/tvb-docs/bin/pip install --upgrade pip
RUN /opt/conda/envs/tvb-docs/bin/pip install h5py>=2.10 formencode cfflib genshi nibabel sqlalchemy==1.1.14 sqlalchemy-migrate==0.11.0 allensdk BeautifulSoup4 subprocess32
RUN /opt/conda/envs/tvb-docs/bin/pip install tvb-gdist typing sphinx==1.2.3 docutils==0.12

RUN conda create -y --name tvb-run python=3.7 nomkl numba scipy numpy networkx scikit-learn cython pip numexpr psutil
RUN conda install -y --name tvb-run pytest pytest-cov pytest-benchmark matplotlib
RUN conda install -y --name tvb-run psycopg2 pytables scikit-image==0.14.2 simplejson cherrypy docutils
RUN conda install -y --name tvb-run -c conda-forge jupyterlab
RUN /opt/conda/envs/tvb-run/bin/pip install --upgrade pip
RUN /opt/conda/envs/tvb-run/bin/pip install h5py>=2.10 formencode cfflib genshi nibabel sqlalchemy==1.1.14 sqlalchemy-migrate==0.11.0 allensdk BeautifulSoup4 subprocess32
RUN /opt/conda/envs/tvb-run/bin/pip install tvb-gdist typing

COPY kernel.json /opt/conda/envs/tvb-run/share/jupyter/kernels/python2/

# Check why this is not created
RUN mkdir /root/.tvb-temp
RUN mkdir /root/.tvb-temp/logs