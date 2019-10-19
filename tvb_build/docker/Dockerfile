FROM continuumio/miniconda3

RUN apt-get -y update && apt-get -y install build-essential gcc
RUN apt-get -y install texlive-base texlive-formats-extra

# TODO install Octave
# TODO fix install of Postgres

#RUN wget -q https://www.postgresql.org/media/keys/ACCC4CF8.asc -O - | apt-key add -
#RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt/ stretch-pgdg main" >> /etc/apt/sources.list.d/pgdg.list'
#RUN apt-get update
#RUN apt-get -y install postgresql
#
#USER postgres
#RUN service postgresql start && createdb -O postgres tvb-test && psql --command "ALTER USER postgres WITH PASSWORD 'postgres';"

USER root

#RUN conda update -n base -c defaults conda
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

# Jupyther notebook configurations: set password
# tvb42
RUN /bin/bash -c "source activate tvb-run"; \
    /opt/conda/envs/tvb-run/bin/jupyter notebook --generate-config; \
    echo "c.NotebookApp.password='sha1:12bff019c253:9daecd92c2e9bdb10b3b8a06767a74a0fe078d7c'">>/root/.jupyter/jupyter_notebook_config.py

RUN mkdir /root/.tvb-temp; mkdir /root/.tvb-temp/logs; mkdir /root/tvb-root

RUN wget https://zenodo.org/record/3497545/files/tvb_data_2_0_1.zip?download=1 -O tvb_data.zip; \
    unzip tvb_data.zip; \
    cd tvb_data; \
    /opt/conda/envs/tvb-run/bin/python setup.py develop

COPY .tvb.configuration /root/.tvb.configuration
RUN mkdir /root/TVB_DATA
WORKDIR /root/tvb-root
# MOUNT -v [local- tvb-root - clone]:/root/tvb-root

# For building static help for web
#CMD ["bash","-c","source activate tvb-docs && cd /root/tvb-root/tvb_build && sh install_full_tvb.sh && python build_step1.py"]

# For running all unit-tests
# inspect output in local tvb-root/tvb_bin/TEST_OUTPUT folder
#CMD ["bash","-c","source activate tvb-run && cd tvb_build && sh install_full_tvb.sh && cd ../tvb_bin && bash run_tests.sh"]

# For running Jupyter notebooks
# bind port 8888:8888
#CMD ["bash","-c","source activate tvb-run && cd tvb_build && sh install_full_tvb.sh && cd ../tvb_documentation && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]

# For running TVB Web GUI
# bind port 8080
# MOUNT -v [local- ~/TVB ]:/root/TVB_DATA
CMD ["bash","-c","source activate tvb-run && cd tvb_build && sh install_full_tvb.sh && cd ../tvb_bin && python -m tvb.interfaces.web.run WEB_PROFILE"]