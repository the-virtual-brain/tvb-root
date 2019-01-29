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

RUN conda create -y --name tvb-run python=2.7
RUN conda install -y --name tvb-run cython psycopg2=2.7.3.2 pytest-cov
RUN conda install -y --name tvb-run -c conda-forge subprocess32
RUN /opt/conda/envs/tvb-run/bin/pip install --upgrade pip
RUN /opt/conda/envs/tvb-run/bin/pip install numpy==1.13.3
RUN /opt/conda/envs/tvb-run/bin/pip install tvb-framework
RUN /opt/conda/envs/tvb-run/bin/pip uninstall -y matplotlib
RUN /opt/conda/envs/tvb-run/bin/pip install matplotlib==2.1.0 docutils scipy==1.0.0 six==1.11.0 urllib3==1.22
RUN /opt/conda/envs/tvb-run/bin/pip uninstall -y tvb-framework tvb-library tvb-data

# Check why this is not created
RUN mkdir /root/.tvb-temp
RUN mkdir /root/.tvb-temp/logs