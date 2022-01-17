FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-get update && apt-get install -y git build-essential python3-pip wget

RUN pip install numba scipy 'numpy<=1.20' networkx scikit-learn cython numexpr psutil \
        pytest pytest-cov pytest-xdist pytest-benchmark pytest-mock matplotlib h5py>=2.10 \
        typing BeautifulSoup4 subprocess32 flask-restx python-keycloak mako pyAesCrypt \
        pyunicore formencode cfflib jinja2==2.11.3 nibabel sqlalchemy alembic allensdk \
        sphinx==1.2.3 docutils==0.12 werkzeug flask==1.1.4 gevent jupyter cherrypy autopep8 \
        pylems lxml pycuda

RUN apt-get install -y zip
RUN wget -q https://zenodo.org/record/4263723/files/tvb_data.zip?download=1 -O tvb_data.zip \
 && mkdir tvb_data \
 && unzip tvb_data.zip -d tvb_data \
 && rm tvb_data.zip

RUN ln -s $(which python3) /usr/bin/python
RUN apt-get install -y libpq-dev wget

ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y icu-devtools

WORKDIR /work
ENV RUNNER_ALLOW_RUNASROOT=1
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=false
# this is built and then run on the rtx workstation with
# an already configured github self-hosted runner with the command
# docker run --gpus all --rm -it -v $PWD:/work -w /work tvb/mrs-ci-rtx bash run.sh
