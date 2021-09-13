FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

RUN apt-get update && apt-get install -y git build-essential python3-pip

RUN pip install numba scipy 'numpy<=1.20' networkx scikit-learn cython numexpr psutil pytest pytest-cov pytest-xdist pytest-benchmark pytest-mock matplotlib h5py>=2.10 typing BeautifulSoup4 subprocess32 flask-restx python-keycloak mako pyAesCrypt pyunicore formencode cfflib jinja2 nibabel sqlalchemy alembic allensdk sphinx==1.2.3 docutils==0.12 werkzeug flask gevent jupyter cherrypy autopep8 pylems lxml pycuda
RUN pip install tvb_data

WORKDIR /work
ENV RUNNER_ALLOW_RUNASROOT=1
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
# this is built and then run on the rtx workstation with
# an already configured github self-hosted runner with the command
# docker run --gpus all --rm -it -v $PWD:/work -w /work tvb/mrs-ci-rtx bash run.sh