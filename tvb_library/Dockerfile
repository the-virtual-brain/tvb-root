FROM continuumio/miniconda3

RUN apt-get -y update && apt-get -y install build-essential gcc

RUN conda install python=3.7 nomkl numba scipy numpy networkx scikit-learn cython pip numexpr psutil \
    pytest pytest-cov pytest-benchmark pytest-mock matplotlib \
    psycopg2 pytables scikit-image==0.14.2 simplejson cherrypy docutils

RUN apt-get install -y zip unzip
RUN cd /opt \
 && wget -q https://zenodo.org/record/4263723/files/tvb_data.zip?download=1 -O tvb_data.zip \
 && mkdir tvb_data \
 && unzip tvb_data.zip -d tvb_data \
 && rm tvb_data.zip \
 && cd tvb_data \
 && python3 setup.py develop

RUN pip install h5py>=2.10 pytest-xdist sqlalchemy mako pylems lxml requests autopep8

RUN pip install --no-build-isolation tvb-gdist

# required for rateML CUDA work
# RUN conda install cudatoolkit \
#  && conda install -c conda-forge cudatoolkit-dev \
#  && pip install pycuda

# docker run --rm -it -v $PWD:/tvb -w /tvb tvb/lib 
# add --gpus all for testing CUDA
CMD ["/bin/bash", "-c", "cd tvb_bin; python3 setup.py develop --no-deps; cd ../tvb_library; python3 setup.py develop --no-deps; python3 -m pytest -n12"]
