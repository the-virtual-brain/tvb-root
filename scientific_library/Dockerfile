FROM continuumio/miniconda3

RUN apt-get -y update && apt-get -y install build-essential gcc

RUN conda install python=3.7 nomkl numba scipy numpy networkx scikit-learn cython pip numexpr psutil \
    pytest pytest-cov pytest-benchmark pytest-mock matplotlib \
    psycopg2 pytables scikit-image==0.14.2 simplejson cherrypy docutils

RUN apt-get install -y zip
RUN cd /opt \
 && wget -q https://zenodo.org/record/3688773/files/tvb-data.zip?download=1 -O tvb_data.zip \
 && mkdir tvb_data \
 && unzip tvb_data.zip -d tvb_data \
 && rm tvb_data.zip \
 && cd tvb_data \
 && python3 setup.py develop

RUN pip install h5py>=2.10 tvb-gdist pytest-xdist
