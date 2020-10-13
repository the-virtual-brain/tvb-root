FROM nvidia/cuda
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get  install -y python3-pip python3-numpy python3-scipy python3-matplotlib
RUN pip3 install tqdm tvb-library tvb-data pycuda
#RUN pip install pycuda
#FROM continuumio/miniconda3
#
#RUN apt-get -y update && apt-get -y install build-essential gcc
#
#RUN conda install python=3.7 nomkl numba scipy numpy networkx scikit-learn cython pip numexpr psutil \
#    pytest pytest-cov pytest-benchmark pytest-mock matplotlib \
#    psycopg2 pytables scikit-image==0.14.2 simplejson cherrypy docutils
#
#RUN apt-get install -y zip
#RUN cd /opt \
# && wget -q https://zenodo.org/record/3688773/files/tvb-data.zip?download=1 -O tvb_data.zip \
# && mkdir tvb_data \
# && unzip tvb_data.zip -d tvb_data \
# && rm tvb_data.zip \
# && cd tvb_data \
# && python3 setup.py develop
#
#RUN pip install h5py>=2.10 tvb-gdist pytest-xdist
#
#RUN pip install mako
#
#RUN conda install cudatoolkit
#RUN pip install pycuda