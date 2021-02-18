FROM nvidia/cuda:11.2.1-devel 

RUN apt-get -y update && apt-get -y install build-essential gcc wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda

ENV PATH=/opt/conda/bin:$PATH

RUN conda install python=3.7 nomkl numba scipy numpy networkx scikit-learn cython pip numexpr psutil \
    pytest pytest-cov pytest-benchmark pytest-mock matplotlib \
    psycopg2 pytables scikit-image==0.14.2 simplejson cherrypy docutils

RUN apt-get install -y zip
RUN cd /opt \
 && wget -q https://zenodo.org/record/4263723/files/tvb_data.zip?download=1 -O tvb_data.zip \
 && mkdir tvb_data \
 && unzip tvb_data.zip -d tvb_data \
 && rm tvb_data.zip \
 && cd tvb_data \
 && python3 setup.py develop

RUN pip install h5py==2.10 tvb-gdist pytest-xdist sqlalchemy mako requests tqdm

RUN pip install pycuda

RUN apt-get install -y git

# binary incompatibility?
RUN pip uninstall -y tvb-gdist

RUN pip install tvb-gdist --no-wheel