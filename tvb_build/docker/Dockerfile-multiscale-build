#
# TVB & NEST simulations environment
#
# Install Python 3.7, NEST, then tvb-library from Pypi
# Finally link tvb-scripts and tvb-nest from Github sources
#

FROM neurodebian:buster
MAINTAINER lia.domide@codemart.ro

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8

RUN apt-get update; apt-get install -y automake libtool build-essential git vim octave \
                       wget python3.7 libpython3.7 libpython3-dev libncurses5-dev libreadline-dev libgsl0-dev cython3 \
                       python3.7-dev python3-pip python3-numpy python3-scipy python3-matplotlib python3-nose \
                       ipython3 python3-venv python-virtualenv python3-pyqt5 python-dev libgsl-dev libncurses-dev \
                       subversion cmake zip unzip; \
    apt-get clean

RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz \
    && tar xf mpich-3.1.4.tar.gz \
    && cd mpich-3.1.4 \
    && ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf mpich-3.1.4 \
    && rm mpich-3.1.4.tar.gz

# Create user different than root
RUN useradd -ms /bin/bash docker
USER docker
ARG HOME=/home/docker
RUN mkdir $HOME/env; mkdir $HOME/packages

# Create a dedicated Python env for working in it
ENV VENV=$HOME/env/neurosci
RUN python3 -m venv $VENV && python3 -m venv --system-site-packages $VENV; \
    $VENV/bin/pip3 install --upgrade pip; \
    $VENV/bin/pip3 install parameters quantities neo "django<1.9" django-tagging future hgapi gitpython sumatra nixio; \
    $VENV/bin/pip3 install --upgrade nose ipython
# django<1.9 is needed for pynest - september 2019 - NEST 2.18

# Install from sources libneurosim (NEST boost dependency)
WORKDIR $HOME/packages
RUN git clone --depth 1 https://github.com/INCF/libneurosim.git; \
    cd libneurosim; ./autogen.sh; \
    mkdir $VENV/build
WORKDIR $VENV/build
RUN mkdir libneurosim; \
    cd libneurosim; \
    PYTHON=$VENV/bin/python $HOME/packages/libneurosim/configure --prefix=$VENV; \
    make; make install; ls $VENV/lib $VENV/include; \
    ln -s /usr/lib/python3.7/config-x86_64-linux-gnu/libpython3.7.so $VENV/lib/

# Install NEST, latest stable version (July 2019)
ARG NEST_VER=2.18.0
ARG NEST=nest-simulator-$NEST_VER
ARG NEST_SRC=$HOME/packages/$NEST
ARG NEST_INSTALL_DIR=$VENV/nest_build
ARG NEST_PYTHON_PREFIX=${NEST_INSTALL_DIR}/lib/python3.7/site-packages
ARG NEST_MODULE_PATH=${NEST_INSTALL_DIR}/lib/nest

WORKDIR $HOME/packages
RUN wget  https://github.com/nest/nest-simulator/archive/v$NEST_VER.tar.gz -O $HOME/packages/$NEST.tar.gz; \
    tar xzf $NEST.tar.gz; rm $NEST.tar.gz; \
    cd $NEST; \
    cmake -DCMAKE_INSTALL_PREFIX=${NEST_INSTALL_DIR} \
#          -Dwith-mpi=ON  \
          -Dwith-python=3 \
          ###-Dwith-music=ON \
          -Dwith-libneurosim=$VENV \
          -DPYTHON_LIBRARY=/usr/lib \
          -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m \
          ${NEST_SRC}; \
    make; make install

# Install nestml
RUN $VENV/bin/pip3 install setuptools; \
    git clone --depth 1 https://github.com/nest/nestml.git; \
    cd nestml; \
    $VENV/bin/python setup.py install

# Install pyNN
# RUN pip install pyNN
#ARG PYNN_VER=0.9.4
#RUN wget https://github.com/NeuralEnsemble/PyNN/archive/${PYNN_VER}.tar.gz -O $HOME/packages/PyNN-${PYNN_VER}.tar.gz; \
#    cd $HOME/packages; \
#    tar xzf PyNN-${PYNN_VER}.tar.gz; rm PyNN-${PYNN_VER}.tar.gz; \
#    cd PyNN-${PYNN_VER}; \
#    $VENV/bin/python setup.py install

# Install TVB specific packages
WORKDIR $HOME/packages
RUN $VENV/bin/pip3 install pandas xarray jupyterlab pytest pytest-cov pytest-benchmark oct2py scikit-learn scipy; \
    $VENV/bin/pip3 install --ignore-installed entrypoints

RUN wget https://zenodo.org/record/3688773/files/tvb-data.zip?download=1 -O tvb_data.zip; \
    mkdir tvb_data; unzip tvb_data.zip -d tvb_data; rm tvb_data.zip; \
    cd tvb_data; \
    $VENV/bin/python setup.py develop

RUN git clone --depth 1 --branch tvb-multiscale-cosim https://github.com/the-virtual-brain/tvb-root.git
ARG LAST_SHA_ROOT=LATEST
RUN cd tvb-root; \
    git pull origin tvb-multiscale-cosim --allow-unrelated-histories; \
    cd scientific_library; \
    $VENV/bin/python setup.py develop;

RUN git clone --depth 1 https://github.com/the-virtual-brain/tvb-multiscale.git
ARG LAST_SHA_MULTISCALE=LATEST
RUN cd tvb-multiscale; \
    git pull; \
    $VENV/bin/python setup.py develop --no-deps;

WORKDIR /home/docker/
# Activate the right python env.
ENV NEST_INSTALL_DIR=${NEST_INSTALL_DIR}
ENV NEST_PYTHON_PREFIX=${NEST_PYTHON_PREFIX}
ENV PYTHONPATH=$PYTHONPATH:$VENV:${NEST_PYTHON_PREFIX}
# Note that the following are automatically reset if NEST is imported
# via a call to  tvb_nest.simulator_nest.nest_factory.load_nest()
ENV NEST_DATA_DIR=${NEST_INSTALL_DIR}/share/nest
ENV SLI_PATH=${NEST_INSTALL_DIR}/share/nest/sli
ENV NEST_MODULE_PATH=${NEST_MODULE_PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${NEST_MODULE_PATH}
ENV NEST_DOC_DIR=${NEST_INSTALL_DIR}/share/doc/nest
RUN echo "source ${VENV}/bin/activate" >> .bashrc


# ----------------------------------------------------------------------------------------------------------------------
# Run the following lines in case you want to install your own NEST module with NEST builders
# The commented lines will install modules in the MODULES_LIST upon building this image
## Install local tvb-multiscale:
#WORKDIR $HOME/packages
#RUN mkdir tvb-multiscale
#COPY ./tvb-multiscale/ tvb-multiscale/
#RUN cd tvb-multiscale; $VENV/bin/python setup.py develop --no-deps
ENV MYMODULES_DIR=$HOME/packages/tvb-multiscale/tvb_nest/nest/modules
ENV MYMODULES_BLD_DIR=$HOME/packages/nest_modules_builds
RUN cp -r ${MYMODULES_DIR} ${MYMODULES_BLD_DIR}
#ARG MYMODULES_LIST="tvb_rate_wongwang iaf_cond_deco2014"
#ARG MYMODULES_DIR=$HOME/packages/tvb-multiscale/tvb_nest/nest/modules
#ARG NEST_CONFIG=${NEST_INSTALL_DIR}/bin/nest-config
#RUN for MYMODULE_NAME in $MYMODULES_LIST; do export MYMODULE_DIR=${MYMODULES_DIR}/${MYMODULE_NAME}; \
#        export MYMODULE_BLD=${MYMODULES_BLD_DIR}/${MYMODULE_NAME}module_bld; \
#        mkdir -p ${MYMODULE_BLD}; cd ${MYMODULE_BLD}; \
#            cmake -Dwith-nest=$NEST_CONFIG ${MYMODULE_DIR}; \
#            make; make install; \
#    done
# ----------------------------------------------------------------------------------------------------------------------

# Final configurations
# tvb42
RUN $VENV/bin/jupyter notebook --generate-config; \
    echo "c.NotebookApp.password='sha1:12bff019c253:9daecd92c2e9bdb10b3b8a06767a74a0fe078d7c'">>/home/docker/.jupyter/jupyter_notebook_config.py

WORKDIR /home/docker/

# Activate the rights python env
ENV PYTHONPATH $VENV
ENV SLI_PATH $VENV
RUN echo "source $VENV/bin/activate" >> .bashrc

# For running examples
#CMD $VENV/bin/python /home/docker/packages/tvb-multiscale/docker/installation_test.py
# CMD $VENV/bin/python /home/docker/packages/tvb-multiscale/tvb_nest/examples/red_ww_exc_io_inh_i.py
# CMD $VENV/bin/python /home/docker/packages/tvb-multiscale/tvb_nest/examples/rate_ww_ampa_nmda_gaba.py

# Test NEST installation
# RUN cd $NEST; make installcheck
# CMD $VENV/bin/python /home/docker/packages/tvb-multiscale/docker/module_build_test.py

# pynestml installation test:
# CMD cd /home/docker/packages/nestml; $VENV/bin/pytest setup.py test

# pyNN-NEST installation test:
# CMD $VENV/bin/python -c "import pyNN.nest as sim; sim.setup(); sim.end(); print(sim)";

#For running tests
#CMD $VENV/bin/pytest /home/docker/packages/tvb-multiscale/tvb_nest/tests

#For running tests with coverage
#CMD $VENV/bin/pytest --cov=nest_tests_coverage /home/docker/packages/tvb-multiscale/tvb_nest/tests

#Run jupyter notebooks
CMD $VENV/bin/jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

#CMD /bin/bash