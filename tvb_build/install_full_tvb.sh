#!/usr/bin/env bash

# Use this script to install TVB from the main code sources repo, to the current python installation

cd ..

cd tvb_framework
pip install -e . --no-deps --user
cd ..

cd tvb_storage
pip install -e . --no-deps --user
cd ..

cd tvb_library
# non-editable: the compiled nanobind extension ships in the install; an
# editable install would re-run cmake on every tvb import (needs nanobind
# + cmake + ninja at runtime, and its rebuild hook crashes under a
# Jupyter kernel whose sys.stderr has no fileno)
pip install . --no-deps --user
cd ..

cd tvb_contrib
pip install -e . --no-deps --user
cd ..

cd tvb_bin
pip install -e . --user
cd ..

cd tvb_build
pip install -e . --no-deps --user
