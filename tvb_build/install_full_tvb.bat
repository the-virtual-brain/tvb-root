@echo off

rem Use this script to install TVB from the main code sources repo, to the current python installation
rem set up external dependencies that we maintain as distutils packages in externals/
cd ..

cd tvb_framework
pip install -e . --no-deps
cd ..

cd tvb_library
pip install -e .
cd ..

cd tvb_storage
pip install -e .
cd ..

cd tvb_contrib
pip install -e . --no-deps
cd ..

cd tvb_bin
pip install -e .
cd ..

cd tvb_build
pip install -e . --no-deps
