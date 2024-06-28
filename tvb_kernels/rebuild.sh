#!/bin/bash

rm -rf build

cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

pushd build
make
popd

ctypesgen -l build/libtvbk.so src/nodes.h > nodes.py
