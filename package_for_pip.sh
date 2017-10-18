#!/usr/bin/env bash

rm -Rf dist
mkdir dist

for pipPackage in "tvb_data" "framework_tvb" "scientific_library" "externals/geodesic_distance"; do

    echo "============================="
    echo "Packing: " $pipPackage
    echo "============================="

    cd $pipPackage
    python setup.py sdist
    python setup.py bdist_wheel

    if [ -d "../dist/" ]
    then
        mv dist/* ../dist/
    else
        mv dist/* ../../dist/
    fi

    rm -R dist
    rm -R build

    if [ -d "../dist/" ]
    then
        cd ..
    else
        cd ../..
    fi
done

## After manual check, do the actual deploy on Pypi
# twine upload dist/*