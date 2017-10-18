#!/usr/bin/env bash
rm -Rf dist
mkdir dist

for pipPackage in "tvb_data" "framework_tvb" "scientific_library"; do
    echo "Packing: " $pipPackage
    echo "========================"
    cd $pipPackage
    python setup.py sdist
    python setup.py bdist_wheel
    mv dist/* ../dist/
    rm -R dist
    rm -R build
    cd ..
done

## After manual check, do the actual deploy on Pypi
# twine upload dist/*