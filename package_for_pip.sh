#!/usr/bin/env bash

rm -Rf dist
mkdir dist

declare -a folders2pack=("tvb_data" "framework_tvb" "scientific_library" "externals/geodesic_distance")
if [ "$1" != "" ]; then
    echo "Received param: " "$1"
    folders2pack=("$1")
fi

echo "============================="
echo " Generating revision number: "
echo "============================="

svnVersion=$(svnversion .)
destFile="framework_tvb/tvb/config/tvb.version"
rm $destFile
echo "$svnVersion" > "$destFile"
echo "Found: " $svnVersion ", written into: " $destFile


for pipPackage in "${folders2pack[@]}"; do

    echo "============================="
    echo " Packing: " $pipPackage
    echo "============================="

    cd $pipPackage
    python setup.py sdist
    python setup.py bdist_wheel

    if [ -d "../dist/" ]; then
        mv dist/* ../dist/
    else
        mv dist/* ../../dist/
    fi

    rm -R dist
    rm -R build

    if [ -d "../dist/" ]; then
        cd ..
    else
        cd ../..
    fi
done

## After manual check, do the actual deploy on Pypi
# twine upload dist/*