#!/usr/bin/env bash

cd ..

python -m pip install --upgrade build

rm -Rf dist
mkdir dist

declare -a folders2pack=("tvb_framework" "tvb_library" "tvb_contrib" "tvb_storage")

if [[ "$1" != "" ]]; then
    echo "Received param: " "$1"
    folders2pack=("$1")
fi

for pipPackage in "${folders2pack[@]}"; do
    echo "============================="
    echo " Packing: " $pipPackage
    echo "============================="

    cd $pipPackage
    python -m build --sdist --wheel

    mv dist/* ../dist/
    rm -R dist
    rm -R build
    cd ..
done

echo "============================="
echo " Packing: tvb-rest-client"
echo "============================="
cd tvb_framework
mv pyproject.toml pyproject_bck.toml
mv pyproject_rest_client.toml pyproject.toml
python -m build --sdist --wheel
mv pyproject.toml pyproject_rest_client.toml
mv pyproject_bck.toml pyproject.toml
mv dist/* ../dist/
rm -R dist
rm -R build
cd ..

echo "============================="
echo " Packing: tvb-bids-monitor"
echo "============================="
cd tvb_framework
mv pyproject.toml pyproject_bck.toml
mv pyproject_bids_monitor.toml pyproject.toml
python -m build --sdist --wheel
mv pyproject.toml pyproject_bids_monitor.toml
mv pyproject_bck.toml pyproject.toml
mv dist/* ../dist/
rm -R dist
rm -R build
cd ..

## After manual check, do the actual deploy on Pypi
# twine upload dist/*
