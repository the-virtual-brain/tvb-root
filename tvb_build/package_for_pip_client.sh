#!/usr/bin/env bash

cd ..

rm -Rf dist
mkdir dist

echo "============================="
echo " Packing tvb-rest-client "
echo "============================="

cd framework_tvb
python setup_client.py sdist
python setup_client.py bdist_wheel

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


## After manual check, do the actual deploy on Pypi
# twine upload dist/*