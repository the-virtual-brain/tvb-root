#!/bin/bash

export LC_ALL='en_US.UTF-8'
export LANG='en_US.UTF-8'

rm -R ../dist/

cd ../tvb_documentation/tvb_documentor/
python doc_generator.py

cd ../doc_site/
make -f MakeFile html

cp -R ../../dist/api _build/html/

cd _build/
mv html tvb-documentation-site
zip -r tvb-documentation-site.zip tvb-documentation-site

mv tvb-documentation-site.zip ../../../

cd ..
rm -R _build
