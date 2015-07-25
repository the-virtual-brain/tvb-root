#!/bin/bash

# This script should be used only to generate PDF documents during development.
# For final distribution, there is a Python script that creates PDFs and  put them
# in the correct place

## Fir some MAC-OS systems, if default locale are not specified, rst2pdf will produce errors.
export LC_ALL='en_US.UTF-8'
export LANG='en_US.UTF-8'

rst2pdf --stylesheets=./styles/pdf_doc.style -o InstallationManual.pdf       InstallationManual/InstallationManual.rst
rst2pdf --stylesheets=./styles/pdf_doc.style -o UserGuide.pdf                UserGuide/UserGuide.rst
rst2pdf --stylesheets=./styles/pdf_doc.style -o DeveloperReferenceManual.pdf DeveloperReference/DeveloperReferenceManual.rst
rst2pdf --stylesheets=./styles/pdf_doc.style -o ContributorsManual.pdf       ContributorsManual/ContributorsManual.rst
