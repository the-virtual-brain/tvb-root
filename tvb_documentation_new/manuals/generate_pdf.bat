REM This file should be used only to generate quickly PDF files while writing them. 
REM For the official release there is a Python script (doc_generator.py) which creates PDF
REM files and place them on the correct folder to be included into distribution.

cls
del *.pdf

rst2pdf --stylesheets=./styles/pdf_doc.style -o InstallationManual.pdf       InstallationManual/InstallationManual.rst
rst2pdf --stylesheets=./styles/pdf_doc.style -o UserGuide.pdf                UserGuide/UserGuide.rst
rst2pdf --stylesheets=./styles/pdf_doc.style -o DeveloperReferenceManual.pdf DeveloperReference/DeveloperReferenceManual.rst
rst2pdf --stylesheets=./styles/pdf_doc.style -o ContributorsManual.pdf       ContributorsManual/ContributorsManual.rst
