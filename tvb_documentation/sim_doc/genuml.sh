#!/bin/sh
set -o verbose

#Reverse engineer the tvb code... ;-)
pyreverse -p uml ../../
#../*py 

#Clean-up first, if you don't it can cause problems...
##rm img/classes.png
##rm img/packages.png
rm img/classes.svg
rm img/packages.svg
rm img/classes.fig
rm img/packages.fig

###Bitmapped
##dot -Tpng classes_uml.dot > img/classes.png
##dot -Tpng packages_uml.dot > img/packages.png

#Vector
dot -Tsvg classes_uml.dot > img/classes.svg
dot -Tsvg packages_uml.dot > img/packages.svg

#Editable
dot -Tfig classes_uml.dot > img/classes.fig
dot -Tfig packages_uml.dot > img/packages.fig

#Clean up
rm *dot
