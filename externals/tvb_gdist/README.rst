External Library Geodesic
=========================

The `gdist` module is a Cython interface to a C++ library
(http://code.google.com/p/geodesic/) for computing
geodesic distance which is the length of shortest line between two
vertices on a triangulated mesh in three dimensions, such that the line
lies on the surface.

The algorithm is due Mitchell, Mount and Papadimitriou, 1987; the implementation
is due to Danil Kirsanov and the Cython interface to Gaurav Malhotra and
Stuart Knock.


Original library (published under MIT license):
http://code.google.com/p/geodesic/

We added a python wrapped and made small fixes to the original library, to make it compatible with cython.

To install this, either run `pip install gdist` or download
sources from Github and run `python setup.py install` in current folder.

Basic test could be::

    python
    import gdist


Python 2.7, Cython, and a C++ compiler are required.

Debian package
==============

In order to produce a Debian package, assuming you have the requisite tools
installed (`apt-get install devscripts python-all-dev python-stdeb`)::

    cd debian
    debuild -us -uc
    cd ../../


and you should find a suitable deb file for your system.
