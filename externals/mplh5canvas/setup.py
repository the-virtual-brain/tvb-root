#!/usr/bin/env python
"""
Install an HTML5 backcend for Maplotlib
"""
from setuptools import setup, find_packages
from matplotlib import __version__
import sys

if tuple([int(x) for x in __version__.split(".")[:2]]) < (0, 99, 1):
    print "The HTML 5 Canvas Backend requires matplotlib 0.99.1.1 or newer. \
           Your version (%s) appears older than this. Unable to continue..." \
           % __version__
    sys.exit(0)

setup (
    name = "mplh5canvas",
    version = "trunk",
    description = "Matplotlib HTML 5 Canvas Backend",
    author = "Simon Ratcliffe, Ludwig Schwardt",
    packages = find_packages(),
    scripts = [],
    zip_safe = False,
)
