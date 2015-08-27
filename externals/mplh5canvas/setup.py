#!/usr/bin/env python

"""
Install an HTML5 backcend for Maplotlib
"""

from setuptools import setup, find_packages


setup(
    name="mplh5canvas",
    version="1.0tvb",
    description="Matplotlib HTML 5 Canvas Backend",
    author="Simon Ratcliffe, Ludwig Schwardt",
    packages=find_packages(),
    scripts=[],
    zip_safe=False,
    install_requires=["matplotlib>=1.2.1"]
)
