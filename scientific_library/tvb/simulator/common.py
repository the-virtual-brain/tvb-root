# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
A module of classes and functions of common use.

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
import os
import zipfile

# route framework imports through this module so they are more easily updated

def get_logger(name):
    try:
        from tvb.basic.logger.builder import get_logger
        return get_logger(name)
    except ImportError:
        import logging
        return logging.getLogger(name)

LOG = get_logger(__name__)

# loose couple psutil so it's an optional dependency
try:
    import psutil
except ImportError:
    msg  = """psutil module not available: no warnings will be issued when a
    simulation may require more memory than available"""
    LOG.warning(msg)
    psutil = None

class Struct(dict):
    """
    the Struct class is a dictionary with matlab/C struct-like access
    to its fields:

    >>> parameters = Struct(x=23.4345, alpha=1.522e-4)
    >>> parameters.x + 1
    24.4345
    >>> parameters.x_init = 6
    >>> parameters.x_init + 1
    7
    >>> print parameters.y
    None

    note that this class returns None if the field does not exist!

    """

    def __getattr__(self, attr):
        return self.get(attr, None)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def linear_interp1d(start_time, end_time, start_value, end_value, interp_point):
    """
    performs a one dimensional linear interpolation using two 
    timepoints (start_time, end_time) for two floating point (possibly
    NumPy arrays) states (start_value, end_value) to return state at timepoint
    start_time < interp_point < end_time.

    """

    mean = (end_value - start_value) / (end_time - start_time)
    return start_value + (interp_point - start_time) * mean


def heaviside(array):
    """
    heaviside() returns 1 if argument > 0, 0 otherwise.

    The copy operation here is necessary to ensure that the
    array passed in is not modified.

    """

    if type(array) == numpy.float64:
        return 0.0 if array < 0.0 else 1.0
    else:
        ret = array.copy()
        ret[array < 0.0] = 0.0
        ret[array > 0.0] = 1.0
        return ret

# FIXME: this may not work yet
# FIXME: write a numpy array subclass that takes care of this 
#         using indexing magic. makes our life easier.
def unravel_history(history, horizon, step, arange=numpy.arange):
    """
    in our simulator, history is a 3D numpy array where the time 
    dimension is periodic. This means sometimes, the layout is like

        [ ... , t(horizon-1), t(horizon), t(1), t(2), ... ]

    but we may need the correctly ordered version 

        [ t(1), t(2), ... , t(horizon-1), t(horizon) ]

    given some step. This function does that. 
    """
    allt, allv, allr = map(arange, history.shape)
    # ISomething like(?):
    # return numpy.roll(history, step, axis=0) 
    return history[ (allt + step) % horizon, allv, allr ]


def iround(x):
    """
    iround(number) -> integer
    Trying to get a stable and portable result when rounding a number to the 
    nearest integer.

    NOTE: I'm introducing this because of the unstability we found when
    using int(round()). Should use always the same rounding strategy.

    Try :    
    >>> int(4.999999999999999)
    4
    >>> int(4.99999999999999999999)
    5

    """
    y = round(x) - .5
    return int(y) + (y > 0)


class Buffer(object):
    """
    Draft of a history object that allows us to track the current
    state and access the history array in different but consistent 
    ways
    """
    step = 0
    raw = None
    horizon = None

    def __init__(self):
        raise NotImplementedError

    def __getindex__(self, idx):
        return self.raw[(idx + self.step)% self.horizon, :, :]

    def __setindex__(self, idx, rawin):
        self.raw[(idx + self.step)% self.horizon, :, :] = rawin


def zip_directory(path, zip_file):
    """
    Zip a given directory...
    Didn't know where to put this. 
    I need to pack zips from the scripting interface. 
    To avoid duplicating code I leave this small function here. 

    :param path -- where to store the zip
    :param zip_file 
    """
    for dirname, subdirs, files in os.walk(path):
        zip_file.write(dirname)
        for filename in files:
            zip_file.write(os.path.join(dirname, filename))
        zip_file.close()
