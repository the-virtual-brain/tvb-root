# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
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

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
import os
import re
import six
import logging
from tvb.basic.logger.builder import GLOBAL_LOGGER_BUILDER, get_logger


def log_debug(debug=False, timestamp=False, prefix=''):
    level_name = 'DEBUG' if debug else 'INFO'
    level = getattr(logging, level_name)
    GLOBAL_LOGGER_BUILDER.set_loggers_level(level)
    LOG = get_logger(__name__)
    for handler in LOG.root.handlers:
        handler.setLevel(level)
        # reset formatter more friendly for console work
        if isinstance(handler, logging.StreamHandler) and not timestamp:
            if prefix:
                prefix += ' '
            handler.setFormatter(logging.Formatter(prefix + '%(levelname)07s  %(message)s'))
    LOG.info('log level set to %s' % (level_name,))


def astr(ary):
    """Make short str repr of numerical value."""
    if isinstance(ary, numpy.ndarray):
        if ary.size == 1:
            val = ary[0]
        else:
            val = 'ndarray(%s, %s)' % (ary.shape, ary.dtype)
    elif isinstance(ary, bool):
        val = str(ary)
    elif isinstance(ary, float) or isinstance(ary, six.integer_types):
        val = ary
    else:
        val = str(ary)

    if isinstance(val, str):
        return val
    else:
        is_py_int = isinstance(val, six.integer_types)
        is_np_int = hasattr(val, 'dtype') and numpy.issubdtype(ary.dtype, numpy.integer)
        if is_py_int or is_np_int:
            return '%d' % (val,)
        else:
            return '%g' % (val,)


def map_astr(self, names):
    """Helper for generating a sequence of astr representation of attributes on self"""
    strs = []
    for name in names.split():
        strs.append(astr(getattr(self, name)))
    return tuple(strs)


def simple_gen_astr(self, names):
    """Helper for generating str for object with only numerical attributes."""
    strs = []
    for name, str_ in zip(names.split(), map_astr(self, names)):
        strs.append('%s=%s' % (name, str_))
    clsname = self.__class__.__name__
    return '%s(%s)' % (clsname, ', '.join(strs))


# workaround lack of ufunc at method for older NumPy versions
def _add_at(dest, map, src):
    for i in numpy.unique(map):
        dest[i] += src[i == map].sum(axis=0)
    return dest


try:
    numpy_add_at = numpy.add.at
except AttributeError:
    numpy_add_at = _add_at

# loose couple psutil so it's an optional dependency
try:
    import psutil
except ImportError:
    msg = """psutil module not available: no warnings will be issued when a
    simulation may require more memory than available"""
    LOG = get_logger(__name__)
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
    >>> print(parameters.y)
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


# FIXME: this may not work yet: write a numpy array subclass that takes care of this
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
    allt, allv, allr = list(map(arange, history.shape))
    # ISomething like(?):
    # return numpy.roll(history, step, axis=0) 
    return history[(allt + step) % horizon, allv, allr]


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
        return self.raw[(idx + self.step) % self.horizon, :, :]

    def __setindex__(self, idx, rawin):
        self.raw[(idx + self.step) % self.horizon, :, :] = rawin


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


def total_ms(duration="", hours=0, minutes=0, seconds=0):
    re_times = re.compile(r'\s*(\d+\.?\d*)\s*(sec|ms|hr|min|s|m|h)\s*$')
    re_decimals = re.compile(r'^[0-9]+([,.][0-9]+)?$')
    total_milliseconds = 0
    if duration == "":
        if not hours == 0:
            total_milliseconds = hours * 3600000
        elif not minutes == 0:
            total_milliseconds = minutes * 60000
        elif not seconds == 0:
            total_milliseconds = seconds * 1000
    elif re_decimals.search(duration):
        total_milliseconds = float(duration)
    else:
        match = re_times.match(duration)
        if not match:
            raise ValueError("unable to parse duration %r" % duration)
        s_time, s_unit = match.group(1, 2)
        s_time = float(s_time)
        if s_unit in ('s', 'sec'):
            total_milliseconds = s_time * 1000
        elif s_unit in ('m', 'min'):
            total_milliseconds = s_time * 60000
        elif s_unit in ('hr', 'h'):
            total_milliseconds = s_time * 3600000
        elif s_unit in ('ms'):
            total_milliseconds = s_time
    return total_milliseconds
