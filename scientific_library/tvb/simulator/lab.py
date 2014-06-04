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
tvb.simulator.lab is a umbrella module designed to make console and scripting
work easier by importing all the simulator pieces at once.

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import os
import pdb
from time import time
from numpy import * # for load & save not available in pylab
import numpy as np
import numpy

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(["-profile", "LIBRARY_PROFILE"], try_reload=False)

from tvb.simulator import (simulator, models, coupling, integrators, monitors, noise)
from tvb.datatypes import connectivity, surfaces, equations, patterns, sensors

from tvb.simulator.common import get_logger
LOG = get_logger(__name__)


PDB = lambda: pdb.set_trace()


def file_exists(fname):
    """
    file_exists(fname) is a convenience function to test whether a file
    already exists or not. Returns True if we can stat the file, otherwise
    False.

    """
    try:
        os.stat(fname)
        return True
    except OSError:
        return False


def timed(fn, t=time):
    fn.times = []
    @functools.wraps(fn)
    def wrapper(*args, **kwds):
        tic = t()
        ret = fn(*args, **kwds)
        toc = t()
        fn.times.append(toc - tic)
        return ret
    return wrapper


# try to import plotting tools and matplotlib
try:
    from tvb.simulator.plot.tools import *
    from matplotlib.pyplot import *
except ImportError as exc:
    LOG.warning("Plotting tools will not be available: %s:", exc)

