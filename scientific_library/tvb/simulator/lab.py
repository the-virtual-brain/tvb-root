# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
tvb.simulator.lab is a umbrella module designed to make console and scripting
work easier by importing all the simulator pieces at once.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import os
import sys

# avoid segfaulting in absence of X11 DISPLAY
if sys.platform in ('linux2', ) and 'DISPLAY' not in os.environ:
    try:
        import matplotlib as mpl
        mpl.use('Agg')
    except Exception:
        pass

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

from tvb.simulator.common import get_logger, log_debug

log_debug(False)

from tvb.simulator import (simulator, models, coupling, integrators, monitors, noise)
from tvb.datatypes import (connectivity, surfaces, equations, patterns, region_mapping, sensors, cortex,
                           local_connectivity, time_series)

from tvb.simulator.plot.tools import (hinton_diagram, plot_3d_centres, plot_connectivity,
                                      plot_fast_kde, plot_local_connectivity, plot_matrix,
                                      plot_pattern, plot_tri_matrix)
from tvb.simulator.plot.utils import generate_region_demo_data

