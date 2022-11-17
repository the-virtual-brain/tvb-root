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
A Stan backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

from .templates import MakoUtilMix
from tvb.simulator.lab import *


class StanBackend(MakoUtilMix):

    def generate_stan_header(self, sim):
        template = '<%include file="stan_inc.hpp.mako"/>'
        content = dict(sim=sim)
        header = self.render_template(template, content)
        return header

    def generate_stan_model(self, sim):
        # this will be rather generic?
        pass

    # it would make more sense to generate plain Stan code first, then
    # worry about cxx in a second step, even if we have a PoC

    # this should rely on API changes to provide:
    # - parameters of interest
    # - observed data (some monitors, not others; maybe FC or spectral feature)

    def run_sim(self, sim: simulator.Simulator):
        "Run the Stan model."
        self.generate_stan_header(sim)