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
Cython accelerated versions of some neuronal dynamics models.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>

"""

from tvb.simulator.common import get_logger
from tvb.simulator import models

LOG = get_logger(__name__)

try:
    import tvb._speedups.models as cmodels

    class CEpileptor(models.Epileptor):
        def dfun(self, state_variables, coupling, local_coupling=0.0):
            return cmodels.epileptor_dfun(state_variables, coupling, local_coupling,
                            self.a, self.b, self.c, self.d, self.aa, self.r,
                            self.Kvf, self.Kf, self.tau, self.Iext, self.Iext2,
                            self.slope, self.x0)

except ImportError:
    pass
