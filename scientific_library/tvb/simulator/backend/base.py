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
Simulator backends provide pluggable numerical implementations,
allowing for different implementations or strategies.

- Follows roughly the DAO pattern which isolates the service layer from database API
  - current simulator+datatypes is the compute "service"
  - backends back the compute service
  - service layer passes traited types into backend
  - backend handles the translation to a particular compute api (numpy, tf)
- Isolates array creation, tracing, assist switching in float32
- 'Service' layer receives closures or generator
- Backend specifies preferred types and array creation routines
- Components can then be associated with a component, e.g.
  - nest component uses nest backend
  - field componetn uses shtns backend
- Isolates array creation, tracing, assist switching in float32
- 'Service' layer receives closures or generator
- Backend specifies preferred types and array creation routines
- Components can then be associated with a component, e.g.
  - nest component uses nest backend
  - field componetn uses shtns backend
- Multibackend-multicomponents need conversions done

"""


class BaseBackend:
    "Type tag for backends."

    @staticmethod
    def default(self):
        "Get default backend."
        # TODO later allow for configuration
        from .ref import ReferenceBackend
        return ReferenceBackend()
