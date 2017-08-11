# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
A Javascript displayer for connectivity, using hierarchical edge bundle diagrams from d3.js.

.. moduleauthor:: Vlad Farcas <vlad.farcas@codemart.ro>

"""

import json
from tvb.datatypes.connectivity import Connectivity
from tvb.core.adapters.abcdisplayer import ABCDisplayer


class ConnectivityEdgeBundle(ABCDisplayer):
    _ui_name = "Connectivity Edge Bundle View"
    _ui_subsection = "connectivity_edge"

    def get_input_tree(self):
        """
        Inform caller of the data we need as input.
        """
        return [{"name": "connectivity",
                 "type": Connectivity,
                 "label": "Connectivity to be displayed in a hierarchical edge bundle",
                 "required": True
                 }]

    def get_required_memory_size(self, **kwargs):
        """Return required memory."""
        return -1

    def launch(self, connectivity):
        """Construct data for visualization and launch it."""

        pars = {"labels": json.dumps(connectivity.region_labels.tolist()),
                "url_base": ABCDisplayer.paths2url(connectivity, attribute_name="weights", flatten="True")
                }

        return self.build_display_result("connectivity_edge_bundle/view", pars)
