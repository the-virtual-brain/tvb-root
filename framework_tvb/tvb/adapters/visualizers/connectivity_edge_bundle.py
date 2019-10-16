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
import os
from tvb.basic.neotraits.api import Attr
from tvb.datatypes.connectivity import Connectivity
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.neotraits._forms import DataTypeSelectField
from tvb.interfaces.neocom._h5loader import DirLoader


class ConnectivityEdgeBundleForm(ABCAdapterForm):

    def __init__(self, prefix='', project_id=None):
        super(ConnectivityEdgeBundleForm, self).__init__(prefix)
        self.connectivity = DataTypeSelectField(Attr(field_type=Connectivity,
                                                     label="Connectivity to be displayed in a hierarchical edge bundle"),
                                                self.get_required_datatype(), self, name=self.get_input_name())
        self.project_id = project_id

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex

    @staticmethod
    def get_input_name():
        return 'connectivity'

    @staticmethod
    def get_filters():
        return None


class ConnectivityEdgeBundle(ABCDisplayer):
    _ui_name = "Connectivity Edge Bundle View"
    _ui_subsection = "connectivity_edge"

    form = None

    def get_input_tree(self):
        return None

    def get_form(self):
        if self.form is None:
            return ConnectivityEdgeBundleForm
        return self.form

    def set_form(self, form):
        self.form = form

    def get_required_memory_size(self, **kwargs):
        """Return required memory."""
        return -1

    def launch(self, connectivity):
        """Construct data for visualization and launch it."""

        loader = DirLoader(os.path.join(os.path.dirname(self.storage_path), str(connectivity.fk_from_operation)))
        connectivity_dt = loader.load(connectivity.gid)

        pars = {"labels": json.dumps(connectivity_dt.region_labels.tolist()),
                "url_base": ABCDisplayer.paths2url(connectivity.gid, attribute_name="weights", flatten="True")
                }

        return self.build_display_result("connectivity_edge_bundle/view", pars)
