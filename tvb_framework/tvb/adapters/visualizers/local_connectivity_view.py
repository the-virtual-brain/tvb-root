# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import json

from tvb.adapters.datatypes.db.local_connectivity import LocalConnectivityIndex
from tvb.adapters.visualizers.surface_view import SurfaceURLGenerator
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.local_connectivity import LocalConnectivity


class LocalConnectivityViewerModel(ViewModel):
    local_conn = DataTypeGidAttr(
        linked_datatype=LocalConnectivity,
        label='Local connectivity'
    )


class LocalConnectivityViewerForm(ABCAdapterForm):
    def __init__(self):
        super(LocalConnectivityViewerForm, self).__init__()
        self.local_conn = TraitDataTypeSelectField(LocalConnectivityViewerModel.local_conn, name='local_conn',
                                                   conditions=self.get_filters())

    @staticmethod
    def get_view_model():
        return LocalConnectivityViewerModel

    @staticmethod
    def get_required_datatype():
        return LocalConnectivityIndex

    @staticmethod
    def get_input_name():
        return 'local_conn'

    @staticmethod
    def get_filters():
        return None


class LocalConnectivityViewer(ABCDisplayer):
    """
    Local connectivity visualizer
    """
    _ui_name = "Local Connectivity Visualizer"
    _ui_subsection = "connectivity_local"

    def get_form_class(self):
        return LocalConnectivityViewerForm

    def _compute_surface_params(self, surface_h5):
        url_vertices_pick, url_normals_pick, url_triangles_pick = SurfaceURLGenerator.get_urls_for_pick_rendering(
            surface_h5)
        url_vertices, url_normals, _, url_triangles, _ = SurfaceURLGenerator.get_urls_for_rendering(surface_h5)

        return {
            'urlVerticesPick': json.dumps(url_vertices_pick),
            'urlTrianglesPick': json.dumps(url_triangles_pick),
            'urlNormalsPick': json.dumps(url_normals_pick),
            'urlVertices': json.dumps(url_vertices),
            'urlTriangles': json.dumps(url_triangles),
            'urlNormals': json.dumps(url_normals),
            'brainCenter': json.dumps(surface_h5.center())
        }

    def launch(self, view_model):
        params = dict(title="Local Connectivity Visualizer", extended_view=False,
                      isOneToOneMapping=False, hasRegionMap=False)

        local_conn_h5 = h5.h5_file_for_gid(view_model.local_conn)
        with local_conn_h5:
            surface_gid = local_conn_h5.surface.load()
            min_value, max_value = local_conn_h5.get_min_max_values()

        with h5.h5_file_for_gid(surface_gid) as surface_h5:
            params.update(self._compute_surface_params(surface_h5))

        params['local_connectivity_gid'] = view_model.local_conn.hex
        params['minValue'] = min_value
        params['maxValue'] = max_value
        return self.build_display_result("local_connectivity/view", params,
                                         pages={"controlPage": "local_connectivity/controls"})

    def get_required_memory_size(self):
        return -1
