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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import json
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.local_connectivity import LocalConnectivity


class LocalConnectivityViewer(ABCDisplayer):
    """
    Local connectivity visualizer
    """
    _ui_name = "Local Connectivity Visualizer"
    _ui_subsection = "connectivity_local"


    def get_input_tree(self):
        return [{'name': 'local_conn', 'label': 'Local connectivity',
                 'type': LocalConnectivity, 'required': True}]

    def _compute_surface_params(self, surface):
        url_vertices_pick, url_normals_pick, url_triangles_pick = surface.get_urls_for_pick_rendering()
        url_vertices, url_normals, _, url_triangles = surface.get_urls_for_rendering()

        return {
            'urlVerticesPick': json.dumps(url_vertices_pick),
            'urlTrianglesPick': json.dumps(url_triangles_pick),
            'urlNormalsPick': json.dumps(url_normals_pick),
            'urlVertices': json.dumps(url_vertices),
            'urlTriangles': json.dumps(url_triangles),
            'urlNormals': json.dumps(url_normals),
            'brainCenter': json.dumps(surface.center())
        }

    def launch(self, local_conn):
        params = dict(title="Local Connectivity Visualizer", extended_view=False,
                      isOneToOneMapping=False, hasRegionMap=False)

        params.update(self._compute_surface_params(local_conn.surface))
        params['local_connectivity_gid'] = local_conn.gid
        min_value, max_value = local_conn.get_min_max_values()
        params['minValue'] = min_value
        params['maxValue'] = max_value
        return self.build_display_result("local_connectivity/view", params,
                                         pages={"controlPage": "local_connectivity/controls"})


    def get_required_memory_size(self):
        return -1

