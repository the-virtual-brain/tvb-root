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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
# Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
# Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
# The Virtual Brain: a simulator of primate brain network dynamics.
# Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json
from tvb.basic.profile import TvbProfile
from tvb.basic.traits.core import KWARG_FILTERS_UI
from tvb.basic.filters.chain import FilterChain, UIFilter
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.storage import dao
from tvb.datatypes.annotations import ConnectivityAnnotations
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionMapping


class ConnectivityAnnotationsView(ABCDisplayer):
    """
    Given a Connectivity Matrix and a Surface data the viewer will display the matrix 'inside' the surface data.
    The surface is only displayed as a shadow.
    """
    _ui_name = "Annotations Visualizer"
    _ui_subsection = "annotations"


    def get_input_tree(self):
        """
        Take as Input a Connectivity Object.
        """

        filters_ui = [UIFilter(linked_elem_name="annotations",
                               linked_elem_field=FilterChain.datatype + "._connectivity"),
                      UIFilter(linked_elem_name="region_map",
                               linked_elem_field=FilterChain.datatype + "._connectivity"),
                      UIFilter(linked_elem_name="connectivity_measure",
                               linked_elem_field=FilterChain.datatype + "._connectivity")]

        json_ui_filter = json.dumps([ui_filter.to_dict() for ui_filter in filters_ui])

        return [{'name': 'connectivity', 'label': 'Connectivity Matrix', 'type': Connectivity,
                 'required': False, KWARG_FILTERS_UI: json_ui_filter},  # Used for filtering

                {'name': 'annotations', 'label': 'Ontology Annotations',
                 'type': ConnectivityAnnotations, 'required': True},
                {'name': 'region_map', 'label': 'Region mapping', 'type': RegionMapping, 'required': False,
                 'description': 'A region map to identify us the Cortical Surface to display ans well as '
                                'how the mapping from Connectivity to Cortex is done '}]


    def get_required_memory_size(self, **kwargs):
        return -1


    def launch(self, annotations, region_map=None, **kwarg):

        if region_map is None:
            region_map = dao.get_generic_entity(RegionMapping, annotations.connectivity.gid, '_connectivity')
            if len(region_map) < 1:
                raise LaunchException(
                    "Can not launch this viewer unless we have at least a RegionMapping for the current Connectivity!")
            region_map = region_map[0]

        boundary_url = region_map.surface.get_url_for_region_boundaries(region_map)
        url_vertices_pick, url_normals_pick, url_triangles_pick = region_map.surface.get_urls_for_pick_rendering()
        url_vertices, url_normals, _, url_triangles, url_region_map = \
            region_map.surface.get_urls_for_rendering(True, region_map)

        params = dict(title="Connectivity Annotations Visualizer",
                      baseUrl=TvbProfile.current.web.BASE_URL,
                      annotationsTreeUrl=self.paths2url(annotations, 'tree_json'),
                      urlTriangleToRegion=self.paths2url(region_map, "get_triangles_mapping"),
                      urlActivationPatterns=self.paths2url(annotations, "get_activation_patterns"),

                      minValue=0,
                      maxValue=annotations.connectivity.number_of_regions - 1,
                      urlColors=json.dumps(url_region_map),

                      urlVerticesPick=json.dumps(url_vertices_pick),
                      urlTrianglesPick=json.dumps(url_triangles_pick),
                      urlNormalsPick=json.dumps(url_normals_pick),
                      brainCenter=json.dumps(region_map.surface.center()),

                      urlVertices=json.dumps(url_vertices),
                      urlTriangles=json.dumps(url_triangles),
                      urlNormals=json.dumps(url_normals),
                      urlRegionBoundaries=boundary_url)

        return self.build_display_result("annotations/annotations_view", params,
                                         pages={"controlPage": "annotations/annotations_controls"})
