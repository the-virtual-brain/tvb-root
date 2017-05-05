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
import numpy
from tvb.basic.filters.chain import UIFilter, FilterChain
from tvb.basic.traits.core import KWARG_FILTERS_UI
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.entities.storage import dao
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.datatypes.surfaces import FaceSurface
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.surfaces import Surface


def prepare_shell_surface_urls(project_id, shell_surface=None, preferred_type=FaceSurface):

    if shell_surface is None:
        shell_surface = dao.try_load_last_entity_of_type(project_id, preferred_type)

        if not shell_surface:
            raise Exception('No Face object found in current project.')

    face_vertices, face_normals, _, face_triangles = shell_surface.get_urls_for_rendering()
    return json.dumps([face_vertices, face_normals, face_triangles])



class SurfaceViewer(ABCDisplayer):
    """
    Static SurfaceData visualizer - for visual inspecting imported surfaces in TVB.
    Optionally it can display associated RegionMapping entities.
    """
    _ui_name = "Surface Visualizer"
    _ui_subsection = "surface"

    def get_input_tree(self):
        # todo: filter connectivity measures: same length as regions and 1-dimensional

        filters_ui = [UIFilter(linked_elem_name="region_map",
                               linked_elem_field=FilterChain.datatype + "._surface"),
                      # UIFilter(linked_elem_name="connectivity_measure",
                      #          linked_elem_field=FilterChain.datatype + "._surface")
                      ]

        json_ui_filter = json.dumps([ui_filter.to_dict() for ui_filter in filters_ui])

        return [{'name': 'surface', 'label': 'Brain surface',
                 'type': Surface, 'required': True,
                 'description': '', KWARG_FILTERS_UI: json_ui_filter},
                {'name': 'region_map', 'label': 'Region mapping',
                 'type': RegionMapping, 'required': False,
                 'description': 'A region map'},
                {'name': 'connectivity_measure', 'label': 'Connectivity measure',
                 'type': ConnectivityMeasure, 'required': False,
                 'description': 'A connectivity measure',
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1])},
                {'name': 'shell_surface', 'label': 'Shell Surface',
                 'type': Surface, 'required': False,
                 'description': "Face surface to be displayed semi-transparently, for orientation only."}]


    @staticmethod
    def _compute_surface_params(surface, region_map):
        rendering_urls = []
        # we want the URLs in json
        # But these string are going to be verbatim strings in js source code
        # This means that js will interpret escapes like \" so the json parser gets "
        # Double escape is needed \\"
        for url in surface.get_urls_for_rendering(True, region_map):
            escaped_url = json.dumps(url).replace('\\', '\\\\')
            rendering_urls.append(escaped_url)
        url_vertices, url_normals, url_lines, url_triangles, url_region_map = rendering_urls

        hemisphere_chunk_mask = surface.get_slices_to_hemisphere_mask()
        return dict(urlVertices=url_vertices, urlTriangles=url_triangles, urlLines=url_lines,
                    urlNormals=url_normals, urlRegionMap=url_region_map,
                    biHemispheric=surface.bi_hemispheric, hemisphereChunkMask=json.dumps(hemisphere_chunk_mask))


    def _compute_measure_points_param(self, surface, region_map):
        if region_map is None:
            measure_points_no = 0
            url_measure_points = ''
            url_measure_points_labels = ''
            boundary_url = ''
        else:
            measure_points_no = region_map.connectivity.number_of_regions
            url_measure_points = self.paths2url(region_map.connectivity, 'centres')
            url_measure_points_labels = self.paths2url(region_map.connectivity, 'region_labels')
            boundary_url = surface.get_url_for_region_boundaries(region_map)
        return dict(noOfMeasurePoints=measure_points_no, urlMeasurePoints=url_measure_points,
                    urlMeasurePointsLabels=url_measure_points_labels, boundaryURL=boundary_url)


    def _compute_measure_param(self, connectivity_measure, measure_points_no):
        if connectivity_measure is None:
            # If there is no measure to show then we what to show the region mapping
            # The client will generate a range signal for this use case.
            min_measure = 0
            max_measure = measure_points_no
            client_measure_url = ''
        else:
            if connectivity_measure.nr_dimensions != 1:
                raise ValueError("connectivity measure must be 1 dimensional")
            if connectivity_measure.length_1d != measure_points_no:
                raise ValueError("connectivity measure has %d values but the connectivity has %d "
                                 "regions" % (connectivity_measure.length_1d, measure_points_no))
            min_measure = numpy.min(connectivity_measure.array_data)
            max_measure = numpy.max(connectivity_measure.array_data)
            # We assume here that the index 0 in the measure corresponds to
            # the region 0 of the region map.
            client_measure_url = self.paths2url(connectivity_measure, "array_data")


        return dict(minMeasure=min_measure, maxMeasure=max_measure, clientMeasureUrl=client_measure_url)


    def launch(self, surface, region_map=None, connectivity_measure=None,
               shell_surface=None, title="Surface Visualizer"):
        params = dict(title=title, extended_view=False, isOneToOneMapping=False,
                      hasRegionMap=region_map is not None)
        params.update(self._compute_surface_params(surface, region_map))
        params.update(self._compute_measure_points_param(surface, region_map))
        params.update(self._compute_measure_param(connectivity_measure, params['noOfMeasurePoints']))

        try:
            params['shelfObject'] = prepare_shell_surface_urls(self.current_project_id, shell_surface)
        except Exception:
            params['shelfObject'] = None

        return self.build_display_result("surface/surface_view", params,
                                         pages={"controlPage": "surface/surface_viewer_controls"})


    def get_required_memory_size(self):
        return -1



class RegionMappingViewer(SurfaceViewer):
    """
    This is a viewer for RegionMapping DataTypes.
    It reuses almost everything from SurfaceViewer, but it make required another input param.
    """
    _ui_name = "Region Mapping Visualizer"
    _ui_subsection = "surface"

    def get_input_tree(self):
        base_tree = SurfaceViewer.get_input_tree(self)
        base_tree[1]['required'] = True
        base_tree.pop(0)
        return base_tree

    def launch(self, region_map, connectivity_measure=None, shell_surface=None):

        return SurfaceViewer.launch(self, region_map.surface, region_map, connectivity_measure, shell_surface,
                                    title=RegionMappingViewer._ui_name)


class ConnectivityMeasureOnSurfaceViewer(SurfaceViewer):
    """
    This displays a connectivity measure on a surface via a RegionMapping
    It reuses almost everything from SurfaceViewer, but it make required another input param.
    """
    _ui_name = "Connectivity Measure Surface Visualizer"
    _ui_subsection = "surface"

    def get_input_tree(self):
        base_tree = SurfaceViewer.get_input_tree(self)
        base_tree[2]['required'] = True
        tree = [base_tree[2], base_tree[1], base_tree[3]]
        return tree

    def launch(self, connectivity_measure, region_map=None, shell_surface=None):
        if (region_map is None or region_map.connectivity.number_of_regions !=
                connectivity_measure.connectivity.number_of_regions):
            # We have no regionmap or the onw we have is not compatible with the measure.
            # first try to find the associated region map
            region_maps = dao.get_generic_entity(RegionMapping, connectivity_measure.connectivity.gid, '_connectivity')
            if region_maps:
                region_map = region_maps[0]
                # else: todo fallback on any region map with the right number of nodes
        return SurfaceViewer.launch(self, region_map.surface, region_map, connectivity_measure, shell_surface,
                                    title=connectivity_measure.display_name)
