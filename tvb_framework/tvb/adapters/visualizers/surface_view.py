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
import uuid
from abc import ABCMeta
from six import add_metaclass
from tvb.adapters.visualizers.time_series import ABCSpaceDisplayer
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.datatypes.h5.surface_h5 import SPLIT_PICK_MAX_TRIANGLE, KEY_VERTICES, KEY_START, SurfaceH5
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import URLGenerator
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.load import try_get_last_datatype
from tvb.core.entities.storage import dao
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neocom import h5
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.surfaces import Surface, SurfaceTypesEnum

LOG = get_logger(__name__)


def ensure_shell_surface(project_id, shell_surface=None, preferred_type=SurfaceTypesEnum.FACE_SURFACE.value):
    filter = FilterChain(fields=[FilterChain.datatype + '.surface_type'], operations=["=="],
                         values=[preferred_type])
    if shell_surface is None:
        shell_surface = try_get_last_datatype(project_id, SurfaceIndex, filter)

        if not shell_surface:
            LOG.warning('No object of type %s found in current project.' % preferred_type)

    return shell_surface


class SurfaceURLGenerator(URLGenerator):

    @staticmethod
    def get_urls_for_rendering(surface_h5, region_mapping_gid=None):
        """
        Compose URLs for the JS code to retrieve a surface from the UI for rendering.
        """
        url_vertices = []
        url_triangles = []
        url_normals = []
        url_lines = []
        url_region_map = []
        gid = surface_h5.gid.load().hex
        for i in range(surface_h5.get_number_of_split_slices()):
            param = "slice_number=" + str(i)
            url_vertices.append(URLGenerator.build_h5_url(gid, 'get_vertices_slice', parameter=param, flatten=True))
            url_triangles.append(URLGenerator.build_h5_url(gid, 'get_triangles_slice', parameter=param, flatten=True))
            url_lines.append(URLGenerator.build_h5_url(gid, 'get_lines_slice', parameter=param, flatten=True))
            url_normals.append(URLGenerator.build_h5_url(gid, 'get_vertex_normals_slice',
                                                         parameter=param, flatten=True))
            if region_mapping_gid is None:
                continue

            start_idx, end_idx = surface_h5.get_slice_vertex_boundaries(i)
            url_region_map.append(URLGenerator.build_h5_url(region_mapping_gid, "get_region_mapping_slice",
                                                            flatten=True, parameter="start_idx=" + str(start_idx) +
                                                                                    ";end_idx=" + str(end_idx)))
        if region_mapping_gid:
            return url_vertices, url_normals, url_lines, url_triangles, url_region_map
        return url_vertices, url_normals, url_lines, url_triangles, None

    @staticmethod
    def get_urls_for_pick_rendering(surface_h5):
        """
        Compose URLS for the JS code to retrieve a surface for picking.
        """
        vertices = []
        triangles = []
        normals = []
        number_of_triangles = surface_h5.number_of_triangles.load()
        number_of_split = number_of_triangles // SPLIT_PICK_MAX_TRIANGLE
        if number_of_triangles % SPLIT_PICK_MAX_TRIANGLE > 0:
            number_of_split += 1

        gid = surface_h5.gid.load().hex
        for i in range(number_of_split):
            param = "slice_number=" + str(i)
            vertices.append(URLGenerator.build_h5_url(gid, 'get_pick_vertices_slice', parameter=param, flatten=True))
            triangles.append(URLGenerator.build_h5_url(gid, 'get_pick_triangles_slice', parameter=param, flatten=True))
            normals.append(
                URLGenerator.build_h5_url(gid, 'get_pick_vertex_normals_slice', parameter=param, flatten=True))

        return vertices, normals, triangles

    @staticmethod
    def get_url_for_region_boundaries(surface_gid, region_mapping_gid, adapter_id):
        return URLGenerator.build_url(adapter_id, 'generate_region_boundaries', surface_gid,
                                      parameter='region_mapping_gid=' + region_mapping_gid)


class BaseSurfaceViewerModel(ViewModel):
    region_map = DataTypeGidAttr(
        linked_datatype=RegionMapping,
        required=False,
        label='Region mapping',
        doc='A region map'
    )

    connectivity_measure = DataTypeGidAttr(
        linked_datatype=ConnectivityMeasure,
        required=False,
        label='Connectivity measure',
        doc='A connectivity measure'
    )

    shell_surface = DataTypeGidAttr(
        linked_datatype=Surface,
        required=False,
        label='Shell Surface',
        doc='Face surface to be displayed semi-transparently, for orientation only.'
    )


@add_metaclass(ABCMeta)
class BaseSurfaceViewerForm(ABCAdapterForm):

    def __init__(self):
        super(BaseSurfaceViewerForm, self).__init__()
        self.region_map = TraitDataTypeSelectField(BaseSurfaceViewerModel.region_map, name='region_map')
        conn_filter = FilterChain(
            fields=[FilterChain.datatype + '.ndim', FilterChain.datatype + '.has_surface_mapping'],
            operations=["==", "=="], values=[1, True])
        self.connectivity_measure = TraitDataTypeSelectField(BaseSurfaceViewerModel.connectivity_measure,
                                                             name='connectivity_measure', conditions=conn_filter)
        self.shell_surface = TraitDataTypeSelectField(BaseSurfaceViewerModel.shell_surface, name='shell_surface')

    @staticmethod
    def get_filters():
        return None


class SurfaceViewerModel(BaseSurfaceViewerModel):
    title = 'Surface Visualizer'

    surface = DataTypeGidAttr(
        linked_datatype=Surface,
        label='Brain surface'
    )


class SurfaceViewerForm(BaseSurfaceViewerForm):
    def __init__(self):
        super(SurfaceViewerForm, self).__init__()
        self.surface = TraitDataTypeSelectField(SurfaceViewerModel.surface, name='surface')

    @staticmethod
    def get_view_model():
        return SurfaceViewerModel

    @staticmethod
    def get_required_datatype():
        return SurfaceIndex

    @staticmethod
    def get_input_name():
        return 'surface'


@add_metaclass(ABCMeta)
class ABCSurfaceDisplayer(ABCSpaceDisplayer):

    def generate_region_boundaries(self, surface_gid, region_mapping_gid):
        """
        Return the full region boundaries, including: vertices, normals and lines indices.
        """
        boundary_vertices = []
        boundary_lines = []
        boundary_normals = []

        with h5.h5_file_for_gid(region_mapping_gid) as rm_h5:
            array_data = rm_h5.array_data[:]

        with h5.h5_file_for_gid(surface_gid) as surface_h5:
            for slice_idx in range(surface_h5.get_number_of_split_slices()):
                # Generate the boundaries sliced for the off case where we might overflow the buffer capacity
                slice_triangles = surface_h5.get_triangles_slice(slice_idx)
                slice_vertices = surface_h5.get_vertices_slice(slice_idx)
                slice_normals = surface_h5.get_vertex_normals_slice(slice_idx)
                first_index_in_slice = surface_h5.split_slices.load()[str(slice_idx)][KEY_VERTICES][KEY_START]
                # These will keep track of the vertices / triangles / normals for this slice that have
                # been processed and were found as a part of the boundary
                processed_vertices = []
                processed_triangles = []
                processed_normals = []
                for triangle in slice_triangles:
                    triangle += first_index_in_slice
                    # Check if there are two points from a triangles that are in separate regions
                    # then send this to further processing that will generate the corresponding
                    # region separation lines depending on the 3rd point from the triangle
                    rt0, rt1, rt2 = array_data[triangle]
                    if rt0 - rt1:
                        reg_idx1, reg_idx2, dangling_idx = 0, 1, 2
                    elif rt1 - rt2:
                        reg_idx1, reg_idx2, dangling_idx = 1, 2, 0
                    elif rt2 - rt0:
                        reg_idx1, reg_idx2, dangling_idx = 2, 0, 1
                    else:
                        continue

                    lines_vert, lines_ind, lines_norm = self._process_triangle(triangle, reg_idx1, reg_idx2,
                                                                               dangling_idx, first_index_in_slice,
                                                                               array_data, slice_vertices,
                                                                               slice_normals)
                    ind_offset = len(processed_vertices) / 3
                    processed_vertices.extend(lines_vert)
                    processed_normals.extend(lines_norm)
                    processed_triangles.extend([ind + ind_offset for ind in lines_ind])
                boundary_vertices.append(processed_vertices)
                boundary_lines.append(processed_triangles)
                boundary_normals.append(processed_normals)
            return [boundary_vertices, boundary_lines, boundary_normals]

    @staticmethod
    def _process_triangle(triangle, reg_idx1, reg_idx2, dangling_idx, indices_offset,
                          region_mapping_array, vertices, normals):
        """
        Process a triangle and generate the required data for a region separation.
        :param triangle: the actual triangle as a 3 element vector
        :param reg_idx1: the first vertex that is in a 'conflicting' region
        :param reg_idx2: the second vertex that is in a 'conflicting' region
        :param dangling_idx: the third vector for which we know nothing yet.
                    Depending on this we might generate a line, or a 3 star centered in the triangle
        :param indices_offset: to take into account the slicing
        :param region_mapping_array: the region mapping raw array for which the regions are computed
        :param vertices: the current vertex slice
        :param normals: the current normals slice
        """

        def _star_triangle(point0, point1, point2, result_array):
            """
            Helper function that for a given triangle generates a 3-way star centered in the triangle center
            """
            center_vertex = [(point0[i] + point1[i] + point2[i]) / 3 for i in range(3)]
            mid_line1 = [(point0[i] + point1[i]) / 2 for i in range(3)]
            mid_line2 = [(point1[i] + point2[i]) / 2 for i in range(3)]
            mid_line3 = [(point2[i] + point0[i]) / 2 for i in range(3)]
            result_array.extend(center_vertex)
            result_array.extend(mid_line1)
            result_array.extend(mid_line2)
            result_array.extend(mid_line3)

        def _slice_triangle(point0, point1, point2, result_array):
            """
            Helper function that for a given triangle generates a line cutting thtough the middle of two edges.
            """
            mid_line1 = [(point0[i] + point1[i]) / 2 for i in range(3)]
            mid_line2 = [(point0[i] + point2[i]) / 2 for i in range(3)]
            result_array.extend(mid_line1)
            result_array.extend(mid_line2)

        # performance opportunity: we are computing some values available in caller

        p0 = vertices[triangle[reg_idx1] - indices_offset]
        p1 = vertices[triangle[reg_idx2] - indices_offset]
        p2 = vertices[triangle[dangling_idx] - indices_offset]
        n0 = normals[triangle[reg_idx1] - indices_offset]
        n1 = normals[triangle[reg_idx2] - indices_offset]
        n2 = normals[triangle[dangling_idx] - indices_offset]
        result_vertices = []
        result_normals = []

        dangling_reg = region_mapping_array[triangle[dangling_idx]]
        reg_1 = region_mapping_array[triangle[reg_idx1]]
        reg_2 = region_mapping_array[triangle[reg_idx2]]

        if dangling_reg != reg_1 and dangling_reg != reg_2:
            # Triangle is actually spanning 3 regions. Create a vertex in the center of the triangle, which connects to
            # the middle of each edge
            _star_triangle(p0, p1, p2, result_vertices)
            _star_triangle(n0, n1, n2, result_normals)
            result_lines = [0, 1, 0, 2, 0, 3]
        elif dangling_reg == reg_1:
            # Triangle spanning only 2 regions, draw a line through the middle of the triangle
            _slice_triangle(p1, p0, p2, result_vertices)
            _slice_triangle(n1, n0, n2, result_normals)
            result_lines = [0, 1]
        else:
            # Triangle spanning only 2 regions, draw a line through the middle of the triangle
            _slice_triangle(p0, p1, p2, result_vertices)
            _slice_triangle(n0, n1, n2, result_normals)
            result_lines = [0, 1]
        return result_vertices, result_lines, result_normals


class SurfaceViewer(ABCSurfaceDisplayer):
    """
    Static SurfaceData visualizer - for visual inspecting imported surfaces in TVB.
    Optionally it can display associated RegionMapping entities.
    """
    _ui_name = "Surface Visualizer"
    _ui_subsection = "surface"

    def get_form_class(self):
        return SurfaceViewerForm

    @staticmethod
    def _compute_surface_params(surface_h5, region_map_gid=None):
        rendering_urls = []
        # we want the URLs in json
        # But these string are going to be verbatim strings in js source code
        # This means that js will interpret escapes like \" so the json parser gets "
        # Double escape is needed \\"
        for url in SurfaceURLGenerator.get_urls_for_rendering(surface_h5, region_map_gid):
            escaped_url = json.dumps(url).replace('\\', '\\\\')
            rendering_urls.append(escaped_url)
        url_vertices, url_normals, url_lines, url_triangles, url_region_map = rendering_urls

        return dict(urlVertices=url_vertices, urlTriangles=url_triangles, urlLines=url_lines,
                    urlNormals=url_normals, urlRegionMap=url_region_map)

    @staticmethod
    def _compute_hemispheric_param(surface_h5):
        bi_hemispheric = surface_h5.bi_hemispheric.load()
        hemisphere_chunk_mask = surface_h5.get_slices_to_hemisphere_mask()
        return dict(biHemispheric=bi_hemispheric, hemisphereChunkMask=json.dumps(hemisphere_chunk_mask))

    def _compute_measure_points_param(self, surface_gid, region_map_gid=None, connectivity_gid=None):
        if region_map_gid is None:
            measure_points_no = 0
            url_measure_points = ''
            url_measure_points_labels = ''
            boundary_url = ''
        else:
            connectivity_index = self.load_entity_by_gid(connectivity_gid)
            measure_points_no = connectivity_index.number_of_regions

            url_measure_points = SurfaceURLGenerator.build_h5_url(connectivity_gid, 'get_centres')
            url_measure_points_labels = SurfaceURLGenerator.build_h5_url(connectivity_gid, 'get_region_labels')

            boundary_url = SurfaceURLGenerator.get_url_for_region_boundaries(surface_gid, region_map_gid,
                                                                             self.stored_adapter.id)

        return dict(noOfMeasurePoints=measure_points_no, urlMeasurePoints=url_measure_points,
                    urlMeasurePointsLabels=url_measure_points_labels, boundaryURL=boundary_url)

    @staticmethod
    def _compute_measure_param(connectivity_measure, measure_points_no):
        # type: (ConnectivityMeasureIndex, int) -> dict
        if connectivity_measure is None:
            # If there is no measure to show then we what to show the region mapping
            # The client will generate a range signal for this use case.
            min_measure = 0
            max_measure = measure_points_no
            client_measure_url = ''
        else:
            connectivity_measure_shape = json.loads(connectivity_measure.shape)
            if len(connectivity_measure_shape) != 1:
                raise ValueError("connectivity measure must be 1 dimensional")
            if connectivity_measure_shape[0] != measure_points_no:
                raise ValueError("connectivity measure has %d values but the connectivity has %d "
                                 "regions" % (connectivity_measure_shape[0], measure_points_no))
            min_measure = connectivity_measure.array_data_min
            max_measure = connectivity_measure.array_data_max
            # We assume here that the index 0 in the measure corresponds to
            # the region 0 of the region map.
            client_measure_url = SurfaceURLGenerator.build_h5_url(connectivity_measure.gid,
                                                                  "get_array_data")

        return dict(minMeasure=min_measure, maxMeasure=max_measure, clientMeasureUrl=client_measure_url)

    def launch(self, view_model):
        # type: (SurfaceViewerModel) -> dict
        surface_index = self.load_entity_by_gid(view_model.surface)
        connectivity_measure_index = None
        region_map_index = None

        if view_model.connectivity_measure:
            connectivity_measure_index = self.load_entity_by_gid(view_model.connectivity_measure)
        if view_model.region_map:
            region_map_index = self.load_entity_by_gid(view_model.region_map)

        surface_h5 = h5.h5_file_for_index(surface_index)
        region_map_gid = region_map_index.gid if region_map_index is not None else None
        connectivity_gid = region_map_index.fk_connectivity_gid if region_map_index is not None else None
        assert isinstance(surface_h5, SurfaceH5)

        params = dict(title=surface_index.display_name, extended_view=False,
                      isOneToOneMapping=False, hasRegionMap=region_map_index is not None)
        params.update(self._compute_surface_params(surface_h5, region_map_gid))
        params.update(self._compute_hemispheric_param(surface_h5))
        params.update(self._compute_measure_points_param(surface_index.gid, region_map_gid, connectivity_gid))
        params.update(self._compute_measure_param(connectivity_measure_index, params['noOfMeasurePoints']))

        surface_h5.close()

        params['shellObject'] = None

        shell_surface_index = None
        if view_model.shell_surface:
            shell_surface_index = self.load_entity_by_gid(view_model.shell_surface)

        shell_surface = ensure_shell_surface(self.current_project_id, shell_surface_index)
        params['shellObject'] = self.prepare_shell_surface_params(shell_surface, SurfaceURLGenerator)
        return self.build_display_result("surface/surface_view", params,
                                         pages={"controlPage": "surface/surface_viewer_controls"})

    def get_required_memory_size(self, view_model):
        return -1


class RegionMappingViewerForm(BaseSurfaceViewerForm):

    def __init__(self):
        super(RegionMappingViewerForm, self).__init__()
        self.region_map.required = True

    @staticmethod
    def get_view_model():
        return BaseSurfaceViewerModel

    @staticmethod
    def get_required_datatype():
        return RegionMappingIndex

    @staticmethod
    def get_input_name():
        return 'region_map'


class RegionMappingViewer(SurfaceViewer):
    """
    This is a viewer for RegionMapping DataTypes.
    It reuses almost everything from SurfaceViewer, but it make required another input param.
    """
    _ui_name = "Region Mapping Visualizer"
    _ui_subsection = "surface"

    def get_form_class(self):
        return RegionMappingViewerForm

    def launch(self, view_model):
        # type: (BaseSurfaceViewerModel) -> dict
        region_map_index = self.load_entity_by_gid(view_model.region_map)
        surface_gid = region_map_index.fk_surface_gid

        surface_viewer_model = SurfaceViewerModel(surface=uuid.UUID(surface_gid),
                                                  region_map=view_model.region_map,
                                                  connectivity_measure=view_model.connectivity_measure,
                                                  shell_surface=view_model.shell_surface)
        surface_viewer_model.title = RegionMappingViewer._ui_name

        return SurfaceViewer.launch(self, surface_viewer_model)


class ConnectivityMeasureOnSurfaceViewerForm(BaseSurfaceViewerForm):

    def __init__(self):
        super(ConnectivityMeasureOnSurfaceViewerForm, self).__init__()
        self.connectivity_measure.required = True

    @staticmethod
    def get_view_model():
        return BaseSurfaceViewerModel

    @staticmethod
    def get_required_datatype():
        return ConnectivityMeasureIndex

    @staticmethod
    def get_input_name():
        return 'connectivity_measure'

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.ndim', FilterChain.datatype + '.has_surface_mapping'],
                           operations=["==", "=="], values=[1, True])


class ConnectivityMeasureOnSurfaceViewer(SurfaceViewer):
    """
    This displays a connectivity measure on a surface via a RegionMapping
    It reuses almost everything from SurfaceViewer, but it make required another input param.
    """
    _ui_name = "Connectivity Measure Surface Visualizer"
    _ui_subsection = "surface"

    def get_form_class(self):
        return ConnectivityMeasureOnSurfaceViewerForm

    def _load_proper_region_mapping(self, view_model):
        return

    def launch(self, view_model):
        # type: (BaseSurfaceViewerModel) -> dict

        connectivity_measure_index = self.load_entity_by_gid(view_model.connectivity_measure)
        cm_connectivity_gid = connectivity_measure_index.fk_connectivity_gid
        cm_connectivity_index = dao.get_datatype_by_gid(cm_connectivity_gid)

        region_map_index = None
        rm_connectivity_index = None
        if view_model.region_map:
            region_map_index = self.load_entity_by_gid(view_model.region_map)
            rm_connectivity_gid = region_map_index.fk_connectivity_gid
            rm_connectivity_index = dao.get_datatype_by_gid(rm_connectivity_gid)

        if not region_map_index or rm_connectivity_index.number_of_regions != cm_connectivity_index.number_of_regions:
            region_maps = dao.get_generic_entity(RegionMappingIndex, cm_connectivity_gid, 'fk_connectivity_gid')
            if region_maps:
                region_map_index = region_maps[0]

        if region_map_index is None:
            raise LaunchException("Can not launch this viewer without a compatible RegionMapping entity in the current project!")
        surface_gid = region_map_index.fk_surface_gid
        surface_viewer_model = SurfaceViewerModel(surface=surface_gid,
                                                  region_map=region_map_index.gid,
                                                  connectivity_measure=view_model.connectivity_measure,
                                                  shell_surface=view_model.shell_surface)
        surface_viewer_model.title = self._ui_name
        return SurfaceViewer.launch(self, surface_viewer_model)
