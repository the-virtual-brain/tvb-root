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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json

from tvb.adapters.datatypes.h5.surface_h5 import SurfaceH5
from tvb.adapters.visualizers.surface_view import ABCSurfaceDisplayer, SurfaceURLGenerator
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.db.annotation import *
from tvb.core.neocom import h5
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import URLGenerator
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.storage import dao
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionMapping


class ConnectivityAnnotationsViewModel(ViewModel):
    connectivity_index = DataTypeGidAttr(
        linked_datatype=Connectivity,
        required=False,
        label='Large Scale Connectivity Matrix'
    )

    annotations_index = DataTypeGidAttr(
        linked_datatype=ConnectivityAnnotations,
        label='Ontology Annotations'
    )

    region_mapping_index = DataTypeGidAttr(
        linked_datatype=RegionMapping,
        required=False,
        label='Region mapping',
        doc='A region map to identify us the Cortical Surface to display,  as well as how the mapping '
            'from Connectivity to Cortex is done '
    )


class ConnectivityAnnotationsViewForm(ABCAdapterForm):

    def __init__(self):
        super(ConnectivityAnnotationsViewForm, self).__init__()
        # Used for filtering
        self.connectivity_index = TraitDataTypeSelectField(ConnectivityAnnotationsViewModel.connectivity_index,
                                                           'connectivity_index')
        self.annotations_index = TraitDataTypeSelectField(ConnectivityAnnotationsViewModel.annotations_index,
                                                          'annotations_index', conditions=self.get_filters())
        self.region_mapping_index = TraitDataTypeSelectField(ConnectivityAnnotationsViewModel.region_mapping_index,
                                                             'region_mapping_index')

    @staticmethod
    def get_view_model():
        return ConnectivityAnnotationsViewModel

    @staticmethod
    def get_required_datatype():
        return ConnectivityAnnotationsIndex

    @staticmethod
    def get_input_name():
        return 'annotations_index'

    @staticmethod
    def get_filters():
        # filters_ui = [UIFilter(linked_elem_name="annotations",
        #                        linked_elem_field=FilterChain.datatype + "._connectivity"),
        #               UIFilter(linked_elem_name="region_map",
        #                        linked_elem_field=FilterChain.datatype + "._connectivity"),
        #               UIFilter(linked_elem_name="connectivity_measure",
        #                        linked_elem_field=FilterChain.datatype + "._connectivity")]
        #
        # json_ui_filter = json.dumps([ui_filter.to_dict() for ui_filter in filters_ui])
        return None


class ConnectivityAnnotationsView(ABCSurfaceDisplayer):
    """
    Given a Connectivity Matrix and a Surface data the viewer will display the matrix 'inside' the surface data.
    The surface is only displayed as a shadow.
    """
    _ui_name = "Annotations Visualizer"
    _ui_subsection = "annotations"

    def get_form_class(self):
        return ConnectivityAnnotationsViewForm

    def get_required_memory_size(self, view_model):
        # type: (ConnectivityAnnotationsViewModel) -> int
        return -1

    def launch(self, view_model):
        # type: (ConnectivityAnnotationsViewModel) -> dict

        annotations_index = self.load_entity_by_gid(view_model.annotations_index)

        if view_model.connectivity_index is None:
            connectivity_index = self.load_entity_by_gid(annotations_index.fk_connectivity_gid)
        else:
            connectivity_index = self.load_entity_by_gid(view_model.connectivity_index)

        if view_model.region_mapping_index is None:
            region_map = dao.get_generic_entity(RegionMappingIndex, connectivity_index.gid,
                                                'fk_connectivity_gid')
            if len(region_map) < 1:
                raise LaunchException(
                    "Can not launch this viewer unless we have at least a RegionMapping for the current Connectivity!")
            region_mapping_index = region_map[0]
        else:
            region_mapping_index = self.load_entity_by_gid(view_model.region_mapping_index)

        boundary_url = SurfaceURLGenerator.get_url_for_region_boundaries(region_mapping_index.fk_surface_gid,
                                                                         region_mapping_index.gid,
                                                                         self.stored_adapter.id)

        surface_h5 = h5.h5_file_for_gid(region_mapping_index.fk_surface_gid)
        assert isinstance(surface_h5, SurfaceH5)
        url_vertices_pick, url_normals_pick, url_triangles_pick = SurfaceURLGenerator.get_urls_for_pick_rendering(
            surface_h5)
        url_vertices, url_normals, _, url_triangles, url_region_map = SurfaceURLGenerator.get_urls_for_rendering(
            surface_h5, region_mapping_index.gid)

        params = dict(title="Connectivity Annotations Visualizer",
                      annotationsTreeUrl=URLGenerator.build_url(self.stored_adapter.id, 'tree_json',
                                                                view_model.annotations_index),
                      urlTriangleToRegion=URLGenerator.build_url(self.stored_adapter.id, "get_triangles_mapping",
                                                                 region_mapping_index.gid),
                      urlActivationPatterns=URLGenerator.paths2url(view_model.annotations_index,
                                                                   "get_activation_patterns"),

                      minValue=0,
                      maxValue=connectivity_index.number_of_regions - 1,
                      urlColors=json.dumps(url_region_map),

                      urlVerticesPick=json.dumps(url_vertices_pick),
                      urlTrianglesPick=json.dumps(url_triangles_pick),
                      urlNormalsPick=json.dumps(url_normals_pick),
                      brainCenter=json.dumps(surface_h5.center()),

                      urlVertices=json.dumps(url_vertices),
                      urlTriangles=json.dumps(url_triangles),
                      urlNormals=json.dumps(url_normals),
                      urlRegionBoundaries=boundary_url)

        return self.build_display_result("annotations/annotations_view", params,
                                         pages={"controlPage": "annotations/annotations_controls"})

    def _get_activation_pattern_labels(self, annotations):
        """
        :return: map {brco_id: list of TVB regions LABELS in which the same BRCO term is being subclass}
        """
        map_with_ids = annotations.get_activation_patterns()
        map_with_labels = dict()

        for ann_id, activated_ids in list(map_with_ids.items()):
            map_with_labels[ann_id] = []
            for string_idx in activated_ids:
                int_idx = int(string_idx)
                conn_label = annotations.connectivity.region_labels[int_idx]
                map_with_labels[ann_id].append(conn_label)

        return map_with_labels

    def tree_json(self, annotations_gid):
        """
        :return: JSON to be rendered in a Tree of entities
        """
        annotations = self.load_with_references(annotations_gid)
        annotations_map = dict()
        regions_map = dict()
        for i in range(annotations.connectivity.number_of_regions):
            regions_map[i] = []

        for ann in annotations.region_annotations:
            ann_obj = AnnotationTerm(ann[0], ann[1], ann[2], ann[3], ann[4], ann[5],
                                     ann[6], ann[7], ann[8], ann[9], ann[10])
            annotations_map[ann_obj.id] = ann_obj
            if ann_obj.parent_id < 0:
                # Root directly under a TVB region node
                regions_map[ann_obj.parent_left].append(ann_obj)
                regions_map[ann_obj.parent_right].append(ann_obj)
            elif ann_obj.parent_id in annotations_map:
                annotations_map[ann_obj.parent_id].add_child(ann_obj)
            else:
                self.logger.warning("Order of processing invalid parent %s child %s" % (ann_obj.parent_id, ann_obj.id))

        left_nodes, right_nodes = [], []
        activation_patterns = self._get_activation_pattern_labels(annotations)
        for region_idx, annotations_list in list(regions_map.items()):
            if_right_hemisphere = annotations.connectivity.is_right_hemisphere(region_idx)
            childred_json = []
            for ann_term in annotations_list:
                childred_json.append(ann_term.to_json(if_right_hemisphere, activation_patterns))
            # This node is built for every TVB region
            child_json = dict(data=dict(icon=ICON_TVB,
                                        title=annotations.connectivity.region_labels[region_idx]),
                              attr=dict(id=NODE_ID_TVB_ROOT + str(region_idx),
                                        title=str(region_idx) + " - " + annotations.connectivity.region_labels[
                                            region_idx]),
                              state="close", children=childred_json)
            if if_right_hemisphere:
                right_nodes.append(child_json)
            else:
                left_nodes.append(child_json)

        # Group everything under a single root
        left_root = dict(data=dict(title="Left Hemisphere", icon=ICON_FOLDER),
                         state="open", children=left_nodes)
        right_root = dict(data=dict(title="Right Hemisphere", icon=ICON_FOLDER),
                          state="open", children=right_nodes)
        root_root = dict(data=dict(title="Connectivity Annotations", icon=ICON_FOLDER),
                         state="open", children=[left_root, right_root])
        return root_root

    def get_triangles_mapping(self, region_mapping_gid):
        """
        :return Numpy array of length triangles and for each the region corresponding to one of its vertices.
        """
        region_mapping = self.load_with_references(region_mapping_gid)
        triangles_no = region_mapping.surface.number_of_triangles
        result = []
        for i in range(triangles_no):
            result.append(region_mapping.array_data[region_mapping.surface.triangles[i][0]])
        return result
