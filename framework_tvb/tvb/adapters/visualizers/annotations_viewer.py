# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
from tvb.adapters.visualizers.surface_view import SurfaceViewer, prepare_shell_surface_urls
from tvb.basic.profile import TvbProfile
from tvb.basic.traits.core import KWARG_FILTERS_UI
from tvb.basic.filters.chain import FilterChain, UIFilter
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.storage import dao
from tvb.datatypes.annotations import ConnectivityAnnotations
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.surfaces_data import SurfaceData



class ConnectivityAnnotationsView(SurfaceViewer):
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
                                'how the mapping from Connectivity to Cortex is done '},
                {'name': 'connectivity_measure', 'label': 'Connectivity measure',
                 'type': ConnectivityMeasure, 'required': False, 'description': 'A connectivity measure',
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1])},
                {'name': 'shell_surface', 'label': 'Shell Surface', 'type': SurfaceData, 'required': False,
                 'description': "Face surface to be displayed semi-transparently, for orientation only."}]


    def get_required_memory_size(self, **kwargs):
        return -1


    def launch(self, annotations, connectivity=None, region_map=None, connectivity_measure=None, shell_surface=None):

        if region_map is None:
            region_map = dao.get_generic_entity(RegionMapping, annotations.connectivity.gid, '_connectivity')
            if len(region_map) < 1:
                raise LaunchException(
                    "Can not launch this viewer unless we have at least a RegionMapping for the current Connectivity!")
            region_map = region_map[0]

        params = dict(title="Connectivity Annotations Visualizer",
                      annotationsTreeUrl=self.paths2url(annotations, 'tree_json'),
                      baseUrl=TvbProfile.current.web.BASE_URL,
                      extended_view=False,
                      isOneToOneMapping=False,
                      hasRegionMap=region_map is not None)

        params.update(self._compute_surface_params(region_map.surface, region_map))
        params.update(self._compute_measure_points_param(region_map.surface, region_map))
        params.update(self._compute_measure_param(connectivity_measure, params['noOfMeasurePoints']))

        try:
            params['shelfObject'] = prepare_shell_surface_urls(self.current_project_id, shell_surface)
        except Exception:
            params['shelfObject'] = None

        return self.build_display_result("annotations/annotations_view", params,
                                         pages={"controlPage": "surface/surface_viewer_controls"})



