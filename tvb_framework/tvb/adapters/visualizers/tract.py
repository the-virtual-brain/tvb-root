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
A tracts visualizer
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from tvb.adapters.datatypes.db.tracts import TractsIndex
from tvb.adapters.visualizers.surface_view import ensure_shell_surface, SurfaceURLGenerator
from tvb.adapters.visualizers.time_series import ABCSpaceDisplayer
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.adapters.abcdisplayer import URLGenerator
from tvb.core.entities import load
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.surfaces import Surface, SurfaceTypesEnum
from tvb.datatypes.tracts import Tracts


class TractViewerModel(ViewModel):
    tracts = DataTypeGidAttr(
        linked_datatype=Tracts,
        label='White matter tracts'
    )

    shell_surface = DataTypeGidAttr(
        linked_datatype=Surface,
        required=False,
        label='Shell Surface',
        doc='Surface to be displayed semi-transparently, for visual purposes only.'
    )


class TractViewerForm(ABCAdapterForm):

    def __init__(self):
        super(TractViewerForm, self).__init__()
        self.tracts = TraitDataTypeSelectField(TractViewerModel.tracts, name='tracts')
        self.shell_surface = TraitDataTypeSelectField(TractViewerModel.shell_surface, name='shell_surface')

    @staticmethod
    def get_view_model():
        return TractViewerModel

    @staticmethod
    def get_required_datatype():
        return TractsIndex

    @staticmethod
    def get_input_name():
        return 'tracts'

    @staticmethod
    def get_filters():
        return None


class TractViewer(ABCSpaceDisplayer):
    """
    Tract visualizer
    """
    _ui_name = "Tract Visualizer"
    _ui_subsection = "surface"

    def get_form_class(self):
        return TractViewerForm

    def launch(self, view_model):
        # type: (TractViewerModel) -> dict
        tracts_index = load.load_entity_by_gid(view_model.tracts)
        region_volume_mapping_index = load.load_entity_by_gid(tracts_index.fk_region_volume_map_gid)

        shell_surface_index = None
        if view_model.shell_surface:
            shell_surface_index = self.load_entity_by_gid(view_model.shell_surface)

        shell_surface_index = ensure_shell_surface(self.current_project_id, shell_surface_index,
                                                   SurfaceTypesEnum.FACE_SURFACE.value)

        tracts_starts = URLGenerator.build_h5_url(tracts_index.gid, 'get_line_starts')
        tracts_vertices = URLGenerator.build_binary_datatype_attribute_url(tracts_index.gid, 'get_vertices')

        params = dict(title="Tract Visualizer",
                      shellObject=self.prepare_shell_surface_params(shell_surface_index, SurfaceURLGenerator),
                      urlTrackStarts=tracts_starts, urlTrackVertices=tracts_vertices)

        connectivity = self.load_traited_by_gid(region_volume_mapping_index.fk_connectivity_gid)
        params.update(self.build_params_for_selectable_connectivity(connectivity))

        return self.build_display_result("tract/tract_view", params,
                                         pages={"controlPage": "tract/tract_viewer_controls"})

    def get_required_memory_size(self, view_model):
        # type: (TractViewerModel) -> int
        return -1
