# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
A tracts visualizer
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from tvb.adapters.visualizers.time_series import ABCSpaceDisplayer
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.adapters.datatypes.db.tracts import TractsIndex
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.surfaces import CorticalSurface, Surface
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

    def __init__(self, project_id=None):
        super(TractViewerForm, self).__init__(project_id)
        self.tracts = TraitDataTypeSelectField(TractViewerModel.tracts, self.project_id, name='tracts')
        self.shell_surface = TraitDataTypeSelectField(TractViewerModel.shell_surface, self.project_id,
                                                      name='shell_surface')

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

    # TODO: migrate to neotraits
    def launch(self, view_model):
        # type: (TractViewerModel) -> dict
        from tvb.adapters.visualizers.surface_view import prepare_shell_surface_urls

        url_track_starts, url_track_vertices = view_model.tracts.get_urls_for_rendering()

        if view_model.tracts.region_volume_map is None:
            raise Exception('only tracts with an associated region volume map are supported at this moment')

        connectivity = view_model.tracts.region_volume_map.connectivity

        params = dict(title="Tract Visualizer",
                      shelfObject=prepare_shell_surface_urls(self.current_project_id, view_model.shell_surface,
                                                             preferred_type=CorticalSurface),

                      urlTrackStarts=url_track_starts,
                      urlTrackVertices=url_track_vertices)

        params.update(self.build_params_for_selectable_connectivity(connectivity))

        return self.build_display_result("tract/tract_view", params,
                                         pages={"controlPage": "tract/tract_viewer_controls"})

    def get_required_memory_size(self, view_model):
        # type: (TractViewerModel) -> int
        return -1
