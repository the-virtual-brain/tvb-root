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
A tracts visualizer
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
from tvb.adapters.visualizers.surface_view import prepare_shell_surface_urls
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.surfaces import CorticalSurface
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.tracts import Tracts


class TractViewer(ABCDisplayer):
    """
    Tract visualizer
    """
    _ui_name = "Tract Visualizer"
    _ui_subsection = "surface"

    def get_input_tree(self):

        return [{'name': 'tracts', 'label': 'White matter tracts',
                 'type': Tracts, 'required': True,
                 'description': ''},
                {'name': 'shell_surface', 'label': 'Shell Surface', 'type': Surface, 'required': False,
                 'description': "Surface to be displayed semi-transparently, for visual purposes only."}]



    def launch(self, tracts, shell_surface=None):

        url_track_starts, url_track_vertices = tracts.get_urls_for_rendering()

        if tracts.region_volume_map is None:
            raise Exception('only tracts with an associated region volume map are supported at this moment')

        connectivity = tracts.region_volume_map.connectivity

        params = dict(title="Tract Visualizer",
                      shelfObject=prepare_shell_surface_urls(self.current_project_id, shell_surface, preferred_type=CorticalSurface),

                      urlTrackStarts=url_track_starts,
                      urlTrackVertices=url_track_vertices)

        params.update(self.build_template_params_for_subselectable_datatype(connectivity))

        return self.build_display_result("tract/tract_view", params,
                                         pages={"controlPage": "tract/tract_viewer_controls"})


    def get_required_memory_size(self):
        return -1