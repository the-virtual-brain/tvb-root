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
.. Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import uuid
import numpy
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.simulator.equation_forms import get_ui_name_to_equation_dict, get_form_for_equation
from tvb.core.adapters.abcadapter import ABCSynchronous, ABCAdapterForm
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import DataTypeSelectField, SimpleSelectField
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.patterns import StimuliSurface, StimuliRegion
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex, StimuliSurfaceIndex
from tvb.datatypes.surfaces import CorticalSurface


class SurfaceStimulusCreatorForm(ABCAdapterForm):

    def __init__(self, spatial_equation_choices, temporal_equation_choices, project_id):
        super(SurfaceStimulusCreatorForm, self).__init__()
        self.project_id = project_id

        # TODO: filter CorticalSurafces
        self.surface = DataTypeSelectField(SurfaceIndex, self, name='surface', required=True, label='Surface',
                                           conditions=self.get_filters())
        self.spatial = SimpleSelectField(spatial_equation_choices, self, name='spatial', required=True,
                                         label='Spatial equation')
        self.temporal = SimpleSelectField(temporal_equation_choices, self, name='temporal', required=True,
                                          label='Temporal equation')

    @staticmethod
    def get_required_datatype():
        return SurfaceIndex

    @staticmethod
    def get_input_name():
        return '_surface'

    @staticmethod
    def get_filters():
        return None


class SurfaceStimulusCreator(ABCSynchronous):
    """
    The purpose of this adapter is to create a StimuliSurface.
    """

    def get_form_class(self):
        return SurfaceStimulusCreatorForm

    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [StimuliSurfaceIndex]

    def launch(self, **kwargs):
        """
        Used for creating a `StimuliSurface` instance
        """
        triangles_indices = kwargs['focal_points_triangles']
        focal_points = []
        fp_triangle_indices = []

        stimuli_surface = StimuliSurface()

        surface_gid = kwargs['surface']
        stimuli_surface.surface = CorticalSurface()
        stimuli_surface.surface.gid = uuid.UUID(surface_gid)

        surface_index = dao.get_datatype_by_gid(surface_gid)
        surface_h5 = h5.h5_file_for_index(surface_index)
        triangles = surface_h5.triangles.load()
        surface_h5.close()

        for triangle_index in triangles_indices:
            focal_points.append(int(triangles[triangle_index][0]))
            fp_triangle_indices.append(int(triangle_index))

        stimuli_surface.focal_points_triangles = numpy.array(fp_triangle_indices)
        stimuli_surface.focal_points_surface = numpy.array(focal_points)

        stimuli_surface.spatial = self.get_spatial_equation(kwargs)
        stimuli_surface.temporal = self.get_temporal_equation(kwargs)

        stimuli_surface_index = StimuliSurfaceIndex()
        stimuli_surface_index.fill_from_has_traits(stimuli_surface)

        h5.store_complete(stimuli_surface, self.storage_path)
        return stimuli_surface_index

    def get_spatial_equation(self, kwargs):
        return self._prepare_equation('spatial', kwargs)

    def get_temporal_equation(self, kwargs):
        return self._prepare_equation('temporal', kwargs)

    def _prepare_equation(self, equation_type, kwargs):
        """
        From a dictionary of arguments build the equation.
        """
        equation_name = kwargs[equation_type]
        equation = get_ui_name_to_equation_dict().get(equation_name)()
        equation_form = get_form_for_equation(type(equation))(prefix=equation_type)
        equation_form.fill_from_post(kwargs)
        equation_form.fill_trait(equation)
        return equation

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return -1

    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        return 0


class RegionStimulusCreatorForm(ABCAdapterForm):

    def __init__(self, equation_choices, project_id):
        super(RegionStimulusCreatorForm, self).__init__()
        self.project_id = project_id

        self.connectivity = DataTypeSelectField(ConnectivityIndex, self, name='connectivity', label='Connectivity',
                                                required=True)
        self.temporal = SimpleSelectField(equation_choices, self, name='equation', label='Temporal equation',
                                          required=True)
        self.temporal.template = 'form_fields/select_field.html'

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return '_connectivity'

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex


class RegionStimulusCreator(ABCSynchronous):
    """
    The purpose of this adapter is to create a StimuliRegion.
    """

    def get_form_class(self):
        return RegionStimulusCreatorForm

    def get_output(self):
        """
        Describes the outputs of the launch method.
        """
        return [StimuliRegionIndex]

    def launch(self, **kwargs):
        """
        Used for creating a `StimuliRegion` instance
        """
        stimuli_region = StimuliRegion()
        stimuli_region.connectivity = Connectivity()
        stimuli_region.connectivity.gid = uuid.UUID(kwargs['connectivity'])
        stimuli_region.weight = numpy.array(kwargs['weight'])
        stimuli_region.temporal = get_ui_name_to_equation_dict().get(kwargs['temporal'])()
        # TODO: keep prefix in one place
        temporal_equation_form = get_form_for_equation(type(stimuli_region.temporal))(prefix='temporal')
        temporal_equation_form.fill_from_post(kwargs)
        temporal_equation_form.fill_trait(stimuli_region.temporal)

        stimuli_region_idx = StimuliRegionIndex()
        stimuli_region_idx.fill_from_has_traits(stimuli_region)

        h5.store_complete(stimuli_region, self.storage_path)
        return stimuli_region_idx

    def get_required_disk_size(self, **kwargs):
        """
        Returns the required disk size to be able to run the adapter. (in kB)
        """
        return 0

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return -1
