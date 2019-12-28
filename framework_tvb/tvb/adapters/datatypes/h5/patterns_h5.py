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
from tvb.basic.neotraits.api import NArray
from tvb.datatypes.patterns import StimuliRegion, StimuliSurface
from tvb.core.neotraits.h5 import H5File, Reference, DataSet, EquationScalar


class StimuliRegionH5(H5File):

    def __init__(self, path):
        super(StimuliRegionH5, self).__init__(path)
        self.spatial = EquationScalar(StimuliRegion.spatial, self)
        self.temporal = EquationScalar(StimuliRegion.temporal, self)
        self.connectivity = Reference(StimuliRegion.connectivity, self)
        self.weight = DataSet(StimuliRegion.weight, self)

    def store(self, datatype, scalars_only=False):
        super(StimuliRegionH5, self).store(datatype, scalars_only)
        self.connectivity.store(datatype.connectivity)


class StimuliSurfaceH5(H5File):

    def __init__(self, path):
        super(StimuliSurfaceH5, self).__init__(path)
        self.spatial = EquationScalar(StimuliSurface.spatial, self)
        self.temporal = EquationScalar(StimuliSurface.temporal, self)
        self.surface = Reference(StimuliSurface.surface, self)
        self.focal_points_surface = DataSet(NArray(dtype=int), self, name='focal_points_surface')
        self.focal_points_triangles = DataSet(StimuliSurface.focal_points_triangles, self)

    def store(self, datatype, scalars_only=False):
        super(StimuliSurfaceH5, self).store(datatype, scalars_only)
        self.focal_points_surface.store(datatype.focal_points_surface)

