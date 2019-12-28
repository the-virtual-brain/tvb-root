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
import json
from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.patterns import StimuliRegion, StimuliSurface
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.entities.model.model_datatype import DataType


class StimuliRegionIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    spatial_equation = Column(String, nullable=False)
    spatial_parameters = Column(String)
    temporal_equation = Column(String, nullable=False)
    temporal_parameters = Column(String)

    connectivity_gid = Column(String(32), ForeignKey(ConnectivityIndex.gid),
                              nullable=not StimuliRegion.connectivity.required)
    connectivity = relationship(ConnectivityIndex, foreign_keys=connectivity_gid,
                                primaryjoin=ConnectivityIndex.gid == connectivity_gid)

    def fill_from_has_traits(self, datatype):
        # type: (StimuliRegion)  -> None
        super(StimuliRegionIndex, self).fill_from_has_traits(datatype)
        self.spatial_equation = datatype.spatial.__class__.__name__
        self.spatial_parameters = json.dumps(datatype.spatial.parameters)
        self.temporal_equation = datatype.temporal.__class__.__name__
        self.temporal_parameters = json.dumps(datatype.temporal.parameters)
        self.connectivity_gid = datatype.connectivity.gid.hex


class StimuliSurfaceIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    spatial_equation = Column(String, nullable=False)
    spatial_parameters = Column(String)
    temporal_equation = Column(String, nullable=False)
    temporal_parameters = Column(String)

    surface_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not StimuliSurface.surface.required)
    surface = relationship(SurfaceIndex, foreign_keys=surface_gid, primaryjoin=SurfaceIndex.gid == surface_gid)

    def fill_from_has_traits(self, datatype):
        # type: (StimuliSurface)  -> None
        super(StimuliSurfaceIndex, self).fill_from_has_traits(datatype)
        self.spatial_equation = datatype.spatial.__class__.__name__
        self.spatial_parameters = json.dumps(datatype.spatial.parameters)
        self.temporal_equation = datatype.temporal.__class__.__name__
        self.temporal_parameters = json.dumps(datatype.temporal.parameters)
        self.surface_gid = datatype.surface.gid.hex
