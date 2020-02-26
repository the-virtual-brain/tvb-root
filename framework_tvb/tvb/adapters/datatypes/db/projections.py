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
from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.projections import ProjectionMatrix
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.entities.model.model_datatype import DataType


class ProjectionMatrixIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    projection_type = Column(String, nullable=False)

    brain_skull_gid = Column(String(32), ForeignKey(SurfaceIndex.gid),
                             nullable=not ProjectionMatrix.brain_skull.required)
    brain_skull = relationship(SurfaceIndex, foreign_keys=brain_skull_gid,
                               primaryjoin=SurfaceIndex.gid == brain_skull_gid, cascade='none')

    skull_skin_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not ProjectionMatrix.skull_skin.required)
    skull_skin = relationship(SurfaceIndex, foreign_keys=skull_skin_gid, primaryjoin=SurfaceIndex.gid == skull_skin_gid,
                              cascade='none')

    skin_air_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not ProjectionMatrix.skin_air.required)
    skin_air = relationship(SurfaceIndex, foreign_keys=skin_air_gid, primaryjoin=SurfaceIndex.gid == skin_air_gid,
                            cascade='none')

    source_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not ProjectionMatrix.sources.required)
    source = relationship(SurfaceIndex, foreign_keys=source_gid, primaryjoin=SurfaceIndex.gid == source_gid,
                          cascade='none')

    sensors_gid = Column(String(32), ForeignKey(SensorsIndex.gid), nullable=not ProjectionMatrix.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=sensors_gid, primaryjoin=SensorsIndex.gid == sensors_gid,
                           cascade='none')

    def fill_from_has_traits(self, datatype):
        # type: (ProjectionMatrix)  -> None
        super(ProjectionMatrixIndex, self).fill_from_has_traits(datatype)
        self.projection_type = datatype.projection_type
        if datatype.brain_skull is not None:
            self.brain_skull_gid = datatype.brain_skull.gid.hex
        if datatype.skull_skin is not None:
            self.skull_skin_gid = datatype.skull_skin.gid.hex
        if datatype.skin_air is not None:
            self.skin_air_gid = datatype.skin_air.gid.hex
        self.sensors_gid = datatype.sensors.gid.hex
        self.source_gid = datatype.sources.gid.hex

    @property
    def display_name(self):
        """
        Overwrite from superclass and add subtype
        """
        previous = "ProjectionMatrix"
        return previous + " - " + str(self.projection_type)
