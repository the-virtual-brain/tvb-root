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
from sqlalchemy import Column, Integer, ForeignKey, String
from sqlalchemy.orm import relationship
from tvb.datatypes.projections import ProjectionMatrix
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.core.entities.model.model_datatype import DataType


class ProjectionMatrixIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    projection_type = Column(String, nullable=False)

    fk_brain_skull_gid = Column(String(32), ForeignKey(SurfaceIndex.gid),
                                nullable=not ProjectionMatrix.brain_skull.required)
    brain_skull = relationship(SurfaceIndex, foreign_keys=fk_brain_skull_gid,
                               primaryjoin=SurfaceIndex.gid == fk_brain_skull_gid, cascade='none')

    fk_skull_skin_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not ProjectionMatrix.skull_skin.required)
    skull_skin = relationship(SurfaceIndex, foreign_keys=fk_skull_skin_gid, primaryjoin=SurfaceIndex.gid == fk_skull_skin_gid,
                              cascade='none')

    fk_skin_air_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not ProjectionMatrix.skin_air.required)
    skin_air = relationship(SurfaceIndex, foreign_keys=fk_skin_air_gid, primaryjoin=SurfaceIndex.gid == fk_skin_air_gid,
                            cascade='none')

    fk_source_gid = Column(String(32), ForeignKey(SurfaceIndex.gid), nullable=not ProjectionMatrix.sources.required)
    source = relationship(SurfaceIndex, foreign_keys=fk_source_gid, primaryjoin=SurfaceIndex.gid == fk_source_gid,
                          cascade='none')

    fk_sensors_gid = Column(String(32), ForeignKey(SensorsIndex.gid), nullable=not ProjectionMatrix.sensors.required)
    sensors = relationship(SensorsIndex, foreign_keys=fk_sensors_gid, primaryjoin=SensorsIndex.gid == fk_sensors_gid,
                           cascade='none')

    def fill_from_has_traits(self, datatype):
        # type: (ProjectionMatrix)  -> None
        super(ProjectionMatrixIndex, self).fill_from_has_traits(datatype)
        self.projection_type = datatype.projection_type
        if datatype.brain_skull is not None:
            self.fk_brain_skull_gid = datatype.brain_skull.gid.hex
        if datatype.skull_skin is not None:
            self.fk_skull_skin_gid = datatype.skull_skin.gid.hex
        if datatype.skin_air is not None:
            self.fk_skin_air_gid = datatype.skin_air.gid.hex
        self.fk_sensors_gid = datatype.sensors.gid.hex
        self.fk_source_gid = datatype.sources.gid.hex

    def get_subtype_attr(self):
        return self.projection_type

    @property
    def display_name(self):
        """
        Overwrite from superclass and add subtype
        """
        previous = "ProjectionMatrix"
        return previous + " - " + str(self.projection_type)
