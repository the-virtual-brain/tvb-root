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
from tvb.datatypes.structural import StructuralMRI
from tvb.adapters.datatypes.db.volume import VolumeIndex
from tvb.core.entities.model.model_datatype import DataTypeMatrix


class StructuralMRIIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    weighting = Column(String, nullable=False)

    fk_volume_gid = Column(String(32), ForeignKey(VolumeIndex.gid), nullable=not StructuralMRI.volume.required)
    volume = relationship(VolumeIndex, foreign_keys=fk_volume_gid, primaryjoin=VolumeIndex.gid == fk_volume_gid)

    def fill_from_has_traits(self, datatype):
        # type: (StructuralMRI)  -> None
        super(StructuralMRIIndex, self).fill_from_has_traits(datatype)
        self.weighting = datatype.weighting
        self.fk_volume_gid = datatype.volume.gid.hex
