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
from sqlalchemy import Column, Integer, ForeignKey, String, Float
from sqlalchemy.orm import relationship
from tvb.datatypes.structural import StructuralMRI
from tvb.adapters.datatypes.db.volume import VolumeIndex
from tvb.core.entities.model.model_datatype import DataTypeMatrix
from tvb.core.neotraits.db import from_ndarray


class StructuralMRIIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    array_data_min = Column(Float)
    array_data_max = Column(Float)
    array_data_mean = Column(Float)

    weighting = Column(String, nullable=False)

    volume_gid = Column(String(32), ForeignKey(VolumeIndex.gid), nullable=not StructuralMRI.volume.required)
    volume = relationship(VolumeIndex, foreign_keys=volume_gid, primaryjoin=VolumeIndex.gid == volume_gid)

    def fill_from_has_traits(self, datatype):
        # type: (StructuralMRI)  -> None
        super(StructuralMRIIndex, self).fill_from_has_traits(datatype)
        self.weighting = datatype.weighting
        self.array_data_min, self.array_data_max, self.array_data_mean = from_ndarray(datatype.array_data)
        self.volume_gid = datatype.volume.gid.hex