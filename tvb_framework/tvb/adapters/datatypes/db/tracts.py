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
from tvb.datatypes.tracts import Tracts
from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex
from tvb.core.entities.model.model_datatype import DataType


class TractsIndex(DataType):
    id = Column(Integer, ForeignKey(DataType.id), primary_key=True)

    fk_region_volume_map_gid = Column(String(32), ForeignKey(RegionVolumeMappingIndex.gid),
                                      nullable=not Tracts.region_volume_map.required)
    region_volume_map = relationship(RegionVolumeMappingIndex, foreign_keys=fk_region_volume_map_gid,
                                     primaryjoin=RegionVolumeMappingIndex.gid == fk_region_volume_map_gid)

    def fill_from_has_traits(self, datatype):
        # type: (Tracts)  -> None
        super(TractsIndex, self).fill_from_has_traits(datatype)
        if datatype.region_volume_map is not None:
            self.fk_region_volume_map_gid = datatype.region_volume_map.gid.hex
