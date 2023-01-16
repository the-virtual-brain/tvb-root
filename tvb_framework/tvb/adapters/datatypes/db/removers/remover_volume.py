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
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.datatypes.db.region_mapping import RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.structural import StructuralMRIIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesVolumeIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.services.exceptions import RemoveDataTypeException


class VolumeRemover(ABCRemover):
    """
    Surface specific validations at remove time.
    """

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Surface is to be removed.
        """
        if not skip_validation:
            key = 'fk_volume_gid'
            associated_ts = dao.get_generic_entity(TimeSeriesVolumeIndex, self.handled_datatype.gid, key)
            associated_rvm = dao.get_generic_entity(RegionVolumeMappingIndex, self.handled_datatype.gid, key)
            associated_s_mri = dao.get_generic_entity(StructuralMRIIndex, self.handled_datatype.gid, key)

            error_msg = "Volume cannot be removed because is still used by a "

            if len(associated_ts) > 0:
                raise RemoveDataTypeException(error_msg + " TimeSeriesVolume.")
            if len(associated_rvm) > 0:
                raise RemoveDataTypeException(error_msg + " RegionVolumeMapping.")
            if len(associated_s_mri) > 0:
                raise RemoveDataTypeException(error_msg + " StructuralMRI.")

        ABCRemover.remove_datatype(self, skip_validation)
