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
from tvb.adapters.datatypes.db.annotation import ConnectivityAnnotationsIndex
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.datatypes.db.patterns import StimuliRegionIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex, RegionVolumeMappingIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.services.exceptions import RemoveDataTypeException


class ConnectivityRemover(ABCRemover):
    """
    Connectivity specific validations at remove time.
    """

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Connectivity is to be removed.
        """
        key = 'fk_connectivity_gid'
        if not skip_validation:
            associated_mes = dao.get_generic_entity(ConnectivityMeasureIndex, self.handled_datatype.gid, key)
            associated_ann = dao.get_generic_entity(ConnectivityAnnotationsIndex, self.handled_datatype.gid, key)
            associated_stim = dao.get_generic_entity(StimuliRegionIndex, self.handled_datatype.gid, key)
            associated_rm = dao.get_generic_entity(RegionMappingIndex, self.handled_datatype.gid, key)
            associated_rvm = dao.get_generic_entity(RegionVolumeMappingIndex, self.handled_datatype.gid, key)
            associated_ts = dao.get_generic_entity(TimeSeriesRegionIndex, self.handled_datatype.gid, key)
            msg = "Connectivity cannot be removed as it is used by at least one "

            if len(associated_ts) > 0:
                raise RemoveDataTypeException(msg + " TimeSeriesRegion.")
            if len(associated_rm) > 0:
                raise RemoveDataTypeException(msg + " RegionMapping.")
            if len(associated_stim) > 0:
                raise RemoveDataTypeException(msg + " StimuliRegion.")
            if len(associated_mes) > 0:
                raise RemoveDataTypeException(msg + " ConnectivityMeasure.")
            if len(associated_rvm) > 0:
                raise RemoveDataTypeException(msg + " RegionVolumeMapping.")
            if len(associated_ann) > 0:
                raise RemoveDataTypeException(msg + " ConnectivityAnnotations.")

        ABCRemover.remove_datatype(self, skip_validation)
