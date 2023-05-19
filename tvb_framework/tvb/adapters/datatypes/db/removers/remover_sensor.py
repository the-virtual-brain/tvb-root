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

from tvb.adapters.datatypes.db.projections import ProjectionMatrixIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesSEEGIndex, TimeSeriesMEGIndex, TimeSeriesEEGIndex
from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.services.exceptions import RemoveDataTypeException


class SensorRemover(ABCRemover):
    """
    Sensor specific validations at remove time.
    """

    def remove_datatype(self, skip_validation=False):
        """
        Called when a Sensor is to be removed.
        """
        if not skip_validation:
            key = 'fk_sensors_gid'
            projection_matrices_sensors = dao.get_generic_entity(ProjectionMatrixIndex, self.handled_datatype.gid, key)
            ts_seeg = dao.get_generic_entity(TimeSeriesSEEGIndex, self.handled_datatype.gid, key)
            ts_meg = dao.get_generic_entity(TimeSeriesMEGIndex, self.handled_datatype.gid, key)
            ts_eeg = dao.get_generic_entity(TimeSeriesEEGIndex, self.handled_datatype.gid, key)

            error_msg = "Cannot be removed as it is used by %d %s entities."

            if len(ts_seeg) > 0:
                raise RemoveDataTypeException(error_msg % (len(ts_seeg),"TimeSeriesSEEG"))
            if len(ts_meg) > 0:
                raise RemoveDataTypeException(error_msg % (len(ts_meg),"TimeSeriesMEG"))
            if len(ts_eeg) > 0:
                raise RemoveDataTypeException(error_msg % (len(ts_eeg),"TimeSeriesEEG"))
            if len(projection_matrices_sensors) > 0:
                raise RemoveDataTypeException(error_msg % (len(projection_matrices_sensors),"ProjectionMatrix"))

        ABCRemover.remove_datatype(self, skip_validation)
