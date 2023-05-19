# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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

"""
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

import numpy as np
from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.contrib.scripts.utils.data_structures_utils import labels_to_inds, monopolar_to_bipolar, \
    split_string_text_numbers
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import Attr, NArray
from tvb.datatypes.sensors import Sensors as TVBSensors, SensorTypes
from tvb.datatypes.sensors import SensorsEEG as TVBSensorsEEG
from tvb.datatypes.sensors import SensorsInternal as TVBSensorsInternal
from tvb.datatypes.sensors import SensorsMEG as TVBSensorsMEG
from tvb.simulator.plot.utils import ensure_list


class Sensors(TVBSensors, BaseModel):
    name = Attr(
        field_type=str,
        label="Sensors' name",
        default='', required=False,
        doc="Sensors' name")

    def configure(self, remove_leading_zeros_from_labels=False):
        if len(self.labels) > 0:
            if remove_leading_zeros_from_labels:
                self.remove_leading_zeros_from_labels()
        super(Sensors, self).configure()

    def sensor_label_to_index(self, labels):
        return self.labels2inds(self.labels, labels)

    def get_sensors_inds_by_sensors_labels(self, lbls):
        # Make sure that the labels are not bipolar:
        lbls = [label.split("-")[0] for label in ensure_list(lbls)]
        return labels_to_inds(self._tvb.labels, lbls)

    def remove_leading_zeros_from_labels(self):
        labels = []
        for label in self._tvb.labels:
            splitLabel = split_string_text_numbers(label)[0]
            n_lbls = len(splitLabel)
            if n_lbls > 0:
                elec_name = splitLabel[0]
                if n_lbls > 1:
                    sensor_ind = splitLabel[1]
                    labels.append(elec_name + sensor_ind.lstrip("0"))
                else:
                    labels.append(elec_name)
            else:
                labels.append(label)
        self._tvb.labels = np.array(labels)

    def get_bipolar_sensors(self, sensors_inds=None):
        if sensors_inds is None:
            sensors_inds = range(self.number_of_sensors)
        return monopolar_to_bipolar(self.labels, sensors_inds)

    def to_tvb_instance(self, datatype=TVBSensors, **kwargs):
        return super(Sensors, self).to_tvb_instance(datatype, **kwargs)


class SensorsEEG(Sensors, TVBSensorsEEG):

    def to_tvb_instance(self, **kwargs):
        return super(SensorsEEG, self).to_tvb_instance(TVBSensorsEEG, **kwargs)


class SensorsMEG(Sensors, TVBSensorsMEG):

    def to_tvb_instance(self, **kwargs):
        return super(SensorsMEG, self).to_tvb_instance(TVBSensorsMEG, **kwargs)


class SensorsInternal(Sensors, TVBSensorsInternal):
    logger = get_logger(__name__)
    elec_labels = NArray(
        dtype=np.str_,
        label="Electrodes' labels", default=None, required=False,
        doc="""Labels of electrodes.""")

    elec_inds = NArray(
        dtype=np.int64,
        label="Electrodes' indices", default=None, required=False,
        doc="""Indices of electrodes.""")

    @property
    def number_of_electrodes(self):
        if self.elec_labels is None:
            return 0
        else:
            return len(self.elec_labels)

    @property
    def channel_labels(self):
        return self.elec_labels

    @property
    def channel_inds(self):
        return self.elec_inds

    def configure(self):
        super(SensorsInternal, self).configure()
        if self.number_of_sensors > 0:
            self.elec_labels, self.elec_inds = self.group_sensors_to_electrodes()
        else:
            self.elec_labels = None
            self.elec_inds = None

    def get_elecs_inds_by_elecs_labels(self, lbls):
        if self.elec_labels is not None:
            return labels_to_inds(self.elec_labels, lbls)
        else:
            return None

    def get_sensors_inds_by_elec_labels(self, lbls):
        elec_inds = self.get_elecs_inds_by_elecs_labels(lbls)
        if elec_inds is not None:
            sensors_inds = []
            for ind in elec_inds:
                sensors_inds += self.elec_inds[ind]
            return np.unique(sensors_inds)

    def group_sensors_to_electrodes(self, labels=None):
        if self.sensors_type == SensorTypes.TYPE_INTERNAL.value:
            if labels is None:
                labels = self.labels
            sensor_names = np.array(split_string_text_numbers(labels))
            elec_labels = np.unique(sensor_names[:, 0])
            elec_inds = []
            for chlbl in elec_labels:
                elec_inds.append(np.where(sensor_names[:, 0] == chlbl)[0])
            return np.array(elec_labels), np.array(elec_inds)
        else:
            self.logger.warning("No multisensor electrodes for %s sensors!" % self.sensors_type)
            return self.elec_labels, self.elec_inds

    def get_bipolar_elecs(self, elecs):
        try:
            bipolar_sensors_lbls = []
            bipolar_sensors_inds = []
            if self.elecs_inds is None:
                return None
            for elec_ind in elecs:
                curr_inds, curr_lbls = self.get_bipolar_sensors(sensors_inds=self.elec_inds[elec_ind])
                bipolar_sensors_inds.append(curr_inds)
                bipolar_sensors_lbls.append(curr_lbls)
        except:
            elecs_inds = self.get_elecs_inds_by_elecs_labels(elecs)
            if elecs_inds is None:
                return None
            bipolar_sensors_inds, bipolar_sensors_lbls = self.get_bipolar_elecs(elecs_inds)
        return bipolar_sensors_inds, bipolar_sensors_lbls

    def to_tvb_instance(self, **kwargs):
        return super(SensorsInternal, self).to_tvb_instance(TVBSensorsInternal, **kwargs)


class SensorsSEEG(SensorsInternal):
    sensors_type = Attr(str, default=SensorTypes.TYPE_INTERNAL.value, required=False)

    def to_tvb_instance(self, **kwargs):
        return super(SensorsSEEG, self).to_tvb_instance(TVBSensorsInternal, **kwargs)
