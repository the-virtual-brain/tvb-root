# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

"""
The TimeSeries datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

from io import BytesIO
from tvb.datatypes import sensors, surfaces, volumes, region_mapping, connectivity
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List, Float, narray_summary_info
from tvb.basic.readers import H5Reader
import numpy
from copy import deepcopy


class TimeSeries(HasTraits):
    """
    Base time-series dataType.
    """
    title = Attr(str)

    data = NArray(
        label="Time-series data",
        doc="""An array of time-series data, with a shape of [tpts, :], where ':' represents 1 or more dimensions""")

    labels_ordering = List(
        default=("Time", "State Variable", "Space", "Mode"),
        label="Dimension Names",
        doc="""List of strings representing names of each data dimension""")

    labels_dimensions = Attr(
        field_type=dict,
        default={},
        label="Specific labels for each dimension for the data stored in this timeseries.",
        doc=""" A dictionary containing mappings of the form {'dimension_name' : [labels for this dimension] }""")

    time = NArray(
        label="Time-series time",
        required=False,
        doc="""An array of time values for the time-series, with a shape of [tpts,].
            This is 'time' as returned by the simulator's monitors.""")

    start_time = Float(label="Start Time:")

    sample_period = Float(label="Sample period", default=1.0)

    # Specify the measure unit for sample period (e.g sec, msec, usec, ...)
    sample_period_unit = Attr(
        field_type=str,
        label="Sample Period Measure Unit",
        default="ms")

    @property
    def nr_dimensions(self):
        return self.data.ndim

    @property
    def sample_rate(self):
        """:returns samples per second [Hz] """
        if self.sample_period_unit in ("s", "sec"):
            return 1.0 / self.sample_period
        elif self.sample_period_unit in ("ms", "msec"):
            return 1000.0 / self.sample_period
        elif self.sample_period_unit in ("us", "usec"):
            return 1000000.0 / self.sample_period
        else:
            raise ValueError(f"{self.sample_period_unit} is not a recognized time unit")

    @property
    def sample_period_ms(self):
        """:returns sample_period is ms """
        if self.sample_period_unit in ("s", "sec"):
            return 1000 * self.sample_period
        elif self.sample_period_unit in ("ms", "msec"):
            return self.sample_period
        elif self.sample_period_unit in ("us", "usec"):
            return self.sample_period / 1000.0
        else:
            raise ValueError(f"{self.sample_period_unit} is not a recognized time unit")

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = {
            "Time-series type": self.__class__.__name__,
            "Time-series name": self.title,
            "Dimensions": self.labels_ordering,
            "Time units": self.sample_period_unit,
            "Sample period": self.sample_period,
            "Start time": self.start_time,
            "Length": self.sample_period * self.data.shape[0]
        }
        summary.update(narray_summary_info(self.data))
        return summary

    def duplicate(self, **kwargs):
        duplicate = super(TimeSeries, self).duplicate()
        for attr, value in kwargs.items():
            setattr(duplicate, attr, value)
        duplicate.configure()
        return duplicate

    def _get_index_of_state_variable(self, sv_label):
        try:
            sv_index = numpy.where(self.variables_labels == sv_label)[0][0]
        except KeyError:
            self.logger.error("There are no state variables defined for this instance. Its shape is: %s",
                              self.data.shape)
            raise
        except IndexError:
            self.logger.error("Cannot access index of state variable label: %s. Existing state variables: %s" % (
                sv_label, self.variables_labels))
            raise
        return sv_index

    def get_state_variable(self, sv_label):
        sv_data = self.data[:, self._get_index_of_state_variable(sv_label), :, :]
        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[1]] = [sv_label]
        if sv_data.ndim == 3:
            sv_data = numpy.expand_dims(sv_data, 1)
        return self.duplicate(data=sv_data, labels_dimensions=subspace_labels_dimensions)

    def _get_indices_for_labels(self, list_of_labels):
        list_of_indices_for_labels = []
        for label in list_of_labels:
            try:
                space_index = numpy.where(self.space_labels == label)[0][0]
            except ValueError:
                self.logger.error("Cannot access index of space label: %s. Existing space labels: %s" %
                                  (label, self.space_labels))
                raise
            list_of_indices_for_labels.append(space_index)
        return list_of_indices_for_labels

    def get_subspace_by_index(self, list_of_index, **kwargs):
        self._check_space_indices(list_of_index)
        subspace_data = self.data[:, :, list_of_index, :]
        subspace_labels_dimensions = deepcopy(self.labels_dimensions)
        subspace_labels_dimensions[self.labels_ordering[2]] = self.space_labels[list_of_index].tolist()
        if subspace_data.ndim == 3:
            subspace_data = numpy.expand_dims(subspace_data, 2)
        return self.duplicate(data=subspace_data, labels_dimensions=subspace_labels_dimensions, **kwargs)

    def get_subspace_by_labels(self, list_of_labels):
        list_of_indices_for_labels = self._get_indices_for_labels(list_of_labels)
        return self.get_subspace_by_index(list_of_indices_for_labels)

    def __getattr__(self, attr_name):
        if self.labels_ordering[1] in self.labels_dimensions.keys():
            if attr_name in self.variables_labels:
                return self.get_state_variable(attr_name)
        if self.labels_ordering[2] in self.labels_dimensions.keys():
            if attr_name in self.space_labels:
                return self.get_subspace_by_labels([attr_name])
        raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def _get_index_for_slice_label(self, slice_label, slice_idx):
        if slice_idx == 1:
            return self._get_indices_for_labels([slice_label])[0]
        if slice_idx == 2:
            return self._get_index_of_state_variable(slice_label)

    @property
    def shape(self):
        return self.data.shape

    @property
    def time_unit(self):
        return self.sample_period_unit

    @property
    def space_labels(self):
         return numpy.array(self.labels_dimensions.get(self.labels_ordering[2], []))

    @property
    def variables_labels(self):
        return numpy.array(self.labels_dimensions.get(self.labels_ordering[1], []))

    def _check_space_indices(self, list_of_index):
        for index in list_of_index:
            if index < 0 or index > self.data.shape[1]:
                self.logger.error("Some of the given indices are out of space range: [0, %s]",
                                  self.data.shape[1])
                raise IndexError

    @classmethod
    def from_bytes_stream(cls, bytes_stream, content_type=".npz"):
        result = TimeSeries()

        if content_type == '.npz':
            ts_data = numpy.load(BytesIO(bytes_stream))
            result.data = ts_data['data']
            result.time = ts_data['time']
            return result

        reader = H5Reader(BytesIO(bytes_stream))
        result.data = reader.read_field("data")
        result.time = reader.read_optional_field("time")
        return result


class SensorsTSBase(TimeSeries):

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(SensorsTSBase, self).summary_info()
        summary.update({"Source Sensors": self.sensors.title})
        return summary


class TimeSeriesEEG(SensorsTSBase):
    """ A time series associated with a set of EEG sensors. """

    sensors = Attr(field_type=sensors.SensorsEEG)
    labels_ordering = List(of=str, default=("Time", "SV", "EEG Sensor", "Mode"))


class TimeSeriesMEG(SensorsTSBase):
    """ A time series associated with a set of MEG sensors. """

    sensors = Attr(field_type=sensors.SensorsMEG)
    labels_ordering = List(of=str, default=("Time", "SV", "MEG Sensor", "Mode"))


class TimeSeriesSEEG(SensorsTSBase):
    """ A time series associated with a set of Internal sensors. """

    sensors = Attr(field_type=sensors.SensorsInternal)
    labels_ordering = List(of=str, default=("Time", "SV", "sEEG Sensor", "Mode"))


class TimeSeriesRegion(TimeSeries):
    """ A time-series associated with the regions of a connectivity. """

    connectivity = Attr(field_type=connectivity.Connectivity)
    region_mapping_volume = Attr(field_type=region_mapping.RegionVolumeMapping, required=False)
    region_mapping = Attr(field_type=region_mapping.RegionMapping, required=False)
    labels_ordering = List(of=str, default=("Time", "State Variable", "Region", "Mode"))

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesRegion, self).summary_info()
        summary.update({
            "Source Connectivity": self.connectivity.title,
            "Region Mapping": self.region_mapping.title if self.region_mapping else "None",
            "Region Mapping Volume": (self.region_mapping_volume.title
                                      if self.region_mapping_volume else "None")
        })
        return summary


class TimeSeriesSurface(TimeSeries):
    """ A time-series associated with a Surface. """

    surface = Attr(field_type=surfaces.CorticalSurface)
    labels_ordering = List(of=str, default=("Time", "State Variable", "Vertex", "Mode"))

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesSurface, self).summary_info()
        summary.update({"Source Surface": self.surface.title})
        return summary


class TimeSeriesVolume(TimeSeries):
    """ A time-series associated with a Volume. """

    volume = Attr(field_type=volumes.Volume)
    labels_ordering = List(of=str, default=("Time", "X", "Y", "Z"))

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesVolume, self).summary_info()
        summary.update({"Source Volume": self.volume.title})
        return summary
