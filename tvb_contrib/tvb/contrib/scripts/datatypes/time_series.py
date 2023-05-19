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

from copy import deepcopy
from enum import Enum

import numpy
import xarray as xr
from six import string_types
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import List, Attr
from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.contrib.scripts.utils.data_structures_utils import is_integer, monopolar_to_bipolar
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.sensors import Sensors, SensorsEEG, SensorsMEG, SensorsInternal
from tvb.datatypes.time_series import TimeSeries as TimeSeriesTVB
from tvb.datatypes.time_series import TimeSeriesEEG as TimeSeriesEEGTVB
from tvb.datatypes.time_series import TimeSeriesMEG as TimeSeriesMEGTVB
from tvb.datatypes.time_series import TimeSeriesRegion as TimeSeriesRegionTVB
from tvb.datatypes.time_series import TimeSeriesSEEG as TimeSeriesSEEGTVB
from tvb.datatypes.time_series import TimeSeriesSurface as TimeSeriesSurfaceTVB
from tvb.datatypes.time_series import TimeSeriesVolume as TimeSeriesVolumeTVB
from tvb.simulator.plot.utils import ensure_list

logger = get_logger(__name__)


class TimeSeriesDimensions(Enum):
    TIME = "Time"
    VARIABLES = "State Variable"

    SPACE = "Space"
    REGIONS = "Region"
    VERTEXES = "Vertex"
    SENSORS = "Sensor"

    SAMPLES = "Sample"
    MODES = "Mode"

    X = "x"
    Y = "y"
    Z = "z"


LABELS_ORDERING = [TimeSeriesDimensions.TIME.value,
                   TimeSeriesDimensions.VARIABLES.value,
                   TimeSeriesDimensions.SPACE.value,
                   TimeSeriesDimensions.SAMPLES.value]


class PossibleVariables(Enum):
    LFP = "lfp"
    SOURCE = "source"
    SENSORS = "sensors"
    EEG = "eeg"
    MEEG = "meeg"
    SEEG = "seeg"
    X = "x"
    Y = "y"
    Z = "z"


def prepare_4d(data, logger):
    if data.ndim < 2:
        logger.error("The data array is expected to be at least 2D!")
        raise ValueError
    if data.ndim < 4:
        if data.ndim == 2:
            data = numpy.expand_dims(data, 2)
        data = numpy.expand_dims(data, 3)
    return data


class TimeSeries(TimeSeriesTVB, BaseModel):

    title = Attr(str, default="Time Series")

    logger = get_logger(__name__)

    def __init__(self, **kwargs):
        data = kwargs.pop("data", numpy.empty((0, 0, 0, 0)))
        if isinstance(data, xr.DataArray):
            super(TimeSeries, self).__init__(data=prepare_4d(data.values, self.logger), **kwargs)
            self.initialze_from_xarray_DataArray(data)
        else:
            super(TimeSeries, self).__init__(data=prepare_4d(data, self.logger), **kwargs)

    def initialze_from_xarray_DataArray(self, xrdtarr):
        # We assume that time is in the first dimension
        labels_ordering = xrdtarr.coords.dims
        labels_dimensions = {}
        for dim in labels_ordering[1:]:
            labels_dimensions[dim] = numpy.array(xrdtarr.coords[dim].values).tolist()
        if xrdtarr.name is not None and len(xrdtarr.name) > 0:
            self.title = xrdtarr.name
        self.labels_ordering = labels_ordering
        self.labels_dimensions = labels_dimensions
        if xrdtarr.size == 0:
            self.time = numpy.empty((0,))
        else:
            self.time = numpy.array(xrdtarr.coords[labels_ordering[0]].values)

    def duplicate(self, **kwargs):
        duplicate = deepcopy(self)
        duplicate.data = prepare_4d(kwargs.pop('data', self.data), self.logger)
        for attr, value in kwargs.items():
            setattr(duplicate, attr, value)
        duplicate.configure()
        return duplicate

    def _assert_index(self, index):
        assert (index >= 0 and index < self.number_of_dimensions)
        return index

    def get_dimension_index(self, dim_name_or_index):
        if is_integer(dim_name_or_index):
            return self._assert_index(dim_name_or_index)
        elif isinstance(dim_name_or_index, string_types):
            return self._assert_index(self.labels_ordering.index(dim_name_or_index))
        else:
            raise ValueError("dim_name_or_index is neither a string nor an integer!")

    def get_dimension_name(self, dim_index):
        try:
            return self.labels_ordering[dim_index]
        except IndexError:
            self.logger.error("Cannot access index %d of labels ordering: %s!" %
                              (int(dim_index), str(self.labels_ordering)))

    def get_dimension_labels(self, dimension_label_or_index):
        if not isinstance(dimension_label_or_index, string_types):
            dimension_label_or_index = self.get_dimension_name(dimension_label_or_index)
        try:
            return self.labels_dimensions[dimension_label_or_index]
        except KeyError:
            self.logger.error("There are no %s labels defined for this instance. Its shape is: %s" %
                              (dimension_label_or_index, self.data.shape))
            raise

    def update_dimension_names(self, dim_names, dim_indices=None):
        dim_names = ensure_list(dim_names)
        if dim_indices is None:
            dim_indices = list(range(len(dim_names)))
        else:
            dim_indices = ensure_list(dim_indices)
        labels_ordering = list(self.labels_ordering)
        for dim_name, dim_index in zip(dim_names, dim_indices):
            labels_ordering[dim_index] = dim_name
            try:
                old_dim_name = self.labels_ordering[dim_index]
                dim_labels = list(self.labels_dimensions[old_dim_name])
                del self.labels_dimensions[old_dim_name]
                self.labels_dimensions[dim_name] = dim_labels
            except:
                pass
        self.labels_ordering = labels_ordering

    def _check_indices(self, indices, dimension):
        dim_index = self.get_dimension_index(dimension)
        for index in ensure_list(indices):
            if index < 0 or index > self.data.shape[dim_index]:
                self.logger.error("Some of the given indices are out of %s range: [0, %s]" %
                                  (self.get_dimension_name(dim_index), self.data.shape[dim_index]))
                raise IndexError

    def _check_space_indices(self, list_of_index):
        self._check_indices(list_of_index, 2)

    def _check_variables_indices(self, list_of_index):
        self._check_indices(list_of_index, 1)

    def _get_index_of_label(self, labels, dimension):
        indices = []
        data_labels = list(self.get_dimension_labels(dimension))
        for label in ensure_list(labels):
            try:
                indices.append(data_labels.index(label))
            # TODO: force list error here to be IndexError instead of ValueError
            except IndexError:
                self.logger.error("Cannot access index of %s label: %s. Existing %s labels: %s" % (
                    dimension, label, dimension, str(data_labels)))
                raise IndexError
        return indices

    def _slice_to_indices(self, slices, dim_index):
        indices = []
        for slice in ensure_list(slices):
            if slice.start is None:
                start = 0
            else:
                start = slice.start
            if slice.stop is None:
                stop = self.data.shape[dim_index]
            else:
                stop = slice.stop
            if slice.step is None:
                step = 1
            else:
                step = slice.step
                slice.step = 1
            indices.append(list(range(start, stop, step)))
        if len(indices) == 1:
            return indices[0]
        return tuple(indices)

    def _get_index_for_slice_label(self, slice_label, slice_idx):
        return self._get_index_of_label(slice_label,
                                        self.get_dimension_name(slice_idx))[0]

    def _check_for_string_or_float_slice_indices(self, current_slice, slice_idx):
        slice_start = current_slice.start
        slice_stop = current_slice.stop

        if isinstance(slice_start, string_types) or isinstance(slice_start, float):
            slice_start = self._get_index_for_slice_label(slice_start, slice_idx)
        if isinstance(slice_stop, string_types) or isinstance(slice_stop, float):
            slice_stop = self._get_index_for_slice_label(slice_stop, slice_idx) + 1  # inclusive last label

        return slice(slice_start, slice_stop, current_slice.step)

    def _get_string_slice_index(self, current_slice_string, slice_idx):
        return self._get_index_for_slice_label(current_slice_string, slice_idx)

    def _process_slice(self, slice_arg, idx):
        if isinstance(slice_arg, slice):
            return self._check_for_string_or_float_slice_indices(slice_arg, idx)
        else:
            if isinstance(slice_arg, string_types):
                return self._get_string_slice_index(slice_arg, idx)
            else:
                return slice_arg

    def _process_slice_tuple(self, slice_tuple):
        if not isinstance(slice_tuple, (tuple, list)):
            slice_tuple = (slice_tuple, )
        n_slices = len(slice_tuple)
        assert (n_slices >= 0 and n_slices <= self.number_of_dimensions)
        slice_list = []
        for idx, current_slice in enumerate(slice_tuple):
            slice_list.append(self._process_slice(current_slice, idx))
        return tuple(slice_list)

    def slice_data_across_dimension_by_index(self, indices, dimension, **kwargs):
        dim_index = self.get_dimension_index(dimension)
        indices = ensure_list(indices)
        self._check_indices(indices, dim_index)
        slices = [slice(None)] * 4
        slices[dim_index] = indices
        data = self.data[tuple(slices)]
        labels_dimensions = deepcopy(self.labels_dimensions)
        try:
            labels_dimensions[self.get_dimension_name(dim_index)] = \
                (numpy.array(labels_dimensions[self.get_dimension_name(dim_index)])[indices]).tolist()
        except:
            self.logger.warning("Failed to get labels subset for indices %s of dimension %d!"
                             % (str(indices), dim_index))
            labels_dimensions[self.get_dimension_name(dim_index)] = []
        return self.duplicate(data=data, labels_dimensions=labels_dimensions, **kwargs)

    def slice_data_across_dimension_by_label(self, labels, dimension, **kwargs):
        dim_index = self.get_dimension_index(dimension)
        return self.slice_data_across_dimension_by_index(
            self._get_index_of_label(labels,
                                     self.get_dimension_name(dim_index)),
            dim_index, **kwargs)

    def slice_data_across_dimension_by_slice(self, slice_arg, dimension, **kwargs):
        dim_index = self.get_dimension_index(dimension)
        return self.slice_data_across_dimension_by_index(
            self._slice_to_indices(
                self._process_slice(slice_arg, dim_index), dim_index),
            dim_index, **kwargs)

    def _index_or_label_or_slice(self, inputs):
        inputs = ensure_list(inputs)
        if numpy.all([is_integer(inp) for inp in inputs]):
            return "index"
        elif numpy.all([isinstance(inp, string_types) for inp in inputs]):
            return "label"
        elif numpy.all([isinstance(inp, slice) for inp in inputs]):
            return "slice"
        else:
            raise ValueError("input %s is not of type integer, string or slice!" % str(inputs))

    def slice_data_across_dimension(self, inputs, dimension, **kwargs):
        dim_index = self.get_dimension_index(dimension)
        return getattr(self,
                       "slice_data_across_dimension_by_%s" %
                       self._index_or_label_or_slice(inputs))(inputs, dim_index, **kwargs)

    def get_data_from_slice(self, slice_tuple, **kwargs):
        output_data = self.slice_data_across_dimension_by_slice(slice_tuple[0], **kwargs)
        for this_slice in enumerate(slice_tuple[1:]):
            output_data = output_data.slice_data_across_dimension_by_slice(this_slice, **kwargs)
        return output_data

    def get_times_by_index(self, list_of_times_indices, **kwargs):
        return self.slice_data_across_dimension_by_index(list_of_times_indices, 0, **kwargs)

    def get_times(self, list_of_times, **kwargs):
        return self.slice_data_across_dimension(list_of_times, 0, **kwargs)

    def get_indices_for_state_variables(self, sv_labels):
        return self._get_index_of_label(sv_labels, self.get_dimension_name(1))

    def get_state_variables_by_index(self, sv_indices, **kwargs):
        return self.slice_data_across_dimension_by_index(sv_indices, 1, **kwargs)

    def get_state_variables_by_label(self, sv_labels, **kwargs):
        return self.slice_data_across_dimension_by_label(sv_labels, 1, **kwargs)

    def get_state_variables_by_slice(self, slice_arg, **kwargs):
        return self.slice_data_across_dimension_by_slice(slice_arg, 1, **kwargs)

    def get_state_variables(self, sv_inputs, **kwargs):
        return getattr(self,
                       "slice_data_across_dimension_by_%s" %
                       self._index_or_label_or_slice(sv_inputs))(sv_inputs, 1, **kwargs)

    def get_indices_for_labels(self, region_labels):
        return self._get_index_of_label(region_labels, self.get_dimension_name(2))

    def get_subspace_by_index(self, list_of_index, **kwargs):
        return self.slice_data_across_dimension_by_index(list_of_index, 2, **kwargs)

    def get_subspace_by_label(self, list_of_labels, **kwargs):
        return self.slice_data_across_dimension_by_label(list_of_labels, 2, **kwargs)

    def get_subspace_by_slice(self, slice_arg, **kwargs):
        return self.slice_data_across_dimension_by_slice(slice_arg, 2, **kwargs)

    def get_subspace(self, subspace_inputs, **kwargs):
        return getattr(self,
                       "slice_data_across_dimension_by_%s" %
                       self._index_or_label_or_slice(subspace_inputs))(subspace_inputs, 2, **kwargs)

    def get_modes_by_index(self, list_of_index, **kwargs):
        return self.slice_data_across_dimension_by_index(list_of_index, 3, **kwargs)

    def get_modes_by_label(self, list_of_labels, **kwargs):
        return self.slice_data_across_dimension_by_label(list_of_labels, 3, **kwargs)

    def get_modes_by_slice(self, slice_arg, **kwargs):
        return self.slice_data_across_dimension_by_slice(slice_arg, 3, **kwargs)

    def get_modes(self, modes_inputs, **kwargs):
        return getattr(self,
                       "slice_data_across_dimension_by_%s" %
                       self._index_or_label_or_slice(modes_inputs))(modes_inputs, 3, **kwargs)

    # TODO: Adjust this behavior to TimeSeriesXarray,
    # i.e., getting, or setting an object of this class
    def __getitem__(self, slice_tuple):
        return self.data[self._process_slice_tuple(slice_tuple)]

    def __setitem__(self, slice_tuple, values):
        self.data[self._process_slice_tuple(slice_tuple)] = values

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return self.data.shape

    @property
    def time_length(self):
        return self.data.shape[0]

    @property
    def number_of_variables(self):
        return self.data.shape[1]

    @property
    def number_of_labels(self):
        return self.data.shape[2]

    @property
    def number_of_samples(self):
        return self.data.shape[3]

    @property
    def end_time(self):
        return self.start_time + (self.time_length - 1) * self.sample_period

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def time_unit(self):
        return self.sample_period_unit

    @property
    def sample_rate(self):
        if len(self.sample_period_unit) > 0 and self.sample_period_unit[0] == "m":
            return 1000.0 / self.sample_period
        return 1.0 / self.sample_period

    @property
    def space_labels(self):
        return numpy.array(self.labels_dimensions.get(self.labels_ordering[2], []))

    @property
    def variables_labels(self):
        return numpy.array(self.labels_dimensions.get(self.labels_ordering[1], []))

    @property
    def number_of_dimensions(self):
        return self.nr_dimensions

    @property
    def squeezed(self):
        return numpy.squeeze(self.data)

    def _get_time_unit_for_index(self, time_index):
        return self.start_time + time_index * self.sample_period

    def _get_index_for_time_unit(self, time_unit):
        return int((time_unit - self.start_time) / self.sample_period)

    def get_time_window(self, index_start, index_end, **kwargs):
        if index_start < 0 or index_end > self.data.shape[0]:
            self.logger.error("The time indices are outside time series interval: [%s, %s]" %
                              (0, self.data.shape[0]))
            raise IndexError
        subtime_data = self.data[index_start:index_end, :, :, :]
        if subtime_data.ndim == 3:
            subtime_data = numpy.expand_dims(subtime_data, 0)
        return self.duplicate(data=subtime_data, time=self.time[index_start:index_end], **kwargs)

    def get_time_window_by_units(self, unit_start, unit_end, **kwargs):
        end_time = self.end_time
        if unit_start < self.start_time or unit_end > end_time:
            self.logger.error("The time units are outside time series interval: [%s, %s]" %
                              (self.start_time, end_time))
            raise IndexError
        index_start = self._get_index_for_time_unit(unit_start)
        index_end = self._get_index_for_time_unit(unit_end)
        return self.get_time_window(index_start, index_end)

    def decimate_time(self, new_sample_period, **kwargs):
        if new_sample_period % self.sample_period != 0:
            self.logger.error("Cannot decimate time if new time step is not a multiple of the old time step")
            raise ValueError

        index_step = int(new_sample_period / self.sample_period)
        time_data = self.data[::index_step, :, :, :]
        return self.duplicate(data=time_data, sample_period=new_sample_period, **kwargs)

    def get_sample_window(self, index_start, index_end, **kwargs):
        subsample_data = self.data[:, :, :, index_start:index_end]
        if subsample_data.ndim == 3:
            subsample_data = numpy.expand_dims(subsample_data, 3)
        return self.duplicate(data=subsample_data, **kwargs)

    def swapaxes(self, ax1, ax2):
        labels_ordering = list(self.labels_ordering)
        labels_ordering[ax1] = self.labels_ordering[ax2]
        labels_ordering[ax2] = self.labels_ordering[ax1]
        return self.duplicate(data=numpy.swapaxes(self.data, ax1, ax2), labels_ordering=labels_ordering)

    def configure(self):
        super(TimeSeries, self).configure()
        if self.time is None:
            self.time = numpy.arange(self.start_time, self.end_time + self.sample_period, self.sample_period)
        else:
            self.start_time = 0.0
            self.sample_period = 0.0
            if len(self.time) > 0:
                self.start_time = self.time[0]
            if len(self.time) > 1:
                self.sample_period = numpy.mean(numpy.diff(self.time))
        self.labels_dimensions[self.labels_ordering[0]] = self.time
        for key, value in self.labels_dimensions.items():
            self.labels_dimensions[key] = list(value)
        self.labels_ordering = list(self.labels_ordering)

    def to_tvb_instance(self, datatype=TimeSeriesTVB, **kwargs):
        return super(TimeSeries, self).to_tvb_instance(datatype, **kwargs)


class TimeSeriesBrain(TimeSeries):

    def get_source(self):
        if self.labels_ordering[1] not in self.labels_dimensions.keys():
            self.logger.error("No state variables are defined for this instance!")
            raise ValueError
        if PossibleVariables.SOURCE.value in self.variables_labels:
            return self.get_state_variables_by_label(PossibleVariables.SOURCE.value)

    @property
    def brain_labels(self):
        return self.space_labels


class TimeSeriesRegion(TimeSeriesBrain, TimeSeriesRegionTVB):
    labels_ordering = List(of=str, default=(TimeSeriesDimensions.TIME.value, TimeSeriesDimensions.VARIABLES.value,
                                            TimeSeriesDimensions.REGIONS.value, TimeSeriesDimensions.SAMPLES.value))

    title = Attr(str, default="Region Time Series")

    @property
    def region_labels(self):
        return self.space_labels

    def to_tvb_instance(self, **kwargs):
        return super(TimeSeriesRegion, self).to_tvb_instance(TimeSeriesRegionTVB, **kwargs)


class TimeSeriesSurface(TimeSeriesBrain, TimeSeriesSurfaceTVB):
    labels_ordering = List(of=str, default=(TimeSeriesDimensions.TIME.value, TimeSeriesDimensions.VARIABLES.value,
                                            TimeSeriesDimensions.VERTEXES.value, TimeSeriesDimensions.SAMPLES.value))

    title = Attr(str, default="Surface Time Series")

    @property
    def surface_labels(self):
        return self.space_labels

    def to_tvb_instance(self, **kwargs):
        return super(TimeSeriesSurface, self).to_tvb_instance(TimeSeriesSurfaceTVB, **kwargs)


class TimeSeriesVolume(TimeSeries, TimeSeriesVolumeTVB):
    labels_ordering = List(of=str, default=(TimeSeriesDimensions.TIME.value, TimeSeriesDimensions.X.value,
                                            TimeSeriesDimensions.Y.value, TimeSeriesDimensions.Z.value))

    title = Attr(str, default="Volume Time Series")

    @property
    def volume_labels(self):
        return self.space_labels

    def to_tvb_instance(self, **kwargs):
        return super(TimeSeriesVolume, self).to_tvb_instance(TimeSeriesVolumeTVB, **kwargs)


class TimeSeriesSensors(TimeSeries):
    labels_ordering = List(of=str, default=(TimeSeriesDimensions.TIME.value, TimeSeriesDimensions.VARIABLES.value,
                                            TimeSeriesDimensions.SENSORS.value, TimeSeriesDimensions.SAMPLES.value))

    title = Attr(str, default="Sensor Time Series")

    @property
    def sensor_labels(self):
        return self.space_labels

    def get_bipolar(self, **kwargs):
        bipolar_labels, bipolar_inds = monopolar_to_bipolar(self.space_labels)
        data = self.data[:, :, bipolar_inds[0]] - self.data[:, :, bipolar_inds[1]]
        bipolar_labels_dimensions = deepcopy(self.labels_dimensions)
        bipolar_labels_dimensions[self.labels_ordering[2]] = list(bipolar_labels)
        return self.duplicate(data=data, labels_dimensions=bipolar_labels_dimensions, **kwargs)


class TimeSeriesEEG(TimeSeriesSensors, TimeSeriesEEGTVB):
    title = Attr(str, default="EEG Time Series")

    def configure(self):
        super(TimeSeriesSensors, self).configure()
        if isinstance(self.sensors, Sensors) and not isinstance(self.sensors, SensorsEEG):
            self.logger.warning(
                "Creating %s with sensors of type %s!" % (self.__class__.__name__, self.sensors.__class__.__name__))

    @property
    def EEGsensor_labels(self):
        return self.space_labels

    def to_tvb_instance(self, **kwargs):
        return super(TimeSeriesEEG, self).to_tvb_instance(TimeSeriesEEGTVB, **kwargs)


class TimeSeriesMEG(TimeSeriesSensors, TimeSeriesMEGTVB):
    title = Attr(str, default="MEG Time Series")

    def configure(self):
        super(TimeSeriesSensors, self).configure()
        if isinstance(self.sensors, Sensors) and not isinstance(self.sensors, SensorsMEG):
            self.logger.warning(
                "Creating %s with sensors of type %s!" % (self.__class__.__name__, self.sensors.__class__.__name__))

    @property
    def MEGsensor_labels(self):
        return self.space_labels

    def to_tvb_instance(self, **kwargs):
        return super(TimeSeriesMEG, self).to_tvb_instance(TimeSeriesMEGTVB, **kwargs)


class TimeSeriesSEEG(TimeSeriesSensors, TimeSeriesSEEGTVB):
    title = Attr(str, default="SEEG Time Series")

    def configure(self):
        super(TimeSeriesSensors, self).configure()
        if isinstance(self.sensors, Sensors) and not isinstance(self.sensors, SensorsInternal):
            self.logger.warning(
                "Creating %s with sensors of type %s!" % (self.__class__.__name__, self.sensors.__class__.__name__))

    @property
    def SEEGsensor_labels(self):
        return self.space_labels

    def to_tvb_instance(self, **kwargs):
        return super(TimeSeriesSEEG, self).to_tvb_instance(TimeSeriesSEEGTVB, **kwargs)


TimeSeriesDict = {TimeSeries.__name__: TimeSeries,
                  TimeSeriesRegion.__name__: TimeSeriesRegion,
                  TimeSeriesVolume.__name__: TimeSeriesVolume,
                  TimeSeriesSurface.__name__: TimeSeriesSurface,
                  TimeSeriesEEG.__name__: TimeSeriesEEG,
                  TimeSeriesMEG.__name__: TimeSeriesMEG,
                  TimeSeriesSEEG.__name__: TimeSeriesSEEG}

if __name__ == "__main__":
    connectivity = Connectivity.from_file()
    kwargs = {"data": numpy.ones((4, 2, 10, 1)), "connectivity": connectivity,
              "start_time": 0.0, "labels_dimensions": {LABELS_ORDERING[1]: ["x", "y"]}}
    ts = TimeSeriesRegion(**kwargs)
    tsy = ts.y
    print(tsy.squeezed)
