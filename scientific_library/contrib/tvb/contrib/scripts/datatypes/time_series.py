# -*- coding: utf-8 -*-

from copy import deepcopy
from enum import Enum

import numpy
from six import string_types
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import List, Attr
from tvb.datatypes.sensors import Sensors, SensorsEEG, SensorsMEG, SensorsInternal
from tvb.datatypes.time_series import TimeSeries as TimeSeriesTVB
from tvb.datatypes.time_series import TimeSeriesEEG as TimeSeriesEEGTVB
from tvb.datatypes.time_series import TimeSeriesMEG as TimeSeriesMEGTVB
from tvb.datatypes.time_series import TimeSeriesRegion as TimeSeriesRegionTVB
from tvb.datatypes.time_series import TimeSeriesSEEG as TimeSeriesSEEGTVB
from tvb.datatypes.time_series import TimeSeriesSurface as TimeSeriesSurfaceTVB
from tvb.datatypes.time_series import TimeSeriesVolume as TimeSeriesVolumeTVB
from tvb.simulator.plot.utils import ensure_list

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.contrib.scripts.utils.data_structures_utils import is_integer, monopolar_to_bipolar


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


def prepare_4d(data, LOG=logger):
    if data.ndim < 2:
        LOG.error("The data array is expected to be at least 2D!")
        raise ValueError
    if data.ndim < 4:
        if data.ndim == 2:
            data = numpy.expand_dims(data, 2)
        data = numpy.expand_dims(data, 3)
    return data


def _slice_data(data, slice_tuple):
    output_data = data[slice_tuple[0]]
    preslices = [slice(None)]
    for i_dim, this_slice in enumerate(slice_tuple[1:]):
        current_slices = tuple(preslices + [this_slice])
        output_data = output_data[current_slices]
        preslices.append(slice(None))
    return output_data


class TimeSeries(TimeSeriesTVB, BaseModel):

    @property
    def name(self):
        return self.title

    @property
    def shape(self):
        return self.data.shape

    @property
    def number_of_dimensions(self):
        return self.nr_dimensions

    @property
    def size(self):
        return self.data.size

    def _return_shape_of_dim(self, dim):
        try:
            return self.data.shape[dim]
        except:
            return None

    @property
    def time_length(self):
        return self._return_shape_of_dim(0)

    @property
    def number_of_variables(self):
        return self._return_shape_of_dim(1)

    @property
    def number_of_labels(self):
        return self._return_shape_of_dim(2)

    @property
    def number_of_samples(self):
        return self._return_shape_of_dim(3)

    @property
    def end_time(self):
        return self.start_time + (self.time_length - 1) * self.sample_period

    @property
    def duration(self):
        return (self.time_length - 1) * self.sample_period

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
    def squeezed(self):
        return self.data.squeeze()

    @property
    def flattened(self):
        return self.data.flatten()

    def from_xarray_DataArray(self, xrdtarr, **kwargs):
        # We assume that time is in the first dimension
        labels_ordering = xrdtarr.coords.dims
        labels_dimensions = {}
        for dim in labels_ordering[1:]:
            labels_dimensions[dim] = numpy.array(xrdtarr.coords[dim].values).tolist()
        if xrdtarr.name is not None and len(xrdtarr.name) > 0:
            kwargs.update({"title": xrdtarr.name})
        if xrdtarr.size == 0:
            return self.duplicate(data=numpy.empty((0, 0, 0, 0)),
                                  time=numpy.empty((0,)),
                                  labels_ordering=labels_ordering,
                                  labels_dimensions=labels_dimensions,
                                  **kwargs)
        return self.duplicate(data=xrdtarr.values,
                              time=numpy.array(xrdtarr.coords[labels_ordering[0]].values),
                              labels_ordering=labels_ordering,
                              labels_dimensions=labels_dimensions,
                              **kwargs)

    def configure(self):
        super(TimeSeries, self).configure()
        if self.time is None or len(self.time) == 0:
            self.time = numpy.arange(self.start_time, self.end_time + self.sample_period, self.sample_period)
        else:
            self.start_time = 0.0
            self.sample_period = 0.0
            self.start_time = self.time[0]
            if len(self.time) > 1:
                self.sample_period = numpy.mean(numpy.diff(self.time))
        for key, value in self.labels_dimensions.items():
            self.labels_dimensions[key] = list(value)
        self.labels_ordering = list(self.labels_ordering)

    def __init__(self, data=None, **kwargs):
        super(TimeSeries, self).__init__(**kwargs)
        if data is not None:
            self.data = prepare_4d(data, logger)
            self.configure()

    def duplicate(self, **kwargs):
        duplicate = deepcopy(self)
        for attr, value in kwargs.items():
            setattr(duplicate, attr, value)
        duplicate.data = prepare_4d(duplicate.data, logger)
        duplicate.configure()
        return duplicate

    def to_tvb_instance(self, datatype=TimeSeriesTVB, **kwargs):
        return super(TimeSeries, self).to_tvb_instance(datatype, **kwargs)

    def _assert_index(self, index):
        if (index < 0 or index >= self.number_of_dimensions):
            raise IndexError("index %d is not within the dimensions [0, %d] of this TimeSeries data:\n%s"
                             % (index, self.number_of_dimensions, str(self)))
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
            logger.error("Cannot access index %d of labels ordering: %s!" %
                              (int(dim_index), str(self.labels_ordering)))

    def get_dimension_labels(self, dimension_label_or_index):
        if not isinstance(dimension_label_or_index, string_types):
            dimension_label_or_index = self.get_dimension_name(dimension_label_or_index)
        try:
            return self.labels_dimensions[dimension_label_or_index]
        except KeyError:
            logger.error("There are no %s labels defined for this instance: %s",
                              (dimension_label_or_index, str(self.labels_dimensions)))
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

    # -----------------------general slicing methods-------------------------------------------

    def _check_indices(self, indices, dimension):
        dim_index = self.get_dimension_index(dimension)
        for index in ensure_list(indices):
            if index < 0 or index > self.data.shape[dim_index]:
                logger.error("Some of the given indices are out of %s range: [0, %s]",
                                  (self.get_dimension_name(dim_index), self.data.shape[dim_index]))
                raise IndexError
        return indices

    def _check_time_indices(self, list_of_index):
        return self._check_indices(list_of_index, 0)

    def _check_variables_indices(self, list_of_index):
        return self._check_indices(list_of_index, 1)

    def _check_space_indices(self, list_of_index):
        return self._check_indices(list_of_index, 2)

    def _check_modes_indices(self, list_of_index):
        return self._check_indices(list_of_index, 2)

    def _get_index_of_label(self, labels, dimension):
        indices = []
        data_labels = list(self.get_dimension_labels(dimension))
        for label in ensure_list(labels):
            try:
                indices.append(data_labels.index(label))
            # TODO: force list error here to be IndexError instead of ValueError
            except IndexError:
                logger.error("Cannot access index of %s label: %s. Existing %s labels: %s" % (
                    dimension, label, dimension, str(data_labels)))
                raise IndexError
        return indices

    def _get_index_for_slice_label(self, slice_label, slice_idx):
        return self._get_index_of_label(slice_label,
                                        self.get_dimension_name(slice_idx))[0]

    def _check_for_string_or_float_slice_indices(self, current_slice, slice_idx):
        slice_start = current_slice.start
        slice_stop = current_slice.stop

        if isinstance(slice_start, string_types) or isinstance(slice_start, float):
            slice_start = self._get_index_for_slice_label(slice_start, slice_idx)
        if isinstance(slice_stop, string_types) or isinstance(slice_stop, float):
            slice_stop = self._get_index_for_slice_label(slice_stop, slice_idx)

        return slice(slice_start, slice_stop, current_slice.step)

    def _process_slice(self, this_slice, dim_index):
        if isinstance(this_slice, slice):
            return self._check_for_string_or_float_slice_indices(this_slice, dim_index)
        else:
            # If not a slice, it will be an iterable:
            for i_slc, slc in enumerate(this_slice):
                if isinstance(slc, string_types) or isinstance(slc, float):
                    dim_name = self.get_dimension_name(dim_index)
                    this_slice[i_slc] = ensure_list(self.labels_dimensions[dim_name]).index(slc)
                else:
                    this_slice[i_slc] = slc
            return this_slice

    def _process_slices(self, slice_tuple):
        n_slices = len(slice_tuple)
        assert (n_slices >= 0 and n_slices <= self.number_of_dimensions)
        slice_list = []
        for idx, current_slice in enumerate(slice_tuple):
            slice_list.append(self._process_slice(current_slice, idx))
        return tuple(slice_list)

    def _slice_to_indices(self, slice_arg, dim_index):
        if slice_arg.start is None:
            start = 0
        else:
            start = slice_arg.start
        if slice_arg.stop is None:
            stop = self.data.shape[dim_index]
        else:
            stop = slice_arg.stop
        if slice_arg.step is None:
            step = 1
        else:
            step = slice_arg.step
        return self._check_indices(list(range(start, stop, step)), dim_index)

    def _slices_to_indices(self, slices):
        indices = []
        for dim_index, slice_arg in enumerate(ensure_list(slices)):
            if isinstance(slice_arg, slice):
                indices.append(self._slice_to_indices(slice_arg, dim_index))
            else:
                # Assuming already indices
                indices.append(slice_arg)
        if len(indices) == 1:
            return indices[0]
        return tuple(indices)

    def _assert_array_indices(self, slice_tuple):
        if is_integer(slice_tuple) or isinstance(slice_tuple, string_types):
            return ([slice_tuple],)
        else:
            if isinstance(slice_tuple, slice):
                slice_tuple = (slice_tuple,)
            slice_list = []
            for slc in slice_tuple:
                if is_integer(slc) or isinstance(slc, string_types):
                    slice_list.append([slc])
                else:
                    slice_list.append(slc)
            return tuple(slice_list)

    def _slice_time_index(self, time_inds, **kwargs):
        time = kwargs.pop("time", None)
        if time is None or len(time) == 0:
            try:
                time = self.time[time_inds]
            except:
                time = self.time
        if time is None or len(time) == 0:
            start_time = kwargs.pop("start_time", self.start_time)
            sample_period = kwargs.pop("sample_period", self.sample_period)
        else:
            start_time = kwargs.pop("start_time", time[0])
            if len(time) > 1:
                sample_period = kwargs.pop("sample_period", numpy.diff(time).mean())
            else:
                sample_period = kwargs.pop("sample_period", self.sample_period)
        return start_time, sample_period

    def _slice_dimensions_labels(self, indices, **kwargs):
        labels_ordering = kwargs.pop("labels_ordering", self.labels_ordering)
        labels_dimensions = dict(self.labels_dimensions)
        for ii, inds in enumerate(indices):
            if len(inds) > 0:
                try:
                    dim_name = labels_ordering[ii]
                    labels_dimensions[dim_name] = \
                        (numpy.array(labels_dimensions[dim_name])[inds]).tolist()
                except:
                    pass
        labels_dimensions.update(kwargs.pop("labels_dimensions", {}))
        return labels_ordering, labels_dimensions

    def _get_item(self, slice_tuple, **kwargs):
        slice_tuple = self._process_slices(self._assert_array_indices(slice_tuple))
        indices = self._slices_to_indices(slice_tuple)
        start_time, sample_period = self._slice_time_index(indices[0], **kwargs)
        labels_ordering, labels_dimensions = self._slice_dimensions_labels(indices, **kwargs)
        return self.duplicate(data=_slice_data(self.data, tuple(indices)),
                              start_time=start_time, sample_period=sample_period,
                              labels_ordering=labels_ordering, labels_dimensions=labels_dimensions, **kwargs)

    # Return a TimeSeries object
    def __getitem__(self, slice_tuple):
        return self._get_item(slice_tuple)

    def __setitem__(self, slice_tuple, values):
        slice_tuple = self._assert_array_indices(slice_tuple)
        self.data[self._process_slices(slice_tuple)] = values

    # -----------------------slicing by a particular dimension-------------------------------------------

    def slice_data_across_dimension_by_index(self, indices, dimension, **kwargs):
        dim_index = self.get_dimension_index(dimension)
        indices = ensure_list(indices)
        self._check_indices(indices, dim_index)
        slices = [slice(None)] * self.nr_dimensions
        slices[dim_index] = indices
        if dim_index == 0:
            start_time, sample_period = self._slice_time_index(indices, **kwargs)
        else:
            start_time = self.start_time
            sample_period = self.sample_period
        all_indices = [[]] * self.nr_dimensions
        all_indices[dim_index] = indices
        labels_ordering, labels_dimensions = self._slice_dimensions_labels(all_indices, **kwargs)
        return self.duplicate(data=self.data[tuple(slices)],
                              start_time=start_time, sample_period=sample_period,
                              labels_ordering=labels_ordering, labels_dimensions=labels_dimensions, **kwargs)

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

    def _index_or_label_or_slice(self, inputs, dim):
        inputs = ensure_list(inputs)
        if numpy.all([is_integer(inp) for inp in inputs]):
            return "index", inputs
        elif numpy.all([isinstance(inp, string_types) for inp in inputs]):
            return "label", inputs
        elif isinstance(inputs, slice):
            return "slice", inputs
        elif numpy.all([isinstance(inp, string_types) or is_integer(inp) for inp in inputs]):
            # resolve mixed integer and label index:
            inputs = self._process_slice(inputs, dim)
            return "index", inputs
        else:
            raise ValueError("input %s is not of type integer, string or slice!" % str(inputs))

    def slice_data_across_dimension(self, inputs, dimension, **kwargs):
        dim_index = self.get_dimension_index(dimension)
        index_or_label_or_slice, inputs = self._index_or_label_or_slice(inputs, dim_index)
        if index_or_label_or_slice == "index":
            return self. slice_data_across_dimension_by_index(inputs, dim_index, **kwargs)
        elif index_or_label_or_slice == "label":
            return self. slice_data_across_dimension_by_label(inputs, dim_index, **kwargs)
        elif index_or_label_or_slice == "slice":
            return self. slice_data_across_dimension_by_slice(inputs, dim_index, **kwargs)
        else:
            raise ValueError("input %s is not of type integer, string or slice!" % str(inputs))

    def get_times_by_index(self, list_of_times_indices, **kwargs):
        return self.slice_data_across_dimension_by_index(list_of_times_indices, 0, **kwargs)

    def _get_time_unit_for_index(self, time_index):
        return self.start_time + time_index * self.sample_period

    def _get_index_for_time_unit(self, time_unit):
        return int((time_unit - self.start_time) / self.sample_period)

    def get_time_window(self, index_start, index_end, **kwargs):
        if index_start < 0 or index_end > self.data.shape[0]:
            logger.error("The time indices are outside time series interval: [%s, %s]" %
                              (0, self.data.shape[0]))
            raise IndexError
        subtime_data = self.data[index_start:index_end, :, :, :]
        if subtime_data.ndim == 3:
            subtime_data = numpy.expand_dims(subtime_data, 0)
        return self.duplicate(data=subtime_data, time=self.time[index_start:index_end], **kwargs)

    def get_time_window_by_units(self, unit_start, unit_end, **kwargs):
        end_time = self.end_time
        if unit_start < self.start_time or unit_end > end_time:
            logger.error("The time units are outside time series interval: [%s, %s]" %
                              (self.start_time, end_time))
            raise IndexError
        index_start = self._get_index_for_time_unit(unit_start)
        index_end = self._get_index_for_time_unit(unit_end)
        return self.get_time_window(index_start, index_end)

    def get_times(self, list_of_times, **kwargs):
        return self.slice_data_across_dimension(list_of_times, 0, **kwargs)

    def decimate_time(self, new_sample_period, **kwargs):
        if new_sample_period % self.sample_period != 0:
            logger.error("Cannot decimate time if new time step is not a multiple of the old time step")
            raise ValueError
        index_step = int(new_sample_period / self.sample_period)
        time_data = self.data[::index_step, :, :, :]
        return self.duplicate(data=time_data, sample_period=new_sample_period, **kwargs)

    def get_indices_for_state_variables(self, sv_labels):
        return self._get_index_of_label(sv_labels, self.get_dimension_name(1))

    def get_state_variables_by_index(self, sv_indices, **kwargs):
        return self.slice_data_across_dimension_by_index(sv_indices, 1, **kwargs)

    def get_state_variables_by_label(self, sv_labels, **kwargs):
        return self.slice_data_across_dimension_by_label(sv_labels, 1, **kwargs)

    def get_state_variables_by_slice(self, slice_arg, **kwargs):
        return self.slice_data_across_dimension_by_slice(slice_arg, 1, **kwargs)

    def get_state_variables(self, sv_inputs, **kwargs):
        return self.slice_data_across_dimension(sv_inputs, 1, **kwargs)

    def get_indices_for_labels(self, region_labels):
        return self._get_index_of_label(region_labels, self.get_dimension_name(2))

    def get_subspace_by_index(self, list_of_index, **kwargs):
        return self.slice_data_across_dimension_by_index(list_of_index, 2, **kwargs)

    def get_subspace_by_label(self, list_of_labels, **kwargs):
        return self.slice_data_across_dimension_by_label(list_of_labels, 2, **kwargs)

    def get_subspace_by_slice(self, slice_arg, **kwargs):
        return self.slice_data_across_dimension_by_slice(slice_arg, 2, **kwargs)

    def get_subspace(self, subspace_inputs, **kwargs):
        return self.slice_data_across_dimension(subspace_inputs, 2, **kwargs)

    def get_modes_by_index(self, list_of_index, **kwargs):
        return self.slice_data_across_dimension_by_index(list_of_index, 3, **kwargs)

    def get_modes_by_label(self, list_of_labels, **kwargs):
        return self.slice_data_across_dimension_by_label(list_of_labels, 3, **kwargs)

    def get_modes_by_slice(self, slice_arg, **kwargs):
        return self.slice_data_across_dimension_by_slice(slice_arg, 3, **kwargs)

    def get_modes(self, modes_inputs, **kwargs):
        return self.slice_data_across_dimension(modes_inputs, 2, **kwargs)

    def get_sample_window(self, index_start, index_end, **kwargs):
        subsample_data = self.data[:, :, :, index_start:index_end]
        if subsample_data.ndim == 3:
            subsample_data = numpy.expand_dims(subsample_data, 3)
        return self.duplicate(data=subsample_data, **kwargs)

    # TODO: find out with this is not working!:

    def __getattr__(self, attr_name):
        # We are here because attr_name is not an attribute of TimeSeries...
        # TODO: find out if this part works, given that it is not really necessary
        # Now try to behave as if this was a getitem call:
        for i_dim in range(1, 4):
            if self.get_dimension_name(i_dim) in self.labels_dimensions.keys() and \
                attr_name in self.get_dimension_labels(i_dim):
                return self.slice_data_across_dimension_by_label(attr_name, i_dim)
        raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def __setattr__(self, attr_name, value):
        try:
            super(TimeSeries, self).__setattr__(attr_name, value)
            return
        except:
            # We are here because attr_name is not an attribute of TimeSeries...
            # TODO: find out if this part works, given that it is not really necessary
            # Now try to behave as if this was a setitem call:
            slice_list = [slice(None)]  # for first dimension index, i.e., time
            for i_dim in range(1, 4):
                if self.get_dimension_name(i_dim) in self.labels_dimensions.keys() and \
                        attr_name in self.get_dimension_labels(i_dim):
                    slice_list.append(attr_name)
                    self[tuple(slice_list)] = value
                    return
                else:
                    slice_list.append(slice(None))
            raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def swapaxes(self, ax1, ax2):
        labels_ordering = list(self.labels_ordering)
        labels_ordering[ax1] = self.labels_ordering[ax2]
        labels_ordering[ax2] = self.labels_ordering[ax1]
        return self.duplicate(data=numpy.swapaxes(self.data, ax1, ax2), labels_ordering=labels_ordering)


class TimeSeriesBrain(TimeSeries):

    def get_source(self):
        if self.labels_ordering[1] not in self.labels_dimensions.keys():
            logger.error("No state variables are defined for this instance!")
            raise ValueError
        if PossibleVariables.SOURCE.value in self.variables_labels:
            return self.get_state_variables_by_label(PossibleVariables.SOURCE.value)

    @property
    def brain_labels(self):
        return self.space_labels


# TODO: Slicing should also slice Connectivity, Surface, Volume, Sensors etc accordingly...


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
            logger.warn("Creating %s with sensors of type %s!" % (self.__class__.__name__, self.sensors.__class__.__name__))

    @property
    def EEGsensor_labels(self):
        return self.space_labels

    def to_tvb_instance(self, **kwargs):
        return super(TimeSeriesEEG, self).to_tvb_instance(TimeSeriesEEGTVB, **kwargs)


class TimeSeriesSEEG(TimeSeriesSensors, TimeSeriesSEEGTVB):
    title = Attr(str, default="SEEG Time Series")

    def configure(self):
        super(TimeSeriesSensors, self).configure()
        if isinstance(self.sensors, Sensors) and not isinstance(self.sensors, SensorsInternal):
            logger.warn("Creating %s with sensors of type %s!" % (self.__class__.__name__, self.sensors.__class__.__name__))

    @property
    def SEEGsensor_labels(self):
        return self.space_labels

    def to_tvb_instance(self, **kwargs):
        return super(TimeSeriesSEEG, self).to_tvb_instance(TimeSeriesSEEGTVB, **kwargs)


class TimeSeriesMEG(TimeSeriesSensors, TimeSeriesMEGTVB):
    title = Attr(str, default="MEG Time Series")

    def configure(self):
        super(TimeSeriesSensors, self).configure()
        if isinstance(self.sensors, Sensors) and not isinstance(self.sensors, SensorsMEG):
            logger.warn("Creating %s with sensors of type %s!" % (self.__class__.__name__, self.sensors.__class__.__name__))

    @property
    def MEGsensor_labels(self):
        return self.space_labels

    def to_tvb_instance(self, **kwargs):
        return super(TimeSeriesMEG, self).to_tvb_instance(TimeSeriesMEGTVB, **kwargs)


TimeSeriesDict = {TimeSeries.__name__: TimeSeries,
                  TimeSeriesRegion.__name__: TimeSeriesRegion,
                  TimeSeriesVolume.__name__: TimeSeriesVolume,
                  TimeSeriesSurface.__name__: TimeSeriesSurface,
                  TimeSeriesEEG.__name__: TimeSeriesEEG,
                  TimeSeriesMEG.__name__: TimeSeriesMEG,
                  TimeSeriesSEEG.__name__: TimeSeriesSEEG}

if __name__ == "__main__":
    kwargs = {"data": numpy.ones((4, 2, 10, 1)), "start_time": 0.0,
              "labels_dimensions": {LABELS_ORDERING[1]: ["x", "y"]}}
    ts = TimeSeriesRegion(**kwargs)
    tsy = ts.y
    print(tsy.squeezed)
