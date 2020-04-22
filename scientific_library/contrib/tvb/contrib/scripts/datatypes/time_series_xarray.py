# -*- coding: utf-8 -*-

from six import string_types
from copy import deepcopy
import numpy as np
import xarray as xr

from tvb.basic.logger.builder import get_logger
from tvb.contrib.scripts.datatypes.time_series import TimeSeries as TimeSeriesTVB
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesRegion as TimeSeriesRegionTVB
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesSurface as TimeSeriesSurfaceTVB
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesVolume as TimeSeriesVolumeTVB
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesSensors as TimeSeriesSensorsTVB
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesEEG as TimeSeriesEEGTVB
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesSEEG as TimeSeriesSEEGTVB
from tvb.contrib.scripts.datatypes.time_series import TimeSeriesMEG as TimeSeriesMEGTVB
from tvb.contrib.scripts.datatypes.time_series import prepare_4d
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, is_integer

from tvb.basic.neotraits.api import HasTraits, Attr, Float, List, narray_summary_info
from tvb.datatypes import sensors, surfaces, volumes, region_mapping, connectivity


logger = get_logger(__name__)


def coords_to_dict(coords):
    d = {}
    for key, val in zip(list(coords.keys()),
                        list([value.values for value in coords.values()])):
        d[key] = val
    return d


class TimeSeries(HasTraits):
    """
    Base time-series dataType.
    """

    _data = xr.DataArray([])

    _default_labels_ordering = List(
        default=("Time", "State Variable", "Space", "Mode"),
        label="Dimension Names",
        doc="""List of strings representing names of each data dimension""")

    title = Attr(str, default="Time Series")

    start_time = Float(label="Start Time")

    sample_period = Float(label="Sample period", default=1.0)

    # Specify the measure unit for sample period (e.g sec, msec, usec, ...)
    sample_period_unit = Attr(
        field_type=str,
        label="Sample Period Measure Unit",
        default="ms")

    @property
    def data(self):
        # Return numpy array
        return self._data.values

    @property
    def name(self):
        return self._data.name

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def nr_dimensions(self):
        return self._data.ndim

    @property
    def number_of_dimensions(self):
        return self._data.ndim

    def _return_shape_of_dim(self, dim):
        try:
            return self._data.shape[dim]
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
    def time(self):
        return self._data.coords[self._data.dims[0]].values

    @property
    def end_time(self):
        try:
            return self.time[-1]
        except:
            return None

    @property
    def duration(self):
        try:
            return self.end_time - self.start_time
        except:
            return None

    @property
    def sample_rate(self):
        try:
            return 1.0 / self.sample_period
        except:
            return None

    # xarrays have a attrs dict with useful attributes

    @property
    def time_unit(self):
        return self.sample_period_unit

    @property
    def dims(self):
        return self._data.dims

    @property
    def coords(self):
        return self._data.coords

    @property
    def labels_ordering(self):
        return list(self._data.dims)

    @property
    def labels_dimensions(self):
        return coords_to_dict(self._data.coords)

    @property
    def space_labels(self):
        return np.array(self._data.coords.get(self.labels_ordering[2], []))

    @property
    def variables_labels(self):
        return np.array(self._data.coords.get(self.labels_ordering[1], []))

    def squeeze(self):
        return self._data.squeeze()

    @property
    def squeezed(self):
        return self._data.values.squeeze()

    @property
    def flattened(self):
        return self._data.value.flatten()

    def __setattr__(self, name, value):
        if name == "data":
            setattr(self._data, "values", value)
        elif name == "labels_ordering":
            setattr(self._data, "dims", value)
        elif name == "labels_dimensions":
            setattr(self._data, "coords", value)
        elif name == "time":
            self._data.coords[self._data.dims[0]] = value
        else:
            super(TimeSeries, self).__setattr__(name, value)

    def from_xarray_DataArray(self, xarr, **kwargs):
        # ...or as args
        # including a xr.DataArray or None
        data = kwargs.pop("data", xarr.values)
        dims = kwargs.pop("dims", kwargs.pop("labels_ordering", xarr.dims))
        coords = kwargs.pop("coords", kwargs.pop("labels_dimensions", coords_to_dict(xarr.coords)))
        attrs = kwargs.pop("attrs", None)
        time = kwargs.pop("time", coords.pop(dims[0], None))
        if time is not None:
            kwargs['start_time'] = kwargs.pop('start_time', time[0])
            kwargs['sample_period'] = kwargs.pop('sample_period', np.diff(time).mean())
            coords[dims[0]] = time
        else:
            kwargs['start_time'] = kwargs.pop('start_time', 0.0)
        if isinstance(xarr.name, string_types) and len(xarr.name) > 0:
            title = xarr.name
        else:
            title = self.__class__.title.default
        kwargs['title'] = kwargs.pop('title', kwargs.pop('name', title))
        self._data = xr.DataArray(data=data, dims=dims, coords=coords, attrs=attrs, name=str(kwargs['title']))
        super(TimeSeries, self).__init__(**kwargs)

    def from_TVB_time_series(self, ts, **kwargs):
        labels_ordering = kwargs.pop("labels_ordering", kwargs.pop("dims", ts.labels_ordering))
        labels_dimensions = kwargs.pop("labels_dimensions", kwargs.pop("coords", ts.labels_dimensions))
        time = kwargs.pop("time", ts.time)
        labels_dimensions[labels_ordering[0]] = time
        for label, dimensions in labels_dimensions.items():
            id = labels_ordering.index(label)
            if ts.shape[id] != len(dimensions):
                labels_dimensions[label] = np.arange(ts.shape[id]).astype("i")
        self.start_time = kwargs.pop('start_time',
                                     getattr(ts, "start_time", ts.time[0]))
        self.sample_period = kwargs.pop('sample_period',
                                        getattr(ts, "sample_period", np.diff(ts.time).mean()))
        self.sample_period_unit = kwargs.pop('sample_period_unit',
                                             getattr(ts, "sample_period_unit",
                                                     self.__class__.sample_period_unit.default))
        self.title = kwargs.pop("title", kwargs.pop("name", ts.title))
        self._data = xr.DataArray(ts.data,
                                  dims=labels_ordering,
                                  coords=labels_dimensions,
                                  name=str(self.title), attrs=kwargs.pop("attrs", None))
        super(TimeSeries, self).__init__(**kwargs)

    def from_numpy(self, data, **kwargs):
        # We have to infer time and labels inputs from kwargs
        data = prepare_4d(data, logger)
        time, self.start_time, end_time, self.sample_period, kwargs = self._configure_input_time(data, **kwargs)
        labels_ordering, labels_dimensions, kwargs = self._configure_input_labels(**kwargs)
        if time is not None:
            if labels_dimensions is None:
                labels_dimensions = {}
            labels_dimensions[labels_ordering[0]] = time
        self.sample_period_unit = kwargs.pop('sample_period_unit', self.__class__.sample_period_unit.default)
        self.title = kwargs.pop('title', kwargs.pop("name", self.__class__.title.default))
        self._data = xr.DataArray(data, dims=labels_ordering, coords=labels_dimensions,
                                  name=str(self.title), attrs=kwargs.pop("attrs", None))
        super(TimeSeries, self).__init__(**kwargs)

    def _configure_input_time(self, data, **kwargs):
        # Method to initialise time attributes
        # It gives priority to an input time vector, if any.
        # Subsequently, it considers start_time and sample_period kwargs
        sample_period = kwargs.pop("sample_period", None)
        start_time = kwargs.pop("start_time", None)
        time = kwargs.pop("time", None)
        time_length = data.shape[0]
        if time_length > 0:
            if time is None:
                if start_time is None:
                    start_time = 0.0
                if sample_period is None:
                    sample_period = 1.0
                end_time = start_time + (time_length - 1) * sample_period
                time = np.arange(start_time, end_time + sample_period, sample_period)
                return time, start_time, end_time, sample_period, kwargs
            else:
                assert time_length == len(time)
                start_time = time[0]
                end_time = time[-1]
                if len(time) > 1:
                    sample_period = np.mean(np.diff(time))
                    assert np.abs(end_time - start_time - (time_length - 1) * sample_period) < 1e-6
                else:
                    sample_period = None
                return time, start_time, end_time, sample_period, kwargs
        else:
            # Empty data
            return None, start_time, None, sample_period, kwargs

    def _configure_input_labels(self, **kwargs):
        # Method to initialise label attributes
        # It gives priority to labels_ordering,
        # i.e., labels_dimensions dict cannot ovewrite it
        # and should agree with it
        labels_ordering = list(kwargs.pop("labels_ordering", self._default_labels_ordering))
        labels_dimensions = kwargs.pop("labels_dimensions", None)
        if isinstance(labels_dimensions, dict):
            assert [key in labels_ordering for key in labels_dimensions.keys()]
        return labels_ordering, labels_dimensions, kwargs

    def _configure_time(self):
        assert self.time[0] == self.start_time
        assert self.time[-1] == self.end_time
        if self.time_length > 1:
            assert np.abs(self.sample_period - (self.end_time - self.start_time) / (self.time_length - 1)) < 1e-6

    def _configure_labels(self):
        for i_dim in range(1, self.nr_dimensions):
            dim_label = self.labels_ordering[i_dim]
            val = self.labels_dimensions.get(dim_label, None)
            if val is not None:
                assert len(val) == self.shape[i_dim]
            else:
                # We set by default integer labels if no input labels are provided by the user
                self._data.coords[self._data.dims[i_dim]] = np.arange(0, self.shape[i_dim])

    def configure(self):
        # To be always used when a new object is created
        # to check that everything is set correctly
        if self._data.name is None or len(self._data.name) == 0:
            self._data.name = self.title
        else:
            self.title = self._data.name
        super(TimeSeries, self).configure()
        try:
            time_length = self.time_length
        except:
            pass  # It is ok if data is empty
        if time_length is not None and time_length > 0:
            self._configure_time()
            self._configure_labels()

    def __init__(self, data=None, **kwargs):
        if isinstance(data, (list, tuple)):
            self.from_numpy(np.array(data), **kwargs)
        elif isinstance(data, np.ndarray):
            self.from_numpy(data, **kwargs)
        elif isinstance(data, self.__class__):
            attributes = data.__dict__.items()
            attributes.update(**kwargs)
            for attr, val in attributes.items():
                setattr(self, attr, val)
        elif isinstance(data, TimeSeriesTVB):
            self.from_TVB_time_series(data, **kwargs)
        elif isinstance(data, xr.DataArray):
            self.from_xarray_DataArray(data, **kwargs)
        else:
            # Assuming data is an input xr.DataArray() can handle,
            if isinstance(data, dict):
                # ...either as kwargs
                self._data = xr.DataArray(**data, attrs=kwargs.pop("attrs", None))
            else:
                # ...or as args
                # including a xr.DataArray or None
                self._data = xr.DataArray(data,
                                          dims=kwargs.pop("dims", kwargs.pop("labels_ordering", None)),
                                          coords=kwargs.pop("coords", kwargs.pop("labels_dimensions", None)),
                                          attrs=kwargs.pop("attrs", None))
            super(TimeSeries, self).__init__(**kwargs)
        self.configure()

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
            "Length": self.time_length
        }
        summary.update(narray_summary_info(self.data))
        return summary

    def duplicate(self, **kwargs):
        # Since all labels are internal to xarray,
        # it suffices to pass a new (e.g., sliced) xarray _data as kwarg
        # for all labels to be set correctly (and confirmed by the call to configure(),
        # whereas any other attributes of TimeSeries will be copied
        time = kwargs.pop("time", None)
        if time is not None:
            kwargs['start_time'] = kwargs.pop('start_time', time[0])
            kwargs['sample_period'] = kwargs.pop('sample_period', np.diff(time).mean())
        else:
            kwargs['start_time'] = kwargs.pop('start_time', self.start_time)
            kwargs['sample_period'] = kwargs.pop('sample_period', self.sample_period)
        kwargs['sample_period_unit'] = kwargs.pop('sample_period_unit', self.sample_period_unit)
        kwargs['title'] = kwargs.pop('title', self.title)
        output = self.__class__()
        output.from_xarray_DataArray(kwargs.pop("_data", self._data), **kwargs)
        return output

    def to_tvb_instance(self, datatype=TimeSeriesTVB, **kwargs):
        return datatype().from_xarray_DataArray(self._data, **kwargs)

    def _assert_index(self, index):
        if (index < 0 or index >= self.number_of_dimensions):
            raise IndexError("index %d is not within the dimensions [0, %d] of this TimeSeries data:\n%s"
                             % (index, self.number_of_dimensions, str(self)))
        return index

    def get_dimension_index(self, dim_name_or_index):
        if is_integer(dim_name_or_index):
            return self._assert_index(dim_name_or_index)
        elif isinstance(dim_name_or_index, string_types):
            return self._assert_index(self._data.dims.index(dim_name_or_index))
        else:
            raise ValueError("dim_name_or_index is neither a string nor an integer!")

    def get_dimension_name(self, dim_index):
        try:
            return self._data.dims[dim_index]
        except IndexError:
            logger.error("Cannot access index %d of labels ordering: %s!" %
                              (int(dim_index), str(self._data.dims)))

    def get_dimension_labels(self, dimension_label_or_index):
        if not isinstance(dimension_label_or_index, string_types):
            dimension_label_or_index = self.get_dimension_name(dimension_label_or_index)
        try:
            return self._data.coords[dimension_label_or_index]
        except KeyError:
            logger.error("There are no %s labels defined for this instance: %s",
                              (dimension_label_or_index, str(self._data.coords)))
            raise

    def update_dimension_names(self, dim_names, dim_indices=None):
        dim_names = ensure_list(dim_names)
        if dim_indices is None:
            dim_indices = list(range(len(dim_names)))
        else:
            dim_indices = ensure_list(dim_indices)
        labels_ordering = list(self._data.dims)
        for dim_name, dim_index in zip(dim_names, dim_indices):
            labels_ordering[dim_index] = dim_name
            try:
                old_dim_name = self._data.dims[dim_index]
                dim_labels = list(self._data.coords[old_dim_name])
                del self._data.coords[old_dim_name]
                self._data.coords[dim_name] = dim_labels
            except:
                pass
        self._data.dims = labels_ordering

    def _check_indices(self, indices, dimension):
        dim_index = self.get_dimension_index(dimension)
        for index in ensure_list(indices):
            if index < 0 or index > self._data.shape[dim_index]:
                logger.error("Some of the given indices are out of %s range: [0, %s]",
                                  (self.get_dimension_name(dim_index), self._data.shape[dim_index]))
                raise IndexError

    def _check_time_indices(self, list_of_index):
        self._check_indices(list_of_index, 0)

    def _check_variables_indices(self, list_of_index):
        self._check_indices(list_of_index, 1)

    def _check_space_indices(self, list_of_index):
        self._check_indices(list_of_index, 2)

    def _check_modes_indices(self, list_of_index):
        self._check_indices(list_of_index, 2)

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
            # NOTE!: In case of a string slice, we consider stop included!
            slice_stop = self._get_index_for_slice_label(slice_stop, slice_idx) + 1

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

    def _get_item(self, slice_tuple, **kwargs):
        slice_tuple = self._assert_array_indices(slice_tuple)
        try:
            # For integer indices
            return self.duplicate(_data=self._data[slice_tuple], **kwargs)
        except:
            try:
                # For label indices
                # xrarray.DataArray.loc slices along labels
                # Assuming that all dimensions without input labels
                # are configured with labels of integer indices=
                return self.duplicate(_data=self._data.loc[slice_tuple], **kwargs)
            except:
                # Still, for a conflicting mixture that has to be resolved
                return self.duplicate(_data=self._data[self._process_slices(slice_tuple)], **kwargs)

    # Return a TimeSeries object
    def __getitem__(self, slice_tuple):
        return self._get_item(slice_tuple)

    def __setitem__(self, slice_tuple, values):
        slice_tuple = self._assert_array_indices(slice_tuple)
        # Mind that xarray can handle setting values both from a numpy array and/or another xarray
        if isinstance(values, self.__class__):
            values = np.array(values.data)
        try:
            # For integer indices
            self._data[slice_tuple] = values
        except:
            try:
                # For label indices
                # xrarray.DataArray.loc slices along labels
                # Assuming that all dimensions without input labels
                # are configured with labels of integer indices
                self._data.loc[slice_tuple] = values
            except:
                # Still, for a conflicting mixture that has to be resolved
                self._data[self._process_slices(slice_tuple)] = values

    def slice_data_across_dimension_by_index(self, indices, dimension, **kwargs):
        dim_index = self.get_dimension_index(dimension)
        indices = ensure_list(indices)
        slices = [slice(None)] * self._data.ndim
        slices[dim_index] = indices
        return self.duplicate(_data=self._data[tuple(slices)], **kwargs)

    def slice_data_across_dimension_by_label(self, labels, dimension, **kwargs):
        dim_index = self.get_dimension_index(dimension)
        labels = ensure_list(labels)
        slices = [slice(None)] * self._data.ndim
        slices[dim_index] = labels
        return self.duplicate(_data=self._data.loc[tuple(slices)], **kwargs)

    def slice_data_across_dimension_by_slice(self, slice_arg, dimension, **kwargs):
        dim_index = self.get_dimension_index(dimension)
        slices = [slice(None)] * self._data.ndims
        slices[dim_index] = slice_arg
        slices = tuple(slices)
        return self._get_item(slices)

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
        subtime_data = self._data[index_start:index_end, :, :, :]
        if subtime_data.ndim == 3:
            subtime_data = np.expand_dims(subtime_data, 0)
        return self.duplicate(_data=subtime_data, time=self.time[index_start:index_end], **kwargs)

    def get_time_window_by_units(self, unit_start, unit_end, **kwargs):
        end_time = self.end_time
        if unit_start < self.start_time or unit_end > end_time:
            logger.error("The time units are outside time series interval: [%s, %s]" %
                              (self.start_time, end_time))
            raise IndexError
        index_start = self._get_index_for_time_unit(unit_start)
        index_end = self._get_index_for_time_unit(unit_end)
        return self.get_time_window(index_start, index_end, **kwargs)

    def get_times(self, list_of_times, **kwargs):
        return self.slice_data_across_dimension(list_of_times, 0, **kwargs)

    def decimate_time(self, new_sample_period, **kwargs):
        if new_sample_period % self.sample_period != 0:
            logger.error("Cannot decimate time if new time step is not a multiple of the old time step")
            raise ValueError
        index_step = int(new_sample_period / self.sample_period)
        return self.duplicate(_data=self._data[::index_step, :, :, :],
                              sample_period=new_sample_period, **kwargs)

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

    def get_sample_window(self, index_start, index_end, **kwargs):
        return self.duplicate(_data=self._data[:, :, :, index_start:index_end], **kwargs)

    # def __getattr__(self, attr_name):
    #     # We are here because attr_name is not an attribute of TimeSeries...
    #     try:
    #         # First try to see if this is a xarray.DataArray attribute
    #         getattr(self._data, attr_name)
    #     except:
    #         # TODO: find out if this part works, given that it is not really necessary
    #         # Now try to behave as if this was a getitem call:
    #         slice_list = [slice(None)]  # for first dimension index, i.e., time
    #         for i_dim in range(1, self.nr_dimensions):
    #             if self.labels_ordering[i_dim] in self.labels_dimensions.keys() and \
    #                     attr_name in self.labels_dimensions[self.labels_ordering[i_dim]]:
    #                 slice_list.append(attr_name)
    #                 return self._data.loc[tuple(slice_list)]
    #             else:
    #                 slice_list.append(slice(None))
    #         raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr_name))
    #
    # def __setattr__(self, attr_name, value):
    #     # We are here because attr_name is not an attribute of TimeSeries...
    #     try:
    #         # First try to see if this is a xarray.DataArray attribute
    #         getattr(self._data, attr_name, value)
    #     except:
    #         # TODO: find out if this part works, given that it is not really necessary
    #         # Now try to behave as if this was a setitem call:
    #         slice_list = [slice(None)]  # for first dimension index, i.e., time
    #         for i_dim in range(1, self.nr_dimensions):
    #             if self.labels_ordering[i_dim] in self.labels_dimensions.keys() and \
    #                     attr_name in self.labels_dimensions[self.labels_ordering[i_dim]]:
    #                 slice_list.append(attr_name)
    #                 self._data.loc[tuple(slice_list)] = value
    #                 return
    #             else:
    #                 slice_list.append(slice(None))
    #         raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def swapaxes(self, ax1, ax2):
        dims = list(self._data.dims)
        dims[ax1] = self._data.dims[ax2]
        dims[ax2] = self._data.dims[ax1]
        new_self = self.duplicate()
        new_self._data = self._data.transpose(*dims)
        new_self.configure()
        return new_self

    def plot(self, time=None, data=None, y=None, hue=None, col=None, row=None, figname=None, plotter=None, **kwargs):
        if data is None:
            data = self._data
        if time is None:
            time = data.dims[0]
        if figname is None:
            figname = kwargs.pop("figname", "%s" % data.name)
        for dim_name, dim in zip(["y", "hue", "col", "row"],
                                 [y, hue, col, row]):
            if dim is not None:
                id = data.dims.index(dim)
                if data.shape[id] > 1:
                    kwargs[dim_name] = dim
        output = data.plot(x=time, **kwargs)
        # TODO: Something better than this temporary hack for base_plotter functionality
        if plotter is not None:
            plotter.base._save_figure(figure_name=figname)
            plotter.base._check_show()
        return output

    def _prepare_plot_args(self, **kwargs):
        plotter = kwargs.pop("plotter", None)
        labels_ordering = self.labels_ordering
        time = labels_ordering[0]
        return time, labels_ordering, plotter, kwargs

    def plot_map(self, **kwargs):
        time, labels_ordering, plotter, kwargs = \
            self._prepare_plot_args(**kwargs)
        # Usually variables
        col = kwargs.pop("col", labels_ordering[1])
        if self._data.shape[2] > 1:
            y = kwargs.pop("y", labels_ordering[2])
            row = kwargs.pop("row", labels_ordering[3])
        else:
            y = kwargs.pop("y", labels_ordering[3])
            row = kwargs.pop("row", None)
        kwargs["robust"] = kwargs.pop("robust", True)
        kwargs["cmap"] = kwargs.pop("cmap", "jet")
        figname = kwargs.pop("figname", "%s" % self.title)
        return self.plot(data=None, y=y, hue=None, col=col, row=row,
                         figname=figname, plotter=plotter, **kwargs)

    def plot_line(self, **kwargs):
        time, labels_ordering, plotter, kwargs = \
            self._prepare_plot_args(**kwargs)
        # Usually variables
        col = kwargs.pop("col", labels_ordering[1])
        if self._data.shape[3] > 1:
            hue = kwargs.pop("hue", labels_ordering[3])
            row = kwargs.pop("row", labels_ordering[2])
        else:
            hue = kwargs.pop("hue", labels_ordering[2])
            row = kwargs.pop("row", None)
        figname = kwargs.pop("figname", "%s" % self.title)
        return self.plot(data=None, y=None, hue=hue, col=col, row=row,
                         figname=figname, plotter=plotter, **kwargs)

    def plot_timeseries(self, **kwargs):
        if kwargs.pop("per_variable", False):
            outputs = []
            for var in self.labels_dimensions[self.labels_ordering[1]]:
                outputs.append(self[:, var].plot_timeseries(**kwargs))
            return outputs
        if np.any([s < 2 for s in self.shape[1:]]):
            if self.shape[1] == 1:  # only one variable
                figname = kwargs.pop("figname", "%s" % (self.title + "Time Series")) + ": " \
                          + self.labels_dimensions[self.labels_ordering[1]][0]
                kwargs["figname"] = figname
            return self.plot_line(**kwargs)
        else:
            return self.plot_raster(**kwargs)

    def plot_raster(self, **kwargs):
        figname = kwargs.pop("figname", "%s" % (self.title + "Time Series raster"))
        if kwargs.pop("per_variable", False):
            outputs = []
            for var in self.labels_dimensions[self.labels_ordering[1]]:
                outputs.append(self[:, var].plot_raster(**kwargs))
            return outputs
        time, labels_ordering, plotter, kwargs = \
            self._prepare_plot_args(**kwargs)
        col = labels_ordering[1]  # Variable
        labels_dimensions = self.labels_dimensions
        if self.shape[1] < 2:
            try:
                figname = figname + ": %s" % labels_dimensions[col][0]
            except:
                pass
        data = xr.DataArray(self._data)
        for i_var, var in enumerate(labels_dimensions[labels_ordering[1]]):
            # Remove mean
            data[:, i_var] -= data[:, i_var].mean()
            # Compute approximate range for this variable
            amplitude = 0.9 * (data[:, i_var].max() - data[:, i_var].min())
            # Add the step on y axis for this variable and for each Region's data
            for i_region in range(self.shape[2]):
                data[:, i_var, i_region] += amplitude * i_region
        # hue: Regions and/or Modes/Samples/Populations etc
        if np.all([s > 1 for s in self.shape[2:]]):
            hue = "%s - %s" % (labels_ordering[2], labels_ordering[3])
            data = data.stack({hue: (labels_ordering[2], labels_ordering[3])})
        elif self.shape[3] > 1:
            hue = labels_ordering[3]
        elif self.shape[2] > 1:
            hue = labels_ordering[2]
        kwargs["col_wrap"] = kwargs.pop("col_wrap", self.shape[1])  # All variables in columns
        return self.plot(data=data, y=None, hue=hue, col=col, row=None,
                         figname=figname, plotter=plotter, **kwargs)


# TODO: Slicing should also slice Connectivity, Surface, Volume, Sensors etc accordingly...


class TimeSeriesRegion(TimeSeries):
    """ A time-series associated with the regions of a connectivity. """

    title = Attr(str, default="Region Time Series")

    connectivity = Attr(field_type=connectivity.Connectivity)
    region_mapping_volume = Attr(field_type=region_mapping.RegionVolumeMapping, required=False)
    region_mapping = Attr(field_type=region_mapping.RegionMapping, required=False)
    _default_labels_ordering = List(of=str, default=("Time", "State Variable", "Region", "Mode"))

    def __init__(self, data=None, **kwargs):
        if not isinstance(data, TimeSeriesRegion):
            self.connectivity = kwargs.pop("connectivity")
            self.region_mapping_volume = kwargs.pop("region_mapping_volume", None)
            self.region_mapping = kwargs.pop("region_mapping", None)
        super(TimeSeriesRegion, self).__init__(data, **kwargs)

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

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesRegionTVB().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesSurface(TimeSeries):
    """ A time-series associated with a Surface. """

    title = Attr(str, default="Surface Time Series")

    surface = Attr(field_type=surfaces.CorticalSurface)
    _default_labels_ordering = List(of=str, default=("Time", "State Variable", "Vertex", "Mode"))

    def __init__(self, data=None, **kwargs):
        if not isinstance(data, TimeSeriesSurface):
            self.surface = kwargs.pop("surface")
        super(TimeSeriesSurface, self).__init__(data, **kwargs)

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesSurface, self).summary_info()
        summary.update({"Source Surface": self.surface.title})
        return summary

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesSurfaceTVB().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesVolume(TimeSeries):
    """ A time-series associated with a Volume. """

    title = Attr(str, default="Volume Time Series")

    volume = Attr(field_type=volumes.Volume)
    _default_labels_ordering = List(of=str, default=("Time", "X", "Y", "Z"))

    def __init__(self, data=None, **kwargs):
        if not isinstance(data, TimeSeriesVolume):
            self.volume = kwargs.pop("volume")
        super(TimeSeriesVolume, self).__init__(data, **kwargs)

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesVolume, self).summary_info()
        summary.update({"Source Volume": self.volume.title})
        return summary

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesVolumeTVB().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesSensors(TimeSeries):
    title = Attr(str, default="Sensor Time Series")

    def __init__(self, data=None, **kwargs):
        if not isinstance(data, TimeSeriesSensors):
            self.sensors = kwargs.pop("sensors")
        super(TimeSeriesSensors, self).__init__(data, **kwargs)

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesSensors, self).summary_info()
        summary.update({"Source Sensors": self.sensors.title})
        return summary

    def to_tvb_instance(self, datatype=TimeSeriesSensorsTVB, **kwargs):
        return datatype().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesEEG(TimeSeriesSensors):
    """ A time series associated with a set of EEG sensors. """

    title = Attr(str, default="EEG Time Series")

    sensors = Attr(field_type=sensors.SensorsEEG)
    _default_labels_ordering = List(of=str, default=("Time", "1", "EEG Sensor", "1"))

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesEEGTVB().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesSEEG(TimeSeriesSensors):
    """ A time series associated with a set of Internal sensors. """

    title = Attr(str, default="SEEG Time Series")

    sensors = Attr(field_type=sensors.SensorsInternal)
    _default_labels_ordering = List(of=str, default=("Time", "1", "sEEG Sensor", "1"))

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesSEEGTVB().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesMEG(TimeSeriesSensors):
    """ A time series associated with a set of MEG sensors. """

    title = Attr(str, default="MEG Time Series")

    sensors = Attr(field_type=sensors.SensorsMEG)
    _default_labels_ordering = List(of=str, default=("Time", "1", "MEG Sensor", "1"))

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesMEGTVB().from_xarray_DataArray(self._data, **kwargs)


TimeSeriesDict = {TimeSeries.__name__: TimeSeries,
                  TimeSeriesRegion.__name__: TimeSeriesRegion,
                  TimeSeriesVolume.__name__: TimeSeriesVolume,
                  TimeSeriesSurface.__name__: TimeSeriesSurface,
                  TimeSeriesEEG.__name__: TimeSeriesEEG,
                  TimeSeriesMEG.__name__: TimeSeriesMEG,
                  TimeSeriesSEEG.__name__: TimeSeriesSEEG}
