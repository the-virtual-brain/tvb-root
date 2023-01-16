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
from tvb.contrib.scripts.service.head_service import HeadService
from tvb.contrib.scripts.utils.data_structures_utils import ensure_list, is_integer

from tvb.basic.neotraits.api import HasTraits, Attr, Float, List, narray_summary_info
from tvb.datatypes import sensors, surfaces, volumes, region_mapping, connectivity
from tvb.simulator.plot.base_plotter import pyplot


logger = get_logger(__name__)


MAX_LBLS_IN_LEGEND = 10


def assert_coords_dict(coords):
    if isinstance(coords, dict):
        val_fun = lambda val: val
    else:
        val_fun = lambda val: val.values
    d = {}
    for key, val in zip(list(coords.keys()),
                        list([val_fun(value) for value in coords.values()])):
        d[key] = list(val)
    return d


def save_show_figure(plotter_config, figure_name=None, fig=None):
    import os
    from matplotlib import pyplot
    if plotter_config.SAVE_FLAG:
        if figure_name is None:
            if fig is None:
                fig = pyplog.gcf()
            figure_name = fig.get_label()
        figure_name = figure_name.replace(": ", "_").replace(" ", "_").replace("\t", "_").replace(",", "")
        figure_name = figure_name[:np.min([100, len(figure_name)])] + '.' + plotter_config.FIG_FORMAT
        figure_dir = plotter_config.FOLDER_FIGURES
        if not (os.path.isdir(figure_dir)):
            os.mkdir(figure_dir)
        pyplot.savefig(os.path.join(figure_dir, figure_name))
    if plotter_config.SHOW_FLAG:
        # mp.use('TkAgg')
        pyplot.ion()
        pyplot.show()
    else:
        # mp.use('Agg')
        pyplot.ioff()
        pyplot.close()


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
        return assert_coords_dict(self._data.coords)

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

    def from_xarray_DataArray(self, xarr, **kwargs):
        # ...or as args
        # including a xr.DataArray or None
        data = kwargs.pop("data", xarr.values)
        dims = kwargs.pop("dims", kwargs.pop("labels_ordering", xarr.dims))
        coords = kwargs.pop("coords", kwargs.pop("labels_dimensions", assert_coords_dict(xarr.coords)))
        attrs = kwargs.pop("attrs", None)
        time = kwargs.pop("time", coords.pop(dims[0], None))
        if time is not None and len(time) > 0:
            kwargs['start_time'] = kwargs.pop('start_time', float(time[0]))
            if len(time) > 1:
                kwargs['sample_period'] = kwargs.pop('sample_period', float(np.diff(time).mean()))
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
                                     getattr(ts, "start_time", float(ts.time[0])))
        self.sample_period = kwargs.pop('sample_period',
                                        getattr(ts, "sample_period", float(np.diff(ts.time).mean())))
        self.sample_period_unit = kwargs.pop('sample_period_unit',
                                             getattr(ts, "sample_period_unit",
                                                     self.__class__.sample_period_unit.default))
        self.title = kwargs.pop("title", kwargs.pop("name", ts.title))
        self._data = xr.DataArray(ts.data,
                                  dims=labels_ordering,
                                  coords=labels_dimensions,
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
            if time is None or len(time) == 0:
                if start_time is None:
                    start_time = 0.0
                if sample_period is None:
                    sample_period = 1.0
                end_time = start_time + (time_length - 1) * sample_period
                time = np.arange(start_time, end_time + sample_period, sample_period)
                return time, start_time, end_time, sample_period, kwargs
            else:
                assert time_length == len(time)
                start_time = float(time[0])
                end_time = float(time[-1])
                if len(time) > 1:
                    sample_period = float(np.mean(np.diff(time)))
                    assert np.abs(end_time - start_time - (time_length - 1) * sample_period) < 1e-6
                else:
                    sample_period = None
                return time, start_time, end_time, sample_period, kwargs
        else:
            if start_time is None:
                start_time = self.__class__.start_time.default
            if sample_period is None:
                sample_period = self.__class__.sample_period.default
            # Empty data
            return None, start_time, None, sample_period, kwargs

    def _configure_input_labels(self, **kwargs):
        # Method to initialise label attributes
        # It gives priority to labels_ordering,
        # i.e., labels_dimensions dict cannot ovewrite it
        # and should agree with it
        labels_ordering = list(kwargs.pop("dims", kwargs.pop("labels_ordering",
                                                             self._default_labels_ordering)))
        labels_dimensions = kwargs.pop("coords", kwargs.pop("labels_dimensions",
                                                            getattr(self, "labels_dimensions", None)))
        if labels_dimensions is not None:
            labels_dimensions = assert_coords_dict(labels_dimensions)
        if isinstance(labels_dimensions, dict) and len(labels_dimensions):
            assert [key in labels_ordering for key in labels_dimensions.keys()]
        return labels_ordering, labels_dimensions, kwargs

    def from_numpy(self, data, **kwargs):
        # We have to infer time and labels inputs from kwargs
        data = prepare_4d(data, logger)
        time, self.start_time, end_time, self.sample_period, kwargs = self._configure_input_time(data, **kwargs)
        labels_ordering, labels_dimensions, kwargs = self._configure_input_labels(**kwargs)
        if time is not None and len(time) > 0:
            if labels_dimensions is None:
                labels_dimensions = {}
            labels_dimensions[labels_ordering[0]] = time
        self.sample_period_unit = kwargs.pop('sample_period_unit', self.__class__.sample_period_unit.default)
        self.title = kwargs.pop('title', kwargs.pop("name", self.__class__.title.default))
        self._data = xr.DataArray(data, dims=list(labels_ordering), coords=labels_dimensions,
                                  name=str(self.title), attrs=kwargs.pop("attrs", None))
        super(TimeSeries, self).__init__(**kwargs)

    def _configure_time(self):
        assert self.time[0] == self.start_time
        assert self.time[-1] == self.end_time
        if self.time_length > 1:
            assert np.abs(self.sample_period - (self.end_time - self.start_time) / (self.time_length - 1)) < 1e-6

    def _configure_labels(self):
        labels_dimensions = self.labels_dimensions
        for i_dim in range(1, self.nr_dimensions):
            dim_label = self.get_dimension_name(i_dim)
            val = labels_dimensions.get(dim_label, None)
            if val is not None:
                assert len(val) == self.shape[i_dim]

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
            time_length = None
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
            attributes = data.__dict__
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
            elif data is not None:
                # ...or as args
                # including a xr.DataArray or None
                self._data = xr.DataArray(data,
                                          dims=kwargs.pop("dims", kwargs.pop("labels_ordering", None)),
                                          coords=kwargs.pop("coords", kwargs.pop("labels_dimensions", None)),
                                          attrs=kwargs.pop("attrs", None))
            super(TimeSeries, self).__init__(**kwargs)

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

    def _duplicate(self, **kwargs):
        # Since all labels are internal to xarray,
        # it suffices to pass a new (e.g., sliced) xarray _data as kwarg
        # for all labels to be set correctly (and confirmed by the call to configure(),
        # whereas any other attributes of TimeSeries will be copied
        _data = kwargs.pop("_data", None)
        if isinstance(_data, xr.DataArray):
            # If we have a DataArray input, we should set defaults through it
            _labels_ordering = list(_data.dims)
            _labels_dimensions = dict(assert_coords_dict(_data.coords))
            if _data.name is not None and len(_data.name) > 0:
                _title = _data.name
            else:
                _title = self.title
        else:
            # Otherwise, we should set defaults through self
            _labels_ordering = list(self._data.dims)
            _labels_dimensions = dict(assert_coords_dict(self._data.coords))
            _title = self.title
            # Also, in this case we have to generate a new output DataArray...
            data = kwargs.pop("data", None)
            if data is None:
                # ...either from self._data
                _data = self._data.copy()
            else:
                # ...or from a potential numpy/list/tuple input
                _data = xr.DataArray(np.array(data))
        # Now set the rest of the properties...
        kwargs["labels_ordering"] = kwargs.pop("dims", kwargs.pop("labels_ordering", _labels_ordering))
        kwargs["labels_dimensions"] = kwargs.pop("labels_dimensions",
                                                 assert_coords_dict(kwargs.pop("coords", _labels_dimensions)))
        # ...with special care for time related ones:
        time = kwargs["labels_dimensions"].get(kwargs["labels_ordering"][0], None)
        time = kwargs.pop("time", time)
        if time is not None and len(time) > 0:
            kwargs['start_time'] = kwargs.pop('start_time', float(time[0]))
            if len(time) > 1:
                kwargs['sample_period'] = kwargs.pop('sample_period', np.diff(time).mean())
            else:
                kwargs['sample_period'] = kwargs.pop('sample_period', self.sample_period)
        else:
            kwargs['start_time'] = kwargs.pop('start_time', self.start_time)
            kwargs['sample_period'] = kwargs.pop('sample_period', self.sample_period)
        kwargs['sample_period_unit'] = kwargs.pop('sample_period_unit', self.sample_period_unit)
        kwargs['title'] = kwargs.pop('title', self.title)
        return _data, kwargs

    def duplicate(self, **kwargs):
        _data, kwargs = self._duplicate(**kwargs)
        duplicate = self.__class__()
        duplicate.from_xarray_DataArray(_data.copy(deep=True), **kwargs.copy())
        return duplicate

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
            return self.labels_dimensions[dimension_label_or_index]
        except KeyError:
            logger.warning("There are no %s labels defined for this instance: %s",
                           (dimension_label_or_index, str(self._data.coords)))
            return []

    def update_dimension_names(self, dim_names, dim_indices=None):
        dim_names = ensure_list(dim_names)
        if dim_indices is None:
            dim_indices = list(range(len(dim_names)))
        else:
            dim_indices = ensure_list(dim_indices)
        new_name = {}
        for dim_name, dim_index in zip(dim_names, dim_indices):
            new_name[self._data.dims[dim_index]] = dim_name
        self._data = self._data.rename(new_name)

    # -----------------------general slicing methods-------------------------------------------

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
        integers = False
        if is_integer(slice_tuple) or isinstance(slice_tuple, string_types):
            integers = True
            return ([slice_tuple],), integers
        else:
            if isinstance(slice_tuple, slice):
                slice_tuple = (slice_tuple,)
            slice_list = []
            for slc in slice_tuple:
                if is_integer(slc) or isinstance(slc, string_types):
                    slice_list.append([slc])
                    if is_integer(slc):
                        integers = True
                else:
                    if isinstance(slc, slice):
                        elements = [slc.start, slc.stop, slc.step]
                    else:
                        elements = slc
                    if np.any([is_integer(islc) for islc in elements]):
                        integers = True
                    slice_list.append(slc)
            return tuple(slice_list), integers

    def _get_item(self, slice_tuple, **kwargs):
        slice_tuple, integers = self._assert_array_indices(slice_tuple)
        try:
            # For integer indices
            return self.duplicate(_data=self._data[slice_tuple], **kwargs)
        except:
            try:
                if integers:
                    raise
                # For label indices
                # xrarray.DataArray.loc slices along labels
                out = self.duplicate(_data=self._data.loc[slice_tuple], **kwargs)
                return out
            except:
                # Still, for a conflicting mixture that has to be resolved
                return self.duplicate(_data=self._data[self._process_slices(slice_tuple)], **kwargs)

    # Return a TimeSeries object
    def __getitem__(self, slice_tuple):
        out = self._get_item(slice_tuple)
        out.configure()
        return out

    def __setitem__(self, slice_tuple, values):
        slice_tuple, integers = self._assert_array_indices(slice_tuple)
        # Mind that xarray can handle setting values both from a numpy array and/or another xarray
        if isinstance(values, self.__class__):
            values = values._data
        try:
            # For integer indices
            self._data[slice_tuple] = values
        except:
            try:
                # For label indices
                # xrarray.DataArray.loc slices along labels
                if integers:
                    raise
                self._data.loc[slice_tuple] = values
            except:
                # Still, for a conflicting mixture that has to be resolved
                self._data[self._process_slices(slice_tuple)] = values
        self.configure()

    #-----------------------slicing by a particular dimension-------------------------------------------

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

    def _index_or_label_or_slice(self, inputs, dim):
        inputs = ensure_list(inputs)
        if np.all([is_integer(inp) for inp in inputs]):
            return "index", inputs
        elif np.all([isinstance(inp, string_types) for inp in inputs]):
            return "label", inputs
        elif isinstance(inputs, slice):
            return "slice", inputs
        elif np.all([isinstance(inp, string_types) or is_integer(inp) for inp in inputs]):
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
        return self.duplicate(_data=self._data[:, :, :, index_start:index_end], **kwargs)

    def __getattr__(self, attr_name):
        # We are here because attr_name is not an attribute of TimeSeries...
        if hasattr(self._data, attr_name):
            # First try to see if this is a xarray.DataArray attribute
            return getattr(self._data, attr_name)
        else:
            # TODO: find out if this part works, given that it is not really necessary
            # Now try to behave as if this was a getitem call:
            for i_dim in range(1, self.nr_dimensions):
                if self.get_dimension_name(i_dim) in self.labels_dimensions.keys() and \
                        attr_name in self.get_dimension_labels(i_dim):
                    return self.slice_data_across_dimension_by_label(attr_name, i_dim)
            raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def __setattr__(self, attr_name, value):
        try:
            super(TimeSeries, self).__setattr__(attr_name, value)
            return None
        except:
            # We are here because attr_name is not an attribute of TimeSeries...
            # First try to see if this is a xarray.DataArray attribute
            if hasattr(self._data, attr_name):
                setattr(self._data, attr_name, value)
            elif attr_name == "data":
                setattr(self._data, "values", value)
            elif attr_name == "labels_ordering":
                self.update_dimension_names(value)
            elif attr_name == "labels_dimensions":
                self._data = self._data.assign_coords(value)
            elif attr_name == "time":
                self._data = self._data.assign_coords({self._data.dims[0]: value})
            else:
                # Now try to behave as if this was a setitem call:
                slice_list = [slice(None)]  # for first dimension index, i.e., time
                for i_dim in range(1, self.nr_dimensions):
                    if self.get_dimension_name(i_dim) in self.labels_dimensions.keys() and \
                            attr_name in self.get_dimension_labels(i_dim):
                        slice_list.append(attr_name)
                        self._data.loc[tuple(slice_list)] = value
                        return None
                    else:
                        slice_list.append(slice(None))
                raise AttributeError("%r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def swapaxes(self, ax1, ax2):
        dims = list(self._data.dims)
        dims[ax1] = self._data.dims[ax2]
        dims[ax2] = self._data.dims[ax1]
        new_self = self.duplicate()
        new_self._data = self._data.transpose(*dims)
        new_self.configure()
        return new_self

    def plot(self, time=None, data=None, y=None, hue=None, col=None, row=None,
             figname=None, plotter_config=None, **kwargs):
        if data is None:
            data = self._data.copy()
        if time is None or len(time) == 0:
            time = data.dims[0]
        if figname is None:
            figname = kwargs.pop("figname", "%s" % data.name)
        data.name = ""
        for dim_name, dim in zip(["y", "hue", "col", "row"],
                                 [y, hue, col, row]):
            if dim is not None:
                id = data.dims.index(dim)
                if data.shape[id] > 1:
                    kwargs[dim_name] = dim
        # If we have a hue argument, and the size of the corresponding dimension > MAX_LBLS_IN_LEGEND, remove the legend.
        if kwargs.get("hue", None) and kwargs.get("add_legend", None) is None and \
                data.shape[data.dims.index(kwargs["hue"])] > MAX_LBLS_IN_LEGEND:
            kwargs["add_legend"] = False
        output = data.plot(x=time, **kwargs)
        pyplot.gcf().suptitle(figname)
        pyplot.gcf().canvas.manager.set_window_title(figname)
        pyplot.gcf().tight_layout()
        # TODO: Something better than this temporary hack for base_plotter functionality
        if plotter_config is not None:
            save_show_figure(plotter_config, figname)
        return output

    def _prepare_plot_args(self, **kwargs):
        plotter_config = kwargs.pop("plotter_config", None)
        labels_ordering = self.labels_ordering
        time = labels_ordering[0]
        return time, labels_ordering, plotter_config, kwargs

    def plot_map(self, **kwargs):
        """In this plotting method, we have by definition y axis defined but no hue"""
        if kwargs.pop("per_variable", False):
            for var in self.labels_dimensions[self.labels_ordering[1]]:
                var_ts = self[:, var]
                var_ts.name = ": ".join([var_ts.name, var])
                var_ts.plot_map(**kwargs)
            return
        if np.all([s < 2 for s in self.shape[1:]]):
            return self.plot_timeseries(**kwargs)
        time, labels_ordering, plotter_config, kwargs = \
            self._prepare_plot_args(**kwargs)
        y = kwargs.pop("y", None)      # The maximum dimension
        row = kwargs.pop("row", None)  # The next maximum dimension of size > 1
        col = kwargs.pop("col", None)  # The next maximum dimension of size > 1
        kwargs.pop("hue", None)  # Ignore hue if set by mistake
        labels_ordered_by_size = labels_ordering(np.argort(self.shape[1:])[::-1]+ 1)
        if y is None:
            # Loop across the dimensions in decreasing size order:
            for dim_label in labels_ordered_by_size:
                # If y_label is not used as a col or row...
                if dim_label not in [col, row]:
                    # ...set it as y
                    y = dim_label
                    break
        if row is None:
            # Loop across the dimensions in decreasing size order:
            for dim_label in labels_ordered_by_size:
                dim = labels_ordering.index(dim_label)
                # ...and if size > 1...
                if self.shape[dim] > 1:
                    # ...and the dimension is not already used...
                    if dim_label not in [y, row]:
                        # ...set is as col
                        row = dim_label
                        break
        if col is None:
            # Loop across the dimensions in decreasing size order:
            for dim_label in labels_ordered_by_size:
                dim = labels_ordering.index(dim_label)
                # ...and if size > 1...
                if self.shape[dim] > 1:
                    # ...and the dimension is not already used...
                    if dim_label not in [y, row]:
                        # ...set is as col
                        col = dim_label
                        break
        kwargs["robust"] = kwargs.pop("robust", False)
        kwargs["cmap"] = kwargs.pop("cmap", "jet")
        if self.shape[1] < 2:  # only one variable
            kwargs["figname"] = kwargs.pop("figname", "%s" % (self.title + "Map")) + ": " \
                      + self.labels_dimensions[labels_ordering[1]][0]
            kwargs["figsize"] = kwargs.pop("figsize", plotter_config.LARGE_SIZE)
        else:
            kwargs["figname"] = kwargs.pop("figname", "%s" % self.title)
            kwargs["figsize"] = kwargs.pop("figsize", plotter_config.VERY_LARGE_SIZE)
        return self.plot(data=None, y=y, hue=None, col=col, row=row,
                         plotter_config=plotter_config, **kwargs)

    def plot_timeseries(self, **kwargs):
        """In this plotting method, we can have hue defined but no y axis."""
        if kwargs.pop("per_variable", False):
            outputs = []
            for var in self.labels_dimensions[self.labels_ordering[1]]:
                var_ts = self[:, var]
                var_ts.name = ": ".join([var_ts.name, var])
                outputs.append(var_ts.plot_timeseries(**kwargs))
            return outputs
        if np.all([s > 1 for s in self.shape[1:]]):
            return self.plot_raster(**kwargs)
        time, labels_ordering, plotter_config, kwargs = \
            self._prepare_plot_args(**kwargs)
        row = kwargs.pop("row", None)  # Regions, or Variables, or Modes/Populations/Neurons <= 10
        col = kwargs.pop("col", None)  # Variables, or Modes/Populations/Neurons <= 4
        hue = kwargs.pop("hue", None)  # Modes/Populations/Neurons, or Variables, or Regions
        kwargs.pop("y", None)  # Ignore y if set by mistake
        if row is None:
            if self.shape[2] > 1 and labels_ordering[2] not in [col, hue]:
                row = labels_ordering[2]
            elif self.shape[1] > 1 and labels_ordering[1] not in [col, hue]:
                row = labels_ordering[1]
            elif self.shape[3] > 1 and self.shape[3] <= 10 and labels_ordering[3] not in [col, hue]:
                row = labels_ordering[3]
        if row is not None and hue is None:
            if self.shape[3] > 1 and labels_ordering[3] not in [col, row]:
                hue = labels_ordering[3]
            elif self.shape[1] > 1 and labels_ordering[1] not in [col, row]:
                hue = labels_ordering[1]
            elif self.shape[2] > 1 and labels_ordering[2] not in [col, row]:
                hue = labels_ordering[2]
        if row is not None and col is None:
            if self.shape[1] > 1 and labels_ordering[1] not in [row, hue]:
                col = labels_ordering[1]
            elif self.shape[3] > 1 and self.shape[3] <= 4 and labels_ordering[3] not in [row, hue]:
                col = labels_ordering[3]
        if self.shape[1] < 2:
            figname = kwargs.pop("figname", "%s" % (self.title)) + ": " \
                      + self.labels_dimensions[labels_ordering[1]][0]
            kwargs["figname"] = figname
            kwargs["figsize"] = kwargs.pop("figsize", plotter_config.LARGE_SIZE)
        else:
            kwargs["figsize"] = kwargs.pop("figsize", plotter_config.VERY_LARGE_SIZE)
        self.plot(data=None, y=None, hue=hue, col=col, row=row,
                  plotter_config=plotter_config, subplot_kws={'ylabel': ''}, **kwargs)

    def plot_raster(self, **kwargs):
        figname = kwargs.pop("figname", "%s" % (self.title + " raster"))
        """In this plotting method, we can have hue defined but no y axis.
           We arrange a dimension along the y axis instead.
        """
        if kwargs.pop("per_variable", False):
            outputs = []
            figsize = kwargs.pop("figsize", None)
            for var in self.labels_dimensions[self.labels_ordering[1]]:
                var_ts = self[:, var]
                outputs.append(var_ts.plot_raster(figsize=figsize, **kwargs))
            return outputs
        if np.all([s < 2 for s in self.shape[1:]]):
            return self.plot_timeseries(**kwargs)
        time, labels_ordering, plotter_config, kwargs = \
            self._prepare_plot_args(**kwargs)
        labels_dimensions = self.labels_dimensions
        # hue: Regions or Modes/Samples/Populations
        kwargs["hue"] = None
        yind = 2  # Regions
        if np.all([s > 1 for s in self.shape[2:]]):
            kwargs["hue"] = labels_ordering[3]  # Modes/Samples/Populations
            # If we have a hue argument, and the size of the corresponding dimension > MAX_LBLS_IN_LEGEND,
            # remove the legend.
            if kwargs.get("add_legend", None) is None and \
                    self.shape[self.dims.index(kwargs["hue"])] > MAX_LBLS_IN_LEGEND:
                kwargs["add_legend"] = False
        elif self.shape[2] == 1 and self.shape[3] > 1:
            yind = 3  # Modes/Samples/Populations
        yticklabels = labels_dimensions[labels_ordering[yind]]
        data = self._data.copy()
        slice_tuple = (slice(None), 0, slice(None), slice(None))
        figname = kwargs.pop("figname", "%s" % (self.title + " raster"))
        if self.shape[1] < 2:
            try:
                figname += ": %s" % labels_dimensions[labels_ordering[1]][0]  # Variable
            except:
                pass
            figsize = kwargs.pop("figsize", plotter_config.LARGE_SIZE)
        else:
            figsize = kwargs.pop("figsize", plotter_config.VERY_LARGE_SIZE)
        fig, axes = pyplot.subplots(ncols=self.shape[1], num=figname, figsize=figsize)
        pyplot.suptitle(figname)
        axes = np.array(ensure_list(axes))
        for i_var, var in enumerate(labels_dimensions[labels_ordering[1]]):
            # Remove mean
            data[:, i_var] -= data[:, i_var].mean()
            # Compute approximate range for this variable
            amplitude = 0.9 * (data[:, i_var].max() - data[:, i_var].min())
            if amplitude == 0.0:
                amplitude = 1.0
            # Add the step on y axis for this variable and for each Region's data
            slice_tuple = [slice(None), i_var, slice(None), slice(None)]
            yticks = []
            for i_y in range(self.shape[yind]):
                slice_tuple[yind] = i_y
                yticks.append(-amplitude * i_y)
                data[tuple(slice_tuple)] += yticks[-1]
                data[tuple(slice_tuple)].plot(x=time, ax=axes[i_var], **kwargs)
            axes[i_var].set_yticks(yticks)
            axes[i_var].set_yticklabels(yticklabels)
            axes[i_var].set_title(var)
        pyplot.gcf().canvas.manager.set_window_title(figname)
        if plotter_config is not None:
            save_show_figure(plotter_config, figname, fig)
        return fig, axes


# TODO: Slicing should also slice Connectivity, Surface, Volume, Sensors etc accordingly...


class TimeSeriesRegion(TimeSeries):
    """ A time-series associated with the regions of a connectivity. """

    title = Attr(str, default="Region Time Series")

    connectivity = Attr(field_type=connectivity.Connectivity)
    region_mapping_volume = Attr(field_type=region_mapping.RegionVolumeMapping, required=False)
    region_mapping = Attr(field_type=region_mapping.RegionMapping, required=False)
    _default_labels_ordering = List(of=str, default=("Time", "State Variable", "Region", "Mode"))

    def __init__(self, data=None, **kwargs):
        if isinstance(data, (TimeSeriesRegion, TimeSeriesRegionTVB)):
            for datatype_name in ["connectivity", "region_mapping_volume", "region_mapping"]:
                setattr(self, datatype_name, getattr(data, datatype_name))
        for datatype_name in ["connectivity", "region_mapping_volume", "region_mapping"]:
            datatype = kwargs.pop(datatype_name, None)
            if datatype is not None:
                setattr(self, datatype_name, datatype)
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

    def _duplicate(self, **kwargs):
        _data, kwargs = super(TimeSeriesRegion, self)._duplicate(**kwargs)
        for datatype_name in ["connectivity", "region_mapping_volume", "region_mapping"]:
            kwargs[datatype_name] = kwargs.pop(datatype_name, getattr(self, datatype_name))
        return _data, kwargs

    def configure(self):
        super(TimeSeriesRegion, self).configure()
        labels = self.get_dimension_labels(2)
        number_of_labels = len(labels)
        if number_of_labels == 0:
            if self.number_of_labels == self.connectivity.region_labels.shape[0]:
                self._data.assign_coords({self.labels_ordering[2]: self.connectivity.region_labels})
            else:
                logger.warning("RegionTimeSeries labels is empty!\n"
                               "Labels can not be set from Connectivity, because RegionTimeSeries shape[2]=%d "
                               "is not equal to the Connectivity size %d!\n"
                               % (self.number_of_labels, self.connectivity.region_labels.shape[0]))
                return
        if self.connectivity.number_of_regions > number_of_labels:
            try:
                self.connectivity = HeadService().slice_connectivity(self.connectivity, labels)
            except Exception as e:
                logger.warning(str(e))
                logger.warning("Connectivity and RegionTimeSeries labels agreement failed!")

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesRegionTVB().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesSurface(TimeSeries):
    """ A time-series associated with a Surface. """

    title = Attr(str, default="Surface Time Series")

    surface = Attr(field_type=surfaces.CorticalSurface)
    _default_labels_ordering = List(of=str, default=("Time", "State Variable", "Vertex", "Mode"))

    def __init__(self, data=None, **kwargs):
        if isinstance(data, TimeSeriesSurface):
            self.surface = data.surface
        surface = kwargs.pop("surface")
        if surface is not None:
            self.surface = surface
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

    def _duplicate(self, **kwargs):
        _data, kwargs = super(TimeSeriesSurface, self)._duplicate(**kwargs)
        kwargs["surface"] = kwargs.pop("surface", getattr(self, datatype_name))
        return _data, kwargs


class TimeSeriesVolume(TimeSeries):
    """ A time-series associated with a Volume. """

    title = Attr(str, default="Volume Time Series")

    volume = Attr(field_type=volumes.Volume)
    _default_labels_ordering = List(of=str, default=("Time", "X", "Y", "Z"))

    def __init__(self, data=None, **kwargs):
        if isinstance(data, (TimeSeriesVolume, TimeSeriesVolumeTVB)):
            self.volume = data.volume
        volume = kwargs.pop("volume", None)
        if volume is not None:
            self.volume, volume
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

    def _duplicate(self, **kwargs):
        _data, kwargs = super(TimeSeriesVolume, self)._duplicate(**kwargs)
        kwargs["volume"] = kwargs.pop("volume", getattr(self, datatype_name))
        return _data, kwargs


class TimeSeriesSensors(TimeSeries):
    title = Attr(str, default="Sensor Time Series")

    def __init__(self, data=None, **kwargs):
        if not isinstance(data, (TimeSeriesSensors, TimeSeriesSensorsTVB)):
            self.sensors = data.sensors
        sensors = kwargs.pop("sensors")
        if sensors is not None:
            self.sensors = sensors
        super(TimeSeriesSensors, self).__init__(data, **kwargs)

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesSensors, self).summary_info()
        summary.update({"Source Sensors": self.sensors.title})
        return summary

    def _duplicate(self, datatype=None, **kwargs):
        if datatype is None:
            datatype = self.__class__
        _data, kwargs = super(datatype, self)._duplicate(**kwargs)
        kwargs["sensors"] = kwargs.pop("sensors", getattr(self, datatype))
        return _data, kwargs

    def configure(self):
        super(TimeSeriesSensors, self).configure()
        labels = self.get_dimension_labels(2)
        number_of_labels = len(labels)
        if number_of_labels == 0:
            if self.number_of_labels == self.sensors.number_of_sensors:
                self._data.assign_coords({self.labels_ordering[2]: self.sensors.labels})
            else:
                logger.warning("SensorsTimeSeries labels is empty!\n"
                               "Labels can not be set from Sensors, because SensorsTimeSeries shape[2]=%d "
                               "is not equal to the Sensors size %d!\n"
                               % (self.number_of_labels, self.sensors.number_of_sensors))
                return
        if self.sensors.number_of_sensors > number_of_labels:
            try:
                self.sensors = HeadService().slice_sensors(self.sensors, labels)
            except Exception as e:
                logger.warning(str(e))
                logger.warning("Sensors and SensorsTimeSeries labels agreement failed!")

    def to_tvb_instance(self, datatype=TimeSeriesSensorsTVB, **kwargs):
        return datatype().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesEEG(TimeSeriesSensors):
    """ A time series associated with a set of EEG sensors. """

    title = Attr(str, default="EEG Time Series")

    sensors = Attr(field_type=sensors.SensorsEEG)
    _default_labels_ordering = List(of=str, default=("Time", "1", "EEG Sensor", "1"))

    def _duplicate(self, **kwargs):
        super(TimeSeriesEEG, self)._duplicate(TimeSeriesEEG, **kwargs)

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesEEGTVB().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesSEEG(TimeSeriesSensors):
    """ A time series associated with a set of Internal sensors. """

    title = Attr(str, default="SEEG Time Series")

    sensors = Attr(field_type=sensors.SensorsInternal)
    _default_labels_ordering = List(of=str, default=("Time", "1", "sEEG Sensor", "1"))

    def _duplicate(self, **kwargs):
        super(TimeSeriesSEEG, self)._duplicate(TimeSeriesSEEG, **kwargs)

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesSEEGTVB().from_xarray_DataArray(self._data, **kwargs)


class TimeSeriesMEG(TimeSeriesSensors):
    """ A time series associated with a set of MEG sensors. """

    title = Attr(str, default="MEG Time Series")

    sensors = Attr(field_type=sensors.SensorsMEG)
    _default_labels_ordering = List(of=str, default=("Time", "1", "MEG Sensor", "1"))

    def _duplicate(self, **kwargs):
        super(TimeSeriesMEG, self)._duplicate(TimeSeriesMEG, **kwargs)

    def to_tvb_instance(self, **kwargs):
        return TimeSeriesMEGTVB().from_xarray_DataArray(self._data, **kwargs)


TimeSeriesDict = {TimeSeries.__name__: TimeSeries,
                  TimeSeriesRegion.__name__: TimeSeriesRegion,
                  TimeSeriesVolume.__name__: TimeSeriesVolume,
                  TimeSeriesSurface.__name__: TimeSeriesSurface,
                  TimeSeriesEEG.__name__: TimeSeriesEEG,
                  TimeSeriesMEG.__name__: TimeSeriesMEG,
                  TimeSeriesSEEG.__name__: TimeSeriesSEEG}
