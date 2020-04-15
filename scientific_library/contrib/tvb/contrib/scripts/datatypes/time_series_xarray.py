# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
import xarray as xr
from six import string_types
from tvb.basic.neotraits.api import HasTraits, Attr, List, narray_summary_info
from tvb.contrib.scripts.datatypes.time_series import TimeSeries as TimeSeriesTVB
from tvb.contrib.scripts.utils.data_structures_utils import is_integer
from tvb.datatypes import sensors, surfaces, volumes, region_mapping, connectivity


def prepare_4d(data):
    if data.ndim < 2:
        raise ValueError("The data array is expected to be at least 2D!")
    if data.ndim < 4:
        if data.ndim == 2:
            data = np.expand_dims(data, 2)
        data = np.expand_dims(data, 3)
    return data


class TimeSeries(HasTraits):
    """
    Base time-series dataType.
    """

    _data = xr.DataArray([])

    _default_labels_ordering = List(
        default=("Time", "State Variable", "Space", "Mode"),
        label="Dimension Names",
        doc="""List of strings representing names of each data dimension""")

    title = Attr(str)

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
    def nr_dimensions(self):
        return self._data.ndim

    @property
    def number_of_dimensions(self):
        return self.nr_dimensions

    @property
    def time_length(self):
        try:
            return self._data.shape[0]
        except:
            return None

    @property
    def number_of_variables(self):
        try:
            return self.shape[1]
        except:
            return None

    @property
    def number_of_labels(self):
        try:
            return self.shape[2]
        except:
            return None

    @property
    def number_of_samples(self):
        try:
            return self.shape[3]
        except:
            return None

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
        d = {}
        for key, val in zip(list(self._data.coords.keys()),
                            list([value.values for value in self._data.coords.values()])):
            d[key] = val
        return d

    @property
    def time(self):
        return self._data.coords[self._data.dims[0]].values

    # xarrays have a attrs dict with useful attributes

    @property
    def start_time(self):
        try:
            return self.time[0]
        except:
            return None

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
    def sample_period(self):
        try:
            return np.mean(np.diff(self.time))
        except:
            return None

    @property
    def sample_rate(self):
        try:
            return 1.0 / self.sample_period
        except:
            return None

    @property
    def sample_period_unit(self):
        try:
            return self._data.attrs["sample_period_unit"]
        except:
            return ""

    @property
    def time_unit(self):
        return self.sample_period_unit

    def squeeze(self):
        return self._data.squeeze()

    @property
    def squeezed(self):
        return self.data.squeeze()

    @property
    def flattened(self):
        return self.data.flatten()

    def __setattr__(self, name, value):
        if name == "data":
            setattr(self._data, "values", value)
        elif name == "labels_ordering":
            setattr(self._data, "dims", value)
        elif name == "labels_dimensions":
            setattr(self._data, "coords", value)
        elif name == "time":
            self._data.coords[self._data.dims[0]] = value
        elif name == "start_time":
            self._data.coords[self._data.dims[0]][0] = value
        elif name == "sample_period":
            self._data.attrs["sample_period"] = value
        elif name == "sample_period_unit":
            self._data.attrs["sample_period_unit"] = value
        else:
            super(TimeSeries, self).__setattr__(name, value)

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
                if start_time is not None and sample_period is not None:
                    end_time = start_time + (time_length - 1) * sample_period
                    time = np.arange(start_time, end_time + sample_period, sample_period)
                    return time, start_time, end_time, sample_period, kwargs
                else:
                    raise ValueError("Neither time vector nor start_time and/or "
                                     "sample_period are provided as input arguments!")
            else:
                assert time_length == len(time)
                start_time = time[0]
                end_time = time[-1]
                if len(time) > 1:
                    sample_period = np.mean(np.diff(time))
                    assert end_time == start_time + (time_length - 1) * sample_period
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

    def from_TVB_time_series(self, ts, **kwargs):
        labels_ordering = kwargs.pop("labels_ordering", kwargs.pop("dims", ts.labels_ordering))
        labels_dimensions = kwargs.pop("labels_dimensions", kwargs.pop("coords", ts.labels_dimensions))
        name = kwargs.pop("name", kwargs.pop("title", ts.title))
        time = kwargs.pop("time", ts.time)
        labels_dimensions[labels_ordering[0]] = time
        for label, dimensions in labels_dimensions.items():
            id = labels_ordering.index(label)
            if ts.shape[id] != len(dimensions):
                labels_dimensions[label] = np.arange(ts.shape[id]).astype("i")
        kwargs["sample_period_unit"] = getattr(ts, "sample_period_unit", kwargs.pop('sample_period_unit', ""))
        self._data = xr.DataArray(ts.data,
                                  dims=labels_ordering,
                                  coords=labels_dimensions,
                                  name=name, attrs=kwargs)

    def from_numpy(self, data, **kwargs):
        # We have to infer time and labels inputs from kwargs
        data = prepare_4d(data)
        time, start_time, end_time, sample_period, kwargs = self._configure_input_time(data, **kwargs)
        labels_ordering, labels_dimensions, kwargs = self._configure_input_labels(**kwargs)
        if time is not None:
            if labels_dimensions is None:
                labels_dimensions = {}
            labels_dimensions[labels_ordering[0]] = time
        self._data = xr.DataArray(data, dims=labels_ordering, coords=labels_dimensions,
                                  name=self.__class__.__name__, attrs=kwargs)

    def _configure_time(self):
        assert self.time[0] == self.start_time
        assert self.time[-1] == self.end_time
        if self.time_length > 1:
            assert self.sample_period == (self.end_time - self.start_time) / (self.time_length - 1)

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
        if self.name is None:
            self.title = "TimeSeries"
        else:
            self.title = self.name
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
            kwargs = self.from_numpy(np.array(data), **kwargs)
        elif isinstance(data, np.ndarray):
            kwargs = self.from_numpy(data, **kwargs)
        elif isinstance(data, self.__class__):
            attributes = data.__dict__.items()
            attributes.update(**kwargs)
            for attr, val in attributes.items():
                setattr(self, attr, val)
        elif isinstance(data, TimeSeriesTVB):
            self.from_TVB_time_series(data, **kwargs)
        else:
            # Assuming data is an input xr.DataArray() can handle,
            if isinstance(data, dict):
                # ...either as kwargs
                self._data = xr.DataArray(**data)
            else:
                # ...or as args
                # including a xr.DataArray or None
                self._data = xr.DataArray(data,
                                          dims=kwargs.pop("dims", kwargs.pop("labels_ordering", None)),
                                          coords=kwargs.pop("coords", kwargs.pop("labels_dimensions", None)))
            self._data.attrs = kwargs
            super(TimeSeries, self).__init__(**kwargs)
        super(TimeSeries, self).__init__()
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
        duplicate = deepcopy(self)
        for attr, value in kwargs.items():
            setattr(duplicate, attr, value)
        duplicate.configure()
        return duplicate

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

    def _get_index_for_slice_label(self, slice_label, slice_idx):
        return self.labels_dimensions[self.labels_ordering[slice_idx]].tolist().index(slice_label)

    def _check_for_string_or_float_slice_indices(self, current_slice, slice_idx):
        slice_start = current_slice.start
        slice_stop = current_slice.stop

        if isinstance(slice_start, string_types) or isinstance(slice_start, float):
            slice_start = self._get_index_for_slice_label(slice_start, slice_idx)
        if isinstance(slice_stop, string_types) or isinstance(slice_stop, float):
            # NOTE!: In case of a string slice, we consider stop included!
            slice_stop = self._get_index_for_slice_label(slice_stop, slice_idx) + 1

        return slice(slice_start, slice_stop, current_slice.step)

    def _resolve_mixted_slice(self, slice_tuple):
        slice_list = []
        for idx, current_slice in enumerate(slice_tuple):
            if isinstance(current_slice, slice):
                slice_list.append(self._check_for_string_or_float_slice_indices(current_slice, idx))
            else:
                # If not a slice, it will be an iterable:
                for i_slc, slc in enumerate(current_slice):
                    if isinstance(slc, string_types) or isinstance(slc, float):
                        current_slice[i_slc] = self.labels_dimensions[self.labels_ordering[idx]].tolist().index(slc)
                    else:
                        current_slice[i_slc] = slc
                slice_list.append(current_slice)
        return tuple(slice_list)

    # Return a TimeSeries object
    def __getitem__(self, slice_tuple):
        slice_tuple = self._assert_array_indices(slice_tuple)
        try:
            # For integer indices
            return self.duplicate(_data=self._data[slice_tuple])
        except:
            try:
                # For label indices
                # xrarray.DataArray.loc slices along labels
                # Assuming that all dimensions without input labels
                # are configured with labels of integer indices=
                return self.duplicate(_data=self._data.loc[slice_tuple])
            except:
                # Still, for a conflicting mixture that has to be resolved
                return self.duplicate(_data=self._data[self._resolve_mixted_slice(slice_tuple)])

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
                self._data[self._resolve_mixted_slice(slice_tuple)] = values

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


class SensorsTSBase(TimeSeries):

    def __init__(self, data=None, **kwargs):
        if not isinstance(data, SensorsTSBase):
            self.sensors = kwargs.pop("sensors")
        super(SensorsTSBase, self).__init__(data, **kwargs)

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
    _default_labels_ordering = List(of=str, default=("Time", "1", "EEG Sensor", "1"))


class TimeSeriesMEG(SensorsTSBase):
    """ A time series associated with a set of MEG sensors. """

    sensors = Attr(field_type=sensors.SensorsMEG)
    _default_labels_ordering = List(of=str, default=("Time", "1", "MEG Sensor", "1"))


class TimeSeriesSEEG(SensorsTSBase):
    """ A time series associated with a set of Internal sensors. """

    sensors = Attr(field_type=sensors.SensorsInternal)
    _default_labels_ordering = List(of=str, default=("Time", "1", "sEEG Sensor", "1"))


class TimeSeriesRegion(TimeSeries):
    """ A time-series associated with the regions of a connectivity. """

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


class TimeSeriesSurface(TimeSeries):
    """ A time-series associated with a Surface. """

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


class TimeSeriesVolume(TimeSeries):
    """ A time-series associated with a Volume. """

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


TimeSeriesDict = {TimeSeries.__name__: TimeSeries,
                  TimeSeriesRegion.__name__: TimeSeriesRegion,
                  TimeSeriesVolume.__name__: TimeSeriesVolume,
                  TimeSeriesSurface.__name__: TimeSeriesSurface,
                  TimeSeriesEEG.__name__: TimeSeriesEEG,
                  TimeSeriesMEG.__name__: TimeSeriesMEG,
                  TimeSeriesSEEG.__name__: TimeSeriesSEEG}
