# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
The TimeSeries datatypes. This brings together the scientific and framework 
methods that are associated with the time-series data.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

from tvb.datatypes import sensors, surfaces, volumes, region_mapping, connectivity
from tvb.basic.logger.builder import get_logger
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, List, Int, Float, narray_summary_info

LOG = get_logger(__name__)


def prepare_time_slice(total_time_length, max_length=10 ** 4):
    """
    Limit the time dimension when retrieving from TS.
    If total time length is greater than MAX, then retrieve only the last part of the TS

    :param total_time_length: TS time dimension
    :param max_length: limiting number of TS steps

    :return: python slice
    """

    if total_time_length < max_length:
        return slice(total_time_length)

    return slice(total_time_length - max_length, total_time_length)


class TimeSeries(HasTraits):
    """
    Base time-series dataType.
    """

    title = Attr(str)

    data = NArray(
        label="Time-series data",
        # file_storage=core.FILE_STORAGE_EXPAND,
        doc="""An array of time-series data, with a shape of [tpts, :], where ':' represents 1 or more dimensions""")

    # mhtodo: should this not be a property
    nr_dimensions = Int(
        label="Number of dimension in timeseries",
        default=4
    )

    # length_1d, length_2d, length_3d, length_4d = [Int() for _ in range(4)]

    labels_ordering = List(
        default=("Time", "State Variable", "Space", "Mode"),
        label="Dimension Names",
        doc="""List of strings representing names of each data dimension"""
    )

    labels_dimensions = Attr(
        field_type=dict,
        default={},
        label="Specific labels for each dimension for the data stored in this timeseries.",
        doc=""" A dictionary containing mappings of the form {'dimension_name' : [labels for this dimension] }""")

    time = NArray(
        # file_storage=core.FILE_STORAGE_EXPAND,
        label="Time-series time",
        required=False,
        doc="""An array of time values for the time-series, with a shape of [tpts,].
            This is 'time' as returned by the simulator's monitors."""
    )

    start_time = Float(label="Start Time:")

    sample_period = Float(label="Sample period", default=1.0)

    # Specify the measure unit for sample period (e.g sec, msec, usec, ...)
    sample_period_unit = Attr(
        field_type=str,
        label="Sample Period Measure Unit",
        default="ms"
    )

    sample_rate = Float(
        label="Sample rate",
        doc="""The sample rate of the timeseries"""
    )

    # has_surface_mapping = Attr(field_type=bool, default=True)
    # has_volume_mapping = Attr(field_type=bool, default=False)

    def configure(self):
        """
        After populating few fields, compute the rest of the fields
        """
        super(TimeSeries, self).configure()
        self.nr_dimensions = self.data.ndim
        self.sample_rate = 1.0 / self.sample_period

        # for i in range(min(self.nr_dimensions, 4)):
        #     setattr(self, 'length_%dd' % (i + 1), int(data_shape[i]))

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
            "Length": self.sample_period * self.data.shape[0]
        }
        summary.update(narray_summary_info(self.data))
        return summary


class SensorsTSBase(TimeSeries):
    """
    Add framework related functionality for TS Sensor classes

    """
    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(SensorsTSBase, self).summary_info()
        summary.update({"Source Sensors": self.sensors.title})
        return summary


class TimeSeriesEEG(SensorsTSBase):
    """ A time series associated with a set of EEG sensors. """
    _ui_name = "EEG time-series"

    sensors = Attr(field_type=sensors.SensorsEEG)
    labels_ordering = List(of=str, default=("Time", "1", "EEG Sensor", "1"))


class TimeSeriesMEG(SensorsTSBase):
    """ A time series associated with a set of MEG sensors. """
    _ui_name = "MEG time-series"

    sensors = Attr(field_type=sensors.SensorsMEG)
    labels_ordering = List(of=str, default=("Time", "1", "MEG Sensor", "1"))


class TimeSeriesSEEG(SensorsTSBase):
    """ A time series associated with a set of Internal sensors. """
    _ui_name = "Stereo-EEG time-series"

    sensors = Attr(field_type=sensors.SensorsInternal)
    labels_ordering = List(of=str, default=("Time", "1", "sEEG Sensor", "1"))


class TimeSeriesRegion(TimeSeries):
    """ A time-series associated with the regions of a connectivity. """
    _ui_name = "Region time-series"
    connectivity = Attr(field_type=connectivity.Connectivity)
    region_mapping_volume = Attr(field_type=region_mapping.RegionVolumeMapping, required=False)
    region_mapping = Attr(field_type=region_mapping.RegionMapping, required=False)
    labels_ordering = List(of=str, default=("Time", "State Variable", "Region", "Mode"))

    def configure(self):
        """
        After populating few fields, compute the rest of the fields
        """
        super(TimeSeriesRegion, self).configure()
        # self.has_surface_mapping = self.region_mapping is not None or self._region_mapping is not None
        # self.has_volume_mapping = self.region_mapping_volume is not None or self._region_mapping_volume is not None

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
    _ui_name = "Surface time-series"
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
    _ui_name = "Volume time-series"
    volume = Attr(field_type=volumes.Volume)
    labels_ordering = List(of=str, default=("Time", "X", "Y", "Z"))

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this datatype.
        """
        summary = super(TimeSeriesVolume, self).summary_info()
        summary.update({"Source Volume": self.volume.title})
        return summary

    def configure(self):
        """
        After populating few fields, compute the rest of the fields
        """
        super(TimeSeriesVolume, self).configure()
        self.has_volume_mapping = True

