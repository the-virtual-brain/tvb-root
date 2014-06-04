# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
The Data component of TimeSeries DataTypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
from tvb.basic.traits.types_mapped import MappedType
import tvb.basic.traits.types_basic as basic
import tvb.basic.traits.core as core
import tvb.datatypes.arrays as arrays
import tvb.datatypes.sensors as sensors_module
import tvb.datatypes.connectivity as connectivity_module
import tvb.datatypes.surfaces as surfaces
import tvb.datatypes.volumes as volumes



class TimeSeriesData(MappedType):
    """
    Base time-series dataType.
    """

    title = basic.String

    data = arrays.FloatArray(
        label="Time-series data",
        file_storage=core.FILE_STORAGE_EXPAND,
        doc="""An array of time-series data, with a shape of [tpts, :], where ':' represents 1 or more dimensions""")

    nr_dimensions = basic.Integer(
        label="Number of dimension in timeseries",
        default=4)

    length_1d, length_2d, length_3d, length_4d = [basic.Integer] * 4

    labels_ordering = basic.List(
        default=["Time", "State Variable", "Space", "Mode"],
        label="Dimension Names",
        doc="""List of strings representing names of each data dimension""")

    labels_dimensions = basic.Dict(
        default={},
        label="Specific labels for each dimension for the data stored in this timeseries.",
        doc=""" A dictionary containing mappings of the form {'dimension_name' : [labels for this dimension] }""")
    ## TODO (for Stuart) : remove TimeLine and make sure the correct Period/start time is returned by different monitors in the simulator

    time = arrays.FloatArray(
        file_storage=core.FILE_STORAGE_EXPAND,
        label="Time-series time",
        required=False,
        doc="""An array of time values for the time-series, with a shape of [tpts,].
        This is 'time' as returned by the simulator's monitors.""")

    start_time = basic.Float(label="Start Time:")

    sample_period = basic.Float(label="Sample period", default=1.0)

    # Specify the measure unit for sample period (e.g sec, msec, usec, ...)
    sample_period_unit = basic.String(
        label="Sample Period Measure Unit",
        default="ms")

    sample_rate = basic.Float(
        label="Sample rate",
        doc="""The sample rate of the timeseries""")



class TimeSeriesEEGData(TimeSeriesData):
    """ A time series associated with a set of EEG sensors. """
    _ui_name = "EEG time-series"
    sensors = sensors_module.SensorsEEG
    labels_ordering = basic.List(default=["Time", "1", "EEG Sensor", "1"])



class TimeSeriesMEGData(TimeSeriesData):
    """ A time series associated with a set of MEG sensors. """
    _ui_name = "MEG time-series"
    sensors = sensors_module.SensorsMEG
    labels_ordering = basic.List(default=["Time", "1", "MEG Sensor", "1"])
    
    
class TimeSeriesSEEGData(TimeSeriesData):
    """ A time series associated with a set of Internal sensors. """
    _ui_name = "Stereo-EEG time-series"
    sensors = sensors_module.SensorsInternal
    labels_ordering = basic.List(default=["Time", "1", "sEEG Sensor", "1"])



class TimeSeriesRegionData(TimeSeriesData):
    """ A time-series associated with the regions of a connectivity. """
    _ui_name = "Region time-series"
    connectivity = connectivity_module.Connectivity
    labels_ordering = basic.List(default=["Time", "State Variable", "Region", "Mode"])



class TimeSeriesSurfaceData(TimeSeriesData):
    """ A time-series associated with a Surface. """
    _ui_name = "Surface time-series"
    surface = surfaces.CorticalSurface
    labels_ordering = basic.List(default=["Time", "State Variable", "Vertex", "Mode"])



class TimeSeriesVolumeData(TimeSeriesData):
    """ A time-series associated with a Volume. """
    _ui_name = "Volume time-series"
    volume = volumes.Volume
    labels_ordering = basic.List(default=["Time", "X", "Y", "Z"])


