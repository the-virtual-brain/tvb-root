# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
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

from tvb.simulator.plot.base_plotter import BasePlotter
from tvb.simulator.plot.config import FiguresConfig, CONFIGURED
from tvb.simulator.plot.head_plotter import HeadPlotter
from tvb.simulator.plot.time_series_plotter import TimeSeriesPlotter


class Plotter(object):

    def __init__(self, config=CONFIGURED):
        # type: (FiguresConfig) -> None
        self.config = config

    @property
    def base(self):
        return BasePlotter(self.config)

    def tvb_plot(self, plot_fun_name, *args, **kwargs):
        return BasePlotter(self.config).tvb_plot(plot_fun_name, *args, **kwargs)

    def plot_head_tvb(self, connectivity, sensors_set):
        return HeadPlotter(self.config).plot_head(connectivity, sensors_set)

    def plot_tvb_connectivity(self, *args, **kwargs):
        return HeadPlotter(self.config).plot_tvb_connectivity(*args, **kwargs)

    def plot_ts(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_ts(*args, **kwargs)

    def plot_ts_raster(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_ts_raster(*args, **kwargs)

    def plot_ts_trajectories(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_ts_trajectories(*args, **kwargs)

    def plot_tvb_timeseries(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_tvb_time_series(*args, **kwargs)

    def plot_timeseries(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_time_series(*args, **kwargs)

    def plot_raster(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_raster(*args, **kwargs)

    def plot_trajectories(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_trajectories(*args, **kwargs)

    def plot_timeseries_interactive(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_time_series_interactive(*args, **kwargs)

    def plot_tvb_timeseries_interactive(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_tvb_time_series_interactive(*args, **kwargs)

    def plot_power_spectra_interactive(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_power_spectra_interactive(*args, **kwargs)

    def plot_tvb_power_spectra_interactive(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_tvb_power_spectra_interactive(*args, **kwargs)

    def plot_ts_spectral_analysis_raster(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_spectral_analysis_raster(self, *args, **kwargs)

    def plot_spectral_analysis_raster(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_spectral_analysis_raster(self, *args, **kwargs)
