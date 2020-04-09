# -*- coding: utf-8 -*-
from tvb.simulator.plot.base_plotter import BasePlotter
from tvb.simulator.plot.head_plotter import HeadPlotter
from tvb.simulator.plot.time_series_plotter import TimeSeriesPlotter


class Plotter(object):

    def __init__(self, config=None):
        self.config = config

    @property
    def base(self):
        return BasePlotter(self.config)

    def tvb_plot(self, plot_fun_name, *args, **kwargs):
        return BasePlotter(self.config).tvb_plot(plot_fun_name, *args, **kwargs)

    def plot_head_tvb(self, connectivity, sensors):
        return HeadPlotter(self.config).plot_head(connectivity, sensors)

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
