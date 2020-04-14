# -*- coding: utf-8 -*-
from tvb.contrib.scripts.plot.time_series_plotter import TimeSeriesPlotter
from tvb.simulator.plot.plotter import Plotter as TVBPlotter


class Plotter(TVBPlotter):
    def plot_head(self, head):
        return self.plot_head_tvb(head.connectivity, head.sensors)

    def plot_timeseries(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_tvb_time_series(*args, **kwargs)

    def plot_timeseries_interactive(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_time_series_interactive(*args, **kwargs)

    def plot_spectral_analysis_raster(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_spectral_analysis_raster(*args, **kwargs)
