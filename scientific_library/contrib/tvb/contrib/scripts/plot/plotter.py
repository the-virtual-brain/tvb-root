# -*- coding: utf-8 -*-
from tvb.contrib.scripts.plot.time_series_plotter import TimeSeriesPlotter
from tvb.simulator.plot.plotter import Plotter as TVBPlotter


class Plotter(TVBPlotter):
    def plot_head(self, head):
        sensors_set = {}
        for s_type in ["eeg", "seeg", "meg"]:
            sensors = getattr(head, "%s_sensors" % s_type)
            projection = getattr(head, "%s_projection" % s_type)
            if sensors is not None and projection is not None:
                sensors_set[sensors] = projection
        return self.plot_head_tvb(head.connectivity, sensors_set)

    def plot_timeseries(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_tvb_time_series(*args, **kwargs)

    def plot_timeseries_interactive(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_time_series_interactive(*args, **kwargs)

    def plot_spectral_analysis_raster(self, *args, **kwargs):
        return TimeSeriesPlotter(self.config).plot_spectral_analysis_raster(*args, **kwargs)
