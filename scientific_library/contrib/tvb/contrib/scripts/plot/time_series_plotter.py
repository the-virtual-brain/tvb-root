# -*- coding: utf-8 -*-

import numpy
from tvb.contrib.scripts.datatypes.time_series import TimeSeries
from tvb.simulator.plot.time_series_plotter import TimeSeriesPlotter as TVBTimeSeriesPlotter


class TimeSeriesPlotter(TVBTimeSeriesPlotter):
    def plot_time_series(self, time_series, mode="ts", subplots=None, special_idx=[], subtitles=[],
                         offset=0.5, title=None, figure_name=None, figsize=None, **kwargs):
        if isinstance(time_series, TimeSeries):
            if title is None:
                title = time_series.title
            return self.plot_ts(numpy.swapaxes(time_series.data, 1, 2),
                                time_series.time, time_series.variables_labels,
                                mode, subplots, special_idx, subtitles, time_series.space_labels,
                                offset, time_series.time_unit, title, figure_name, figsize)
        else:
            super(TimeSeriesPlotter, self).plot_time_series(time_series, mode, subplots, special_idx, subtitles, offset,
                                                            title, figure_name, figsize, **kwargs)

    def plot_time_series_interactive(self, time_series, first_n=-1, **kwargs):
        if isinstance(time_series, TimeSeries):
            self.plot_tvb_time_series_interactive(time_series._tvb, first_n, **kwargs)
        else:
            super(TimeSeriesPlotter, self).plot_time_series_interactive(time_series, first_n, **kwargs)

    def plot_spectral_analysis_raster(self, time_series, freq=None, spectral_options={},
                                      special_idx=[], labels=[], title='Spectral Analysis', figure_name=None,
                                      figsize=None, **kwargs):
        if isinstance(time_series, TimeSeries):
            return self.plot_ts_spectral_analysis_raster(numpy.swapaxes(time_series._tvb.data, 1, 2).squeeze(),
                                                         time_series.time, time_series.time_unit, freq,
                                                         spectral_options,
                                                         special_idx, labels, title, figure_name, figsize)
        else:
            super(TimeSeriesPlotter, self).plot_spectral_analysis_raster(time_series, freq, spectral_options,
                                                                         special_idx, labels, title, figure_name,
                                                                         figsize, **kwargs)
