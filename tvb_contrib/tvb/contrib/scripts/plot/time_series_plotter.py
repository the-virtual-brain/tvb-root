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
