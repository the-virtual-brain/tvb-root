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

from tvb.contrib.scripts.plot.time_series_plotter import TimeSeriesPlotter
from tvb.simulator.plot.base_plotter import BasePlotter
from tvb.simulator.plot.plotter import Plotter as TVBPlotter


class Plotter(TVBPlotter, BasePlotter):
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
