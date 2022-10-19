# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
.. moduleauthor:: Gabriel Florea <gabriel.florea@codemart.ro>
"""
import matplotlib
import numpy
from matplotlib import pyplot, gridspec
from matplotlib.colors import Normalize
from six import string_types
from tvb.basic.logger.builder import get_logger
from tvb.datatypes.time_series import TimeSeries
from tvb.simulator.plot.base_plotter import BasePlotter
from tvb.simulator.plot.timeseries_interactive import TimeSeriesInteractivePlotter
from tvb.simulator.plot.power_spectra_interactive import PowerSpectraInteractive
from tvb.simulator.plot.utils import ensure_string, generate_region_labels, ensure_list, isequal_string, \
    time_spectral_analysis, raise_value_error

LOGGER = get_logger(__name__)


def assert_time(time, n_times, time_unit="ms", logger=None):
    if time_unit.find("ms"):
        dt = 0.001
    else:
        dt = 1.0
    try:
        time = numpy.array(time).flatten()
        n_time = len(time)
        if n_time > n_times:
            # self.logger.warning("Input time longer than data time points! Removing redundant tail time points!")
            time = time[:n_times]
        elif n_time < n_times:
            # self.logger.warning("Input time shorter than data time points! "
            #                     "Extending tail time points with the same average time step!")
            if n_time > 1:
                dt = numpy.mean(numpy.diff(time))
            n_extra_points = n_times - n_time
            start_time_point = time[-1] + dt
            end_time_point = start_time_point + n_extra_points * dt
            time = numpy.concatenate([time, numpy.arange(start_time_point, end_time_point, dt)])
    except:
        if logger:
            logger.warning("Setting a default time step vector manually! Input time: " + str(time))
        time = numpy.arange(0, n_times * dt, dt)
    return time


class TimeSeriesPlotter(BasePlotter):
    linestyle = "-"
    linewidth = 1
    marker = None
    markersize = 2
    markerfacecolor = None
    tick_font_size = 12
    print_ts_indices = True

    def __init__(self, config=None):
        super(TimeSeriesPlotter, self).__init__(config)
        self.interactive_plotter = None
        self.print_ts_indices = self.print_regions_indices
        self.HighlightingDataCursor = lambda *args, **kwargs: None
        if matplotlib.get_backend() in matplotlib.rcsetup.interactive_bk and self.config.MOUSE_HOOVER:
            try:
                from mpldatacursor import HighlightingDataCursor
                self.HighlightingDataCursor = HighlightingDataCursor
            except ImportError:
                self.config.MOUSE_HOOVER = False
                # self.logger.warning("Importing mpldatacursor failed! No highlighting functionality in plots!")
        else:
            # self.logger.warning("Noninteractive matplotlib backend! No highlighting functionality in plots!")
            self.config.MOUSE_HOOVER = False

    @property
    def line_format(self):
        return {"linestyle": self.linestyle, "linewidth": self.linewidth,
                "marker": self.marker, "markersize": self.markersize, "markerfacecolor": self.markerfacecolor}

    def _ts_plot(self, time, n_vars, nTS, n_times, time_unit, subplots, offset=0.0, data_lims=[]):

        time_unit = ensure_string(time_unit)
        data_fun = lambda data, time, icol: (data[icol], time, icol)

        def plot_ts(x, iTS, colors, labels):
            x, time, ivar = x
            time = assert_time(time, len(x[:, iTS]), time_unit, self.logger)
            try:
                return pyplot.plot(time, x[:, iTS], color=colors[iTS], label=labels[iTS], **self.line_format)
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(time, x[:, iTS], color=colors[iTS], label=str(iTS), **self.line_format)

        def plot_ts_raster(x, iTS, colors, labels, offset):
            x, time, ivar = x
            time = assert_time(time, len(x[:, iTS]), time_unit, self.logger)
            try:
                return pyplot.plot(time, -x[:, iTS] + (offset * iTS + x[:, iTS].mean()), color=colors[iTS],
                                   label=labels[iTS], **self.line_format)
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(time, -x[:, iTS] + offset * iTS, color=colors[iTS],
                                   label=str(iTS), **self.line_format)

        def axlabels_ts(labels, n_rows, irow, iTS):
            if irow == n_rows:
                pyplot.gca().set_xlabel("Time (" + time_unit + ")")
            if n_rows > 1:
                try:
                    pyplot.gca().set_ylabel(str(iTS) + "." + labels[iTS])
                except:
                    self.logger.warning("Cannot convert labels' strings for y axis labels!")
                    pyplot.gca().set_ylabel(str(iTS))

        def axlimits_ts(data_lims, time, icol):
            pyplot.gca().set_xlim([time[0], time[-1]])
            if n_rows > 1:
                pyplot.gca().set_ylim([data_lims[icol][0], data_lims[icol][1]])
            else:
                pyplot.autoscale(enable=True, axis='y', tight=True)

        def axYticks(labels, nTS, offsets=offset):
            pyplot.gca().set_yticks((offset * numpy.array([list(range(nTS))]).flatten()).tolist())
            try:
                pyplot.gca().set_yticklabels(labels.flatten().tolist())
            except:
                labels = generate_region_labels(nTS, [], "", True)
                self.logger.warning("Cannot convert region labels' strings for y axis ticks!")

        if offset > 0.0:
            plot_lines = lambda x, iTS, colors, labels: \
                plot_ts_raster(x, iTS, colors, labels, offset)
        else:
            plot_lines = lambda x, iTS, colors, labels: \
                plot_ts(x, iTS, colors, labels)
        this_axYticks = lambda labels, nTS: axYticks(labels, nTS, offset)
        if subplots:
            n_rows = nTS
            def_alpha = 1.0
        else:
            n_rows = 1
            def_alpha = 0.5
        subtitle_col = lambda subtitle: pyplot.gca().set_title(subtitle)
        subtitle = lambda iTS, labels: None
        projection = None
        axlabels = lambda labels, vars, n_vars, n_rows, irow, iTS: axlabels_ts(labels, n_rows, irow, iTS)
        axlimits = lambda data_lims, time, n_vars, icol: axlimits_ts(data_lims, time, icol)
        loopfun = lambda nTS, n_rows, icol: list(range(nTS))
        return data_fun, time, plot_lines, projection, n_rows, n_vars, def_alpha, loopfun, \
               subtitle, subtitle_col, axlabels, axlimits, this_axYticks

    def _trajectories_plot(self, n_dims, nTS, nSamples, subplots):
        data_fun = lambda data, time, icol: data

        def plot_traj_2D(x, iTS, colors, labels):
            x, y = x
            try:
                return pyplot.plot(x[:, iTS], y[:, iTS], color=colors[iTS], label=labels[iTS], **self.line_format)
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(x[:, iTS], y[:, iTS], color=colors[iTS], label=str(iTS), **self.line_format)

        def plot_traj_3D(x, iTS, colors, labels):
            x, y, z = x
            try:
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS], color=colors[iTS],
                                   label=labels[iTS], **self.line_format)
            except:
                self.logger.warning("Cannot convert labels' strings for line labels!")
                return pyplot.plot(x[:, iTS], y[:, iTS], z[:, iTS], color=colors[iTS],
                                   label=str(iTS), **self.line_format)

        def subtitle_traj(labels, iTS):
            try:
                if self.print_ts_indices:
                    pyplot.gca().set_title(str(iTS) + "." + labels[iTS])
                else:
                    pyplot.gca().set_title(labels[iTS])
            except:
                self.logger.warning("Cannot convert labels' strings for subplot titles!")
                pyplot.gca().set_title(str(iTS))

        def axlabels_traj(vars, n_vars):
            pyplot.gca().set_xlabel(vars[0])
            pyplot.gca().set_ylabel(vars[1])
            if n_vars > 2:
                pyplot.gca().set_zlabel(vars[2])

        def axlimits_traj(data_lims, n_vars):
            pyplot.gca().set_xlim([data_lims[0][0], data_lims[0][1]])
            pyplot.gca().set_ylim([data_lims[1][0], data_lims[1][1]])
            if n_vars > 2:
                pyplot.gca().set_zlim([data_lims[2][0], data_lims[2][1]])

        if n_dims == 2:
            plot_lines = lambda x, iTS, colors, labels: \
                plot_traj_2D(x, iTS, colors, labels)
            projection = None
        elif n_dims == 3:
            plot_lines = lambda x, iTS, colors, labels: \
                plot_traj_3D(x, iTS, colors, labels)
            projection = '3d'
        else:
            raise_value_error("Data dimensions are neigher 2D nor 3D!, but " + str(n_dims) + "D!", LOGGER)
        n_rows = 1
        n_cols = 1
        if subplots is None:
            # if nSamples > 1:
            n_rows = int(numpy.floor(numpy.sqrt(nTS)))
            n_cols = int(numpy.ceil((1.0 * nTS) / n_rows))
        elif isinstance(subplots, (list, tuple)):
            n_rows = subplots[0]
            n_cols = subplots[1]
            if n_rows * n_cols < nTS:
                raise_value_error("Not enough subplots for all time series:"
                                  "\nn_rows * n_cols = product(subplots) = product(" + str(subplots) + " = "
                                  + str(n_rows * n_cols) + "!", LOGGER)
        if n_rows * n_cols > 1:
            def_alpha = 0.5
            subtitle = lambda labels, iTS: subtitle_traj(labels, iTS)
            subtitle_col = lambda subtitles, icol: None
        else:
            def_alpha = 1.0
            subtitle = lambda labels, iTS: None
            subtitle_col = lambda subtitles, icol: pyplot.gca().set_title(pyplot.gcf().title)
        axlabels = lambda labels, vars, n_vars, n_rows, irow, iTS: axlabels_traj(vars, n_vars)
        axlimits = lambda data_lims, time, n_vars, icol: axlimits_traj(data_lims, n_vars)
        loopfun = lambda nTS, n_rows, icol: list(range(icol, nTS, n_rows))
        return data_fun, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
               subtitle, subtitle_col, axlabels, axlimits

    # TODO: refactor to not have the plot commands here
    def plot_ts(self, data, time=None, var_labels=[], mode="ts", subplots=None, special_idx=[],
                subtitles=[], labels=[], offset=0.5, time_unit="ms",
                title='Time series', figure_name=None, figsize=None):
        if not isinstance(figsize, (list, tuple)):
            figsize = self.config.LARGE_SIZE
        if isinstance(data, dict):
            var_labels = data.keys()
            data = data.values()
        elif isinstance(data, numpy.ndarray):
            if len(data.shape) < 3:
                if len(data.shape) < 2:
                    data = numpy.expand_dims(data, 1)
                data = numpy.expand_dims(data, 2)
                data = [data]
            else:
                # Assuming a structure of Time X Space X Variables X Samples
                data = [data[:, :, iv].squeeze() for iv in range(data.shape[2])]
        elif isinstance(data, (list, tuple)):
            data = ensure_list(data)
        else:
            raise_value_error("Input timeseries: %s \n" "is not on of one of the following types: "
                              "[numpy.ndarray, dict, list, tuple]" % str(data), LOGGER)
        n_vars = len(data)
        data_lims = []
        for id, d in enumerate(data):
            if isequal_string(mode, "raster"):
                data[id] = (d - d.mean(axis=0))
                drange = numpy.max(data[id].max(axis=0) - data[id].min(axis=0))
                data[id] = data[id] / drange  # zscore(d, axis=None)
            data_lims.append([d.min(), d.max()])
        data_shape = data[0].shape
        if len(data_shape) == 1:
            n_times = data_shape[0]
            nTS = 1
            for iV in range(n_vars):
                data[iV] = data[iV][:, numpy.newaxis]
        else:
            n_times, nTS = data_shape[:2]
        if len(data_shape) > 2:
            nSamples = data_shape[2]
        else:
            nSamples = 1
        if special_idx is None:
            special_idx = []
        n_special_idx = len(special_idx)
        if len(subtitles) == 0:
            subtitles = var_labels
        if isinstance(labels, list) and len(labels) == n_vars:
            labels = [generate_region_labels(nTS, label, ". ", self.print_ts_indices) for label in labels]
        else:
            labels = [generate_region_labels(nTS, labels, ". ", self.print_ts_indices) for _ in range(n_vars)]
        if isequal_string(mode, "traj"):
            data_fun, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
            subtitle, subtitle_col, axlabels, axlimits = \
                self._trajectories_plot(n_vars, nTS, nSamples, subplots)
        else:
            if isequal_string(mode, "raster"):
                data_fun, time, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
                subtitle, subtitle_col, axlabels, axlimits, axYticks = \
                    self._ts_plot(time, n_vars, nTS, n_times, time_unit, 0, offset, data_lims)

            else:
                data_fun, time, plot_lines, projection, n_rows, n_cols, def_alpha, loopfun, \
                subtitle, subtitle_col, axlabels, axlimits, axYticks = \
                    self._ts_plot(time, n_vars, nTS, n_times, time_unit, ensure_list(subplots)[0])
        alpha_ratio = 1.0 / nSamples
        alphas = numpy.maximum(numpy.array([def_alpha] * nTS) * alpha_ratio, 0.1)
        alphas[special_idx] = numpy.maximum(alpha_ratio, 0.1)
        if isequal_string(mode, "traj") and (n_cols * n_rows > 1):
            colors = numpy.zeros((nTS, 4))
            colors[special_idx] = \
                numpy.array([numpy.array([1.0, 0, 0, 1.0]) for _ in range(n_special_idx)]).reshape((n_special_idx, 4))
        else:
            cmap = matplotlib.cm.get_cmap('jet')
            colors = numpy.array([cmap(0.5 * iTS / nTS) for iTS in range(nTS)])
            colors[special_idx] = \
                numpy.array([cmap(1.0 - 0.25 * iTS / nTS) for iTS in range(n_special_idx)]).reshape((n_special_idx, 4))
        colors[:, 3] = alphas
        lines = []
        pyplot.figure(title, figsize=figsize)
        axes = []
        for icol in range(n_cols):
            if n_rows == 1:
                # If there are no more rows, create axis, and set its limits, labels and possible subtitle
                axes += ensure_list(pyplot.subplot(n_rows, n_cols, icol + 1, projection=projection))
                axlimits(data_lims, time, n_vars, icol)
                axlabels(labels[icol % n_vars], var_labels, n_vars, n_rows, 1, 0)
                pyplot.gca().set_title(subtitles[icol])
            for iTS in loopfun(nTS, n_rows, icol):
                if n_rows > 1:
                    # If there are more rows, create axes, and set their limits, labels and possible subtitles
                    axes += ensure_list(pyplot.subplot(n_rows, n_cols, iTS + 1, projection=projection))
                    axlimits(data_lims, time, n_vars, icol)
                    subtitle(labels[icol % n_vars], iTS)
                    axlabels(labels[icol % n_vars], var_labels, n_vars, n_rows, (iTS % n_rows) + 1, iTS)
                lines += ensure_list(plot_lines(data_fun(data, time, icol), iTS, colors, labels[icol % n_vars]))
            if isequal_string(mode, "raster"):  # set yticks as labels if this is a raster plot
                axYticks(labels[icol % n_vars], nTS)
                yticklabels = pyplot.gca().yaxis.get_ticklabels()
                self.tick_font_size = numpy.minimum(self.tick_font_size,
                                                    int(numpy.round(self.tick_font_size * 100.0 / nTS)))
                for iTS in range(nTS):
                    yticklabels[iTS].set_fontsize(self.tick_font_size)
                    if iTS in special_idx:
                        yticklabels[iTS].set_color(colors[iTS, :3].tolist() + [1])
                pyplot.gca().yaxis.set_ticklabels(yticklabels)
                pyplot.gca().invert_yaxis()

        if self.config.MOUSE_HOOVER:
            for line in lines:
                self.HighlightingDataCursor(line, formatter='{label}'.format, bbox=dict(fc='white'),
                                            arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))

        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return pyplot.gcf(), axes, lines

    def plot_ts_raster(self, data, time, var_labels=[], time_unit="ms", special_idx=[],
                       title='Raster plot', subtitles=[], labels=[], offset=0.5, figure_name=None, figsize=None):
        if not isinstance(figsize, (list, tuple)):
            figsize = self.config.VERY_LARGE_SIZE
        return self.plot_ts(data, time, var_labels, "raster", None, special_idx, subtitles, labels, offset, time_unit,
                            title, figure_name, figsize)

    def plot_ts_trajectories(self, data, var_labels=[], subtitles=None, special_idx=[], labels=[],
                             title='State space trajectories', figure_name=None, figsize=None):
        if not isinstance(figsize, (list, tuple)):
            figsize = self.config.LARGE_SIZE
        return self.plot_ts(data, [], var_labels, "traj", subtitles, special_idx, labels=labels, title=title,
                            figure_name=figure_name, figsize=figsize)

    def plot_tvb_time_series(self, time_series, mode="ts", subplots=None, special_idx=[], subtitles=[],
                             offset=0.5, title=None, figure_name=None, figsize=None):
        if not isinstance(figsize, (list, tuple)):
            figsize = self.config.LARGE_SIZE
        if title is None:
            title = time_series.title
        variables_labels = time_series.labels_dimensions.get(time_series.labels_ordering[1], [])
        space_labels = time_series.labels_dimensions.get(time_series.labels_ordering[2], [])
        return self.plot_ts(numpy.swapaxes(time_series.data, 1, 2), time_series.time, variables_labels,
                            mode, subplots, special_idx, subtitles, space_labels,
                            offset, time_series.time_unit, title, figure_name, figsize)

    def plot_time_series(self, time_series, mode="ts", subplots=None, special_idx=[], subtitles=[],
                         offset=0.5, title=None, figure_name=None, figsize=None, **kwargs):
        if isinstance(time_series, TimeSeries):
            self.plot_tvb_time_series(time_series, mode, subplots, special_idx,
                                      subtitles, offset, title, figure_name, figsize)
        elif isinstance(time_series, (numpy.ndarray, dict, list, tuple)):
            time = kwargs.get("time", None)
            time_unit = kwargs.get("time_unit", "ms")
            labels = kwargs.get("labels", [])
            var_labels = kwargs.get("var_labels", [])
            if title is None:
                title = "Time Series"
            return self.plot_ts(time_series, time=time, mode=mode, time_unit=time_unit,
                                labels=labels, var_labels=var_labels, subplots=subplots, special_idx=special_idx,
                                subtitles=subtitles, offset=offset, title=title, figure_name=figure_name,
                                figsize=figsize)
        else:
            raise_value_error("Input time_series: %s \n" "is not on of one of the following types: [TimeSeries "
                              "(tvb-contrib), TimeSeries (tvb-library), numpy.ndarray, dict]" % str(time_series),
                              LOGGER)

    def plot_raster(self, time_series, subplots=None, special_idx=[], subtitles=[],
                    offset=0.5, title=None, figure_name=None, figsize=None, **kwargs):
        return self.plot_time_series(time_series, "raster", subplots, special_idx,
                                     subtitles, offset, title, figure_name, figsize, **kwargs)

    def plot_trajectories(self, time_series, subplots=None, special_idx=[], subtitles=[],
                          offset=0.5, title=None, figure_name=None, figsize=None, **kwargs):
        return self.plot_time_series(time_series, "traj", subplots, special_idx,
                                     subtitles, offset, title, figure_name, figsize, **kwargs)

    @staticmethod
    def plot_tvb_time_series_interactive(time_series, first_n=-1, **kwargs):
        interactive_plotter = TimeSeriesInteractivePlotter(time_series=time_series, first_n=first_n)
        interactive_plotter.configure()
        block = kwargs.pop("block", True)
        interactive_plotter.show(block=block, **kwargs)

    def plot_time_series_interactive(self, time_series, first_n=-1, **kwargs):
        if isinstance(time_series, TimeSeries):
            self.plot_tvb_time_series_interactive(time_series, first_n, **kwargs)
        elif isinstance(time_series, numpy.ndarray):
            self.plot_tvb_time_series_interactive(TimeSeries(data=time_series), first_n, **kwargs)
        elif isinstance(time_series, (list, tuple)):
            self.plot_tvb_time_series_interactive(TimeSeries(data=TimeSeries(data=numpy.stack(time_series, axis=1))),
                                                  first_n, **kwargs)
        elif isinstance(time_series, dict):
            ts = numpy.stack(time_series.values(), axis=1)
            time_series = TimeSeries(data=ts, labels_dimensions={"State Variable": time_series.keys()})
            self.plot_tvb_time_series_interactive(time_series, first_n, **kwargs)
        else:
            raise_value_error("Input time_series: %s \n" "is not on of one of the following types: [TimeSeries "
                              "(tvb-contrib), TimeSeriesTVB (tvb-library), numpy.ndarray, dict, list, tuple]" %
                              str(time_series), LOGGER)

    @staticmethod
    def plot_tvb_power_spectra_interactive(time_series, spectral_props, **kwargs):
        interactive_plotters = PowerSpectraInteractive(time_series=time_series, **spectral_props)
        interactive_plotters.configure()
        block = kwargs.pop("block", True)
        interactive_plotters.show(blocl=block, **kwargs)

    def plot_power_spectra_interactive(self, time_series, spectral_props, **kwargs):
        self.plot_tvb_power_spectra_interactive(self, time_series._tvb, spectral_props, **kwargs)

    # TODO: refactor to not have the plot commands here
    def _plot_ts_spectral_analysis_raster(self, data, time=None, var_label="", time_unit="ms",
                                          freq=None, spectral_options={}, special_idx=[], labels=[],
                                          title='Spectral Analysis', figure_name=None, figsize=None):
        if not isinstance(figsize, (list, tuple)):
            figsize = self.config.VERY_LARGE_SIZE
        if len(data.shape) == 1:
            n_times = data.shape[0]
            nS = 1
        else:
            n_times, nS = data.shape[:2]
        time = assert_time(time, n_times, time_unit, self.logger)
        if not isinstance(time_unit, string_types):
            time_unit = list(time_unit)[0]
        time_unit = ensure_string(time_unit)
        if time_unit in ("ms", "msec"):
            fs = 1000.0
        else:
            fs = 1.0
        fs = fs / numpy.mean(numpy.diff(time))
        n_special_idx = len(special_idx)
        if n_special_idx > 0:
            data = data[:, special_idx]
            nS = data.shape[1]
            if len(labels) > n_special_idx:
                labels = numpy.array([str(ilbl) + ". " + str(labels[ilbl]) for ilbl in special_idx])
            elif len(labels) == n_special_idx:
                labels = numpy.array([str(ilbl) + ". " + str(label) for ilbl, label in zip(special_idx, labels)])
            else:
                labels = numpy.array([str(ilbl) for ilbl in special_idx])
        else:
            if len(labels) != nS:
                labels = numpy.array([str(ilbl) for ilbl in range(nS)])
        if nS > 20:
            LOGGER.warning("It is not possible to plot spectral analysis plots for more than 20 signals!")
            return

        log_norm = spectral_options.get("log_norm", False)
        mode = spectral_options.get("mode", "psd")
        psd_label = mode
        if log_norm:
            psd_label = "log" + psd_label
        stf, time, freq, psd = time_spectral_analysis(data, fs,
                                                      freq=freq,
                                                      mode=mode,
                                                      nfft=spectral_options.get("nfft"),
                                                      window=spectral_options.get("window", 'hanning'),
                                                      nperseg=spectral_options.get("nperseg", int(numpy.round(fs / 4))),
                                                      detrend=spectral_options.get("detrend", 'constant'),
                                                      noverlap=spectral_options.get("noverlap"),
                                                      f_low=spectral_options.get("f_low", 10.0),
                                                      log_scale=spectral_options.get("log_scale", False))
        min_val = numpy.min(stf.flatten())
        max_val = numpy.max(stf.flatten())
        if nS > 2:
            figsize = self.config.VERY_LARGE_SIZE
        if len(var_label):
            title += ": " % var_label
        fig = pyplot.figure(title, figsize=figsize)
        fig.suptitle(title)
        gs = gridspec.GridSpec(nS, 23)
        ax = numpy.empty((nS, 2), dtype="O")
        img = numpy.empty((nS,), dtype="O")
        line = numpy.empty((nS,), dtype="O")
        for iS in range(nS, -1, -1):
            if iS < nS - 1:
                ax[iS, 0] = pyplot.subplot(gs[iS, :20], sharex=ax[iS, 0])
                ax[iS, 1] = pyplot.subplot(gs[iS, 20:22], sharex=ax[iS, 1], sharey=ax[iS, 0])
            else:
                # TODO: find and correct bug here
                ax[iS, 0] = pyplot.subplot(gs[iS, :20])
                ax[iS, 1] = pyplot.subplot(gs[iS, 20:22], sharey=ax[iS, 0])
            img[iS] = ax[iS, 0].imshow(numpy.squeeze(stf[:, :, iS]).T, cmap=pyplot.set_cmap('jet'),
                                       interpolation='none',
                                       norm=Normalize(vmin=min_val, vmax=max_val), aspect='auto', origin='lower',
                                       extent=(time.min(), time.max(), freq.min(), freq.max()))
            # img[iS].clim(min_val, max_val)
            ax[iS, 0].set_title(labels[iS])
            ax[iS, 0].set_ylabel("Frequency (Hz)")
            line[iS] = ax[iS, 1].plot(psd[:, iS], freq, 'k', label=labels[iS])
            pyplot.setp(ax[iS, 1].get_yticklabels(), visible=False)
            # ax[iS, 1].yaxis.tick_right()
            # ax[iS, 1].yaxis.set_ticks_position('both')
            if iS == (nS - 1):
                ax[iS, 0].set_xlabel("Time (" + time_unit + ")")

                ax[iS, 1].set_xlabel(psd_label)
            else:
                pyplot.setp(ax[iS, 0].get_xticklabels(), visible=False)
            pyplot.setp(ax[iS, 1].get_xticklabels(), visible=False)
            ax[iS, 0].autoscale(tight=True)
            ax[iS, 1].autoscale(tight=True)
        # make a color bar
        cax = pyplot.subplot(gs[:, 22])
        pyplot.colorbar(img[0], cax=pyplot.subplot(gs[:, 22]))  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
        cax.set_title(psd_label)
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return fig, ax, img, line, time, freq, stf, psd

    def plot_ts_spectral_analysis_raster(self, data, time=None, time_unit="ms", freq=None, spectral_options={},
                                         special_idx=[], labels=[], title='Spectral Analysis', figure_name=None,
                                         figsize=None):
        if isinstance(data, dict):
            var_labels = data.keys()
            data = data.values()
        else:
            var_labels = []
            if isinstance(data, (list, tuple)):
                data = data[0]
            elif isinstance(data, numpy.ndarray) and data.ndim > 2:
                # Assuming a structure of Time X Space X Variables X Samples
                if data.ndim > 3:
                    data = data[:, :, :, 0]
                data = [data[:, :, iv].squeeze() for iv in range(data.shape[2])]
        if len(var_labels) == 0:
            var_labels = [""] * len(data)
        for d, var_label in zip(data, var_labels):
            self._plot_ts_spectral_analysis_raster(d, time, var_label, time_unit, freq, spectral_options,
                                                   special_idx, labels, title, figure_name, figsize)

    def plot_spectral_analysis_raster(self, time_series, freq=None, spectral_options={},
                                      special_idx=[], labels=[], title='Spectral Analysis', figure_name=None,
                                      figsize=None, **kwargs):
        if isinstance(time_series, TimeSeries):
            return self.plot_ts_spectral_analysis_raster(numpy.swapaxes(time_series.data, 1, 2).squeeze(),
                                                         time_series.time, time_series.time_unit, freq,
                                                         spectral_options,
                                                         special_idx, labels, title, figure_name, figsize)
        elif isinstance(time_series, (numpy.ndarray, dict, list, tuple)):
            time = kwargs.get("time", None)
            return self.plot_ts_spectral_analysis_raster(time_series, time=time, freq=freq,
                                                         spectral_options=spectral_options, special_idx=special_idx,
                                                         labels=labels, title=title, figure_name=figure_name,
                                                         figsize=figsize)
        else:
            raise_value_error("Input time_series: %s \n"
                              "is not on of one of the following types: [TimeSeries (tvb-contrib), "
                              "TimeSeries (tvb-library), numpy.ndarray, dict]" % str(time_series),
                              LOGGER)
