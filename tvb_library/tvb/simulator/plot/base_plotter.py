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

import os
from itertools import zip_longest

import matplotlib
import numpy
import tvb.simulator.plot.tools as TVB_plot_tools
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tvb.basic.logger.builder import get_logger
from tvb.simulator.plot.config import CONFIGURED
from tvb.simulator.plot.utils import generate_region_labels, ensure_list


class BasePlotter(object):

    def __init__(self, config=CONFIGURED):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.print_regions_indices = True
        matplotlib.use(self.config.MATPLOTLIB_BACKEND)
        pyplot.rcParams["font.size"] = self.config.FONTSIZE

    def _check_show(self):
        if self.config.SHOW_FLAG:
            # mp.use('TkAgg')
            pyplot.ion()
            pyplot.show()
        else:
            # mp.use('Agg')
            pyplot.ioff()
            pyplot.close()

    @staticmethod
    def _figure_filename(fig=pyplot.gcf(), figure_name=None):
        if figure_name is None:
            figure_name = fig.get_label()
        figure_name = figure_name.replace(": ", "_").replace(" ", "_").replace("\t", "_").replace(",", "")
        return figure_name

    def _save_figure(self, fig=pyplot.gcf(), figure_name=None):
        if self.config.SAVE_FLAG:
            figure_name = self._figure_filename(fig, figure_name)
            figure_name = figure_name[:numpy.min([100, len(figure_name)])] + '.' + self.config.FIG_FORMAT
            figure_dir = self.config.FOLDER_FIGURES
            if not (os.path.isdir(figure_dir)):
                os.mkdir(figure_dir)
            pyplot.savefig(os.path.join(figure_dir, figure_name))

    @staticmethod
    def rect_subplot_shape(n, mode="col"):
        nj = int(numpy.ceil(numpy.sqrt(n)))
        ni = int(numpy.ceil(1.0 * n / nj))
        if mode.find("row") >= 0:
            return nj, ni
        else:
            return ni, nj

    def plot_vector(self, vector, labels, subplot, title, show_y_labels=True, indices_red=None, sharey=None):
        ax = pyplot.subplot(subplot, sharey=sharey)
        pyplot.title(title)
        n_vector = labels.shape[0]
        y_ticks = numpy.array(range(n_vector), dtype=numpy.int32)
        color = 'k'
        colors = numpy.repeat([color], n_vector)
        coldif = False
        if indices_red is not None:
            colors[indices_red] = 'r'
            coldif = True
        if len(vector.shape) == 1:
            ax.barh(y_ticks, vector, color=colors, align='center')
        else:
            ax.barh(y_ticks, vector[0, :], color=colors, align='center')
        # ax.invert_yaxis()
        ax.grid(True, color='grey')
        ax.set_yticks(y_ticks)
        if show_y_labels:
            region_labels = generate_region_labels(n_vector, labels, ". ", self.print_regions_indices)
            ax.set_yticklabels(region_labels)
            if coldif:
                labels = ax.yaxis.get_ticklabels()
                for ids in indices_red:
                    labels[ids].set_color('r')
                ax.yaxis.set_ticklabels(labels)
        else:
            ax.set_yticklabels([])
        ax.autoscale(tight=True)
        if sharey is None:
            ax.invert_yaxis()
        return ax

    def plot_vector_violin(self, dataset, vector=[], lines=[], labels=[], subplot=111, title="", violin_flag=True,
                           colormap="YlOrRd", show_y_labels=True, indices_red=None, sharey=None):
        ax = pyplot.subplot(subplot, sharey=sharey)
        pyplot.title(title)
        n_violins = dataset.shape[1]
        y_ticks = numpy.array(range(n_violins), dtype=numpy.int32)
        # the vector plot
        coldif = False
        if indices_red is None:
            indices_red = []
        if violin_flag:
            # the violin plot
            colormap = matplotlib.cm.ScalarMappable(cmap=pyplot.set_cmap(colormap))
            colormap = colormap.to_rgba(numpy.mean(dataset, axis=0), alpha=0.75)
            violin_parts = ax.violinplot(dataset, y_ticks, vert=False, widths=0.9,
                                         showmeans=True, showmedians=True, showextrema=True)
            violin_parts['cmeans'].set_color("k")
            violin_parts['cmins'].set_color("b")
            violin_parts['cmaxes'].set_color("b")
            violin_parts['cbars'].set_color("b")
            violin_parts['cmedians'].set_color("b")
            for ii in range(len(violin_parts['bodies'])):
                violin_parts['bodies'][ii].set_color(numpy.reshape(colormap[ii], (1, 4)))
                violin_parts['bodies'][ii]._alpha = 0.75
                violin_parts['bodies'][ii]._edgecolors = numpy.reshape(colormap[ii], (1, 4))
                violin_parts['bodies'][ii]._facecolors = numpy.reshape(colormap[ii], (1, 4))
        else:
            colorcycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
            n_samples = dataset.shape[0]
            for ii in range(n_violins):
                for jj in range(n_samples):
                    ax.plot(dataset[jj, ii], y_ticks[ii], "D",
                            mfc=colorcycle[jj % n_samples], mec=colorcycle[jj % n_samples], ms=20)
        color = 'k'
        colors = numpy.repeat([color], n_violins)
        if indices_red is not None:
            colors[indices_red] = 'r'
            coldif = True
        if len(vector) == n_violins:
            for ii in range(n_violins):
                ax.plot(vector[ii], y_ticks[ii], '*', mfc=colors[ii], mec=colors[ii], ms=10)
        if len(lines) == 2 and lines[0].shape[0] == n_violins and lines[1].shape[0] == n_violins:
            for ii in range(n_violins):
                yy = (y_ticks[ii] - 0.45 * lines[1][ii] / numpy.max(lines[1][ii])) \
                     * numpy.ones(numpy.array(lines[0][ii]).shape)
                ax.plot(lines[0][ii], yy, '--', color=colors[ii])

        ax.grid(True, color='grey')
        ax.set_yticks(y_ticks)
        if show_y_labels:
            region_labels = generate_region_labels(n_violins, labels, ". ", self.print_regions_indices)
            ax.set_yticklabels(region_labels)
            if coldif:
                labels = ax.yaxis.get_ticklabels()
                for ids in indices_red:
                    labels[ids].set_color('r')
                ax.yaxis.set_ticklabels(labels)
        else:
            ax.set_yticklabels([])
        if sharey is None:
            ax.invert_yaxis()
        ax.autoscale()
        return ax

    def _plot_matrix(self, matrix, xlabels, ylabels, subplot=111, title="", show_x_labels=True, show_y_labels=True,
                     x_ticks=numpy.array([]), y_ticks=numpy.array([]), indices_red_x=None, indices_red_y=None,
                     sharex=None, sharey=None, cmap='autumn_r', vmin=None, vmax=None):
        ax = pyplot.subplot(subplot, sharex=sharex, sharey=sharey)
        pyplot.title(title)
        nx, ny = matrix.shape
        indices_red = [indices_red_x, indices_red_y]
        ticks = [x_ticks, y_ticks]
        labels = [xlabels, ylabels]
        nticks = []
        for ii, (n, tick) in enumerate(zip([nx, ny], ticks)):
            if len(tick) == 0:
                ticks[ii] = numpy.array(range(n), dtype=numpy.int32)
            nticks.append(len(ticks[ii]))
        cmap = pyplot.set_cmap(cmap)
        img = pyplot.imshow(matrix[ticks[0]][:, ticks[1]].T, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
        pyplot.grid(True, color='black')
        for ii, (xy, tick, ntick, ind_red, show, lbls, rot) in enumerate(zip(["x", "y"], ticks, nticks, indices_red,
                                                                             [show_x_labels, show_y_labels], labels,
                                                                             [90, 0])):
            if show:
                labels[ii] = generate_region_labels(len(tick), numpy.array(lbls)[tick], ". ",
                                                    self.print_regions_indices, tick)
                # labels[ii] = numpy.array(["%d. %s" % l for l in zip(tick, lbls[tick])])
                getattr(pyplot, xy + "ticks")(numpy.array(range(ntick)), labels[ii], rotation=rot)
            else:
                labels[ii] = numpy.array(["%d." % l for l in tick])
                getattr(pyplot, xy + "ticks")(numpy.array(range(ntick)), labels[ii])
            if ind_red is not None:
                tck = tick.tolist()
                ticklabels = getattr(ax, xy + "axis").get_ticklabels()
                for iidx, indr in enumerate(ind_red):
                    try:
                        ticklabels[tck.index(indr)].set_color('r')
                    except:
                        pass
                getattr(ax, xy + "axis").set_ticklabels(ticklabels)
        ax.autoscale(tight=True)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
        return ax, cax1

    def plot_regions2regions(self, adj, labels, subplot, title, show_x_labels=True, show_y_labels=True,
                             x_ticks=numpy.array([]), y_ticks=numpy.array([]), indices_red_x=None, indices_red_y=None,
                             sharex=None, sharey=None, cmap='autumn_r', vmin=None, vmax=None):
        return self._plot_matrix(adj, labels, labels, subplot, title, show_x_labels, show_y_labels,
                                 x_ticks, y_ticks, indices_red_x, indices_red_y, sharex, sharey, cmap, vmin, vmax)

    def _set_axis_labels(self, fig, sub, n_regions, region_labels, indices2emphasize, color='k', position='left'):
        y_ticks = range(n_regions)
        # region_labels = numpy.array(["%d. %s" % l for l in zip(y_ticks, region_labels)])
        region_labels = generate_region_labels(len(y_ticks), region_labels, ". ", self.print_regions_indices, y_ticks)
        big_ax = fig.add_subplot(sub, frameon=False)
        if position == 'right':
            big_ax.yaxis.tick_right()
            big_ax.yaxis.set_label_position("right")
        big_ax.set_yticks(y_ticks)
        big_ax.set_yticklabels(region_labels, color='k')
        if not (color == 'k'):
            labels = big_ax.yaxis.get_ticklabels()
            for idx in indices2emphasize:
                labels[idx].set_color(color)
            big_ax.yaxis.set_ticklabels(labels)
        big_ax.invert_yaxis()
        big_ax.axes.get_xaxis().set_visible(False)
        # TODO: find out what is the next line about and why it fails...
        # big_ax.axes.set_facecolor('none')

    def plot_in_columns(self, data_dict_list, labels, width_ratios=[], left_ax_focus_indices=[],
                        right_ax_focus_indices=[], description="", title="", figure_name=None,
                        figsize=None, **kwargs):
        if not isinstance(figsize, (tuple, list)):
            figsize = self.config.VERY_LARGE_SIZE
        fig = pyplot.figure(title, frameon=False, figsize=figsize)
        fig.suptitle(description)
        n_subplots = len(data_dict_list)
        if not width_ratios:
            width_ratios = numpy.ones((n_subplots,)).tolist()
        matplotlib.gridspec.GridSpec(1, n_subplots, width_ratios=width_ratios)
        if 10 > n_subplots > 0:
            subplot_ind0 = 100 + 10 * n_subplots
        else:
            raise ValueError("\nSubplots' number " + str(n_subplots) + " is not between 1 and 9!")
        n_regions = len(labels)
        subplot_ind = subplot_ind0
        ax = None
        ax0 = None
        for iS, data_dict in enumerate(data_dict_list):
            subplot_ind += 1
            data = data_dict["data"]
            focus_indices = data_dict.get("focus_indices")
            if subplot_ind == 0:
                if not left_ax_focus_indices:
                    left_ax_focus_indices = focus_indices
            else:
                ax0 = ax
            if data_dict.get("plot_type") == "vector_violin":
                ax = self.plot_vector_violin(data_dict.get("data_samples", []), data, [],
                                             labels, subplot_ind, data_dict["name"],
                                             colormap=kwargs.get("colormap", "YlOrRd"), show_y_labels=False,
                                             indices_red=focus_indices, sharey=ax0)
            elif data_dict.get("plot_type") == "regions2regions":
                # TODO: find a more general solution, in case we don't want to apply focus indices to x_ticks
                ax = self.plot_regions2regions(data, labels, subplot_ind, data_dict["name"], x_ticks=focus_indices,
                                               show_x_labels=True, show_y_labels=False, indices_red_x=focus_indices,
                                               sharey=ax0)
            else:
                ax = self.plot_vector(data, labels, subplot_ind, data_dict["name"], show_y_labels=False,
                                      indices_red=focus_indices, sharey=ax0)
        if right_ax_focus_indices == []:
            right_ax_focus_indices = focus_indices
        self._set_axis_labels(fig, 121, n_regions, labels, left_ax_focus_indices, 'r')
        self._set_axis_labels(fig, 122, n_regions, labels, right_ax_focus_indices, 'r', 'right')
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return fig

    # TODO: name is too generic
    def plots(self, data_dict, shape=None, transpose=False, skip=0, xlabels={}, xscales={}, yscales={}, title='Plots',
              lgnd={}, figure_name=None, figsize=None):
        if not isinstance(figsize, (tuple, list)):
            figsize = self.config.VERY_LARGE_SIZE
        if shape is None:
            shape = (1, len(data_dict))
        fig, axes = pyplot.subplots(shape[0], shape[1], figsize=figsize)
        fig.set_label(title)
        for i, key in enumerate(data_dict.keys()):
            ind = numpy.unravel_index(i, shape)
            if transpose:
                axes[ind].plot(data_dict[key].T[skip:])
            else:
                axes[ind].plot(data_dict[key][skip:])
            axes[ind].set_xscale(xscales.get(key, "linear"))
            axes[ind].set_yscale(yscales.get(key, "linear"))
            axes[ind].set_xlabel(xlabels.get(key, ""))
            axes[ind].set_ylabel(key)
            this_legend = lgnd.get(key, None)
            if this_legend is not None:
                axes[ind].legend(this_legend)
        fig.tight_layout()
        self._save_figure(fig, figure_name)
        self._check_show()
        return fig, axes

    def pair_plots(self, data, keys, diagonal_plots={}, transpose=False, skip=0,
                   title='Pair plots', legend_prefix="", subtitles=None, figure_name=None, figsize=None):

        def confirm_y_coordinate(data, ymax):
            data = list(data)
            data.append(ymax)
            return tuple(data)

        if not isinstance(figsize, (tuple, list)):
            figsize = self.config.VERY_LARGE_SIZE

        if subtitles is None:
            subtitles = keys
        data = ensure_list(data)
        n = len(keys)
        fig, axes = pyplot.subplots(n, n, figsize=figsize)
        fig.set_label(title)
        colorcycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
        for i, key_i in enumerate(keys):
            for j, key_j in enumerate(keys):
                for datai in data:
                    if transpose:
                        di = datai[key_i].T[skip:]
                    else:
                        di = datai[key_i][skip:]
                    if i == j:
                        if di.shape[0] > 1:
                            hist_data = axes[i, j].hist(di, int(numpy.round(numpy.sqrt(len(di)))), log=True)[0]
                            if i == 0 and len(di.shape) > 1 and di.shape[1] > 1:
                                axes[i, j].legend([legend_prefix + str(ii + 1) for ii in range(di.shape[1])])
                            y_max = numpy.array(hist_data).max()
                            # The mean line
                            axes[i, j].vlines(di.mean(axis=0), 0, y_max, color=colorcycle, linestyle='dashed',
                                              linewidth=1)
                        else:
                            # This is for the case of only 1 sample (optimization)
                            y_max = 1.0
                            for ii in range(di.shape[1]):
                                axes[i, j].plot(di[0, ii], y_max, "D", color=colorcycle[ii % di.shape[1]],
                                                markersize=20,
                                                label=legend_prefix + str(ii + 1))
                            if i == 0 and len(di.shape) > 1 and di.shape[1] > 1:
                                axes[i, j].legend()
                        # Plot a line (or marker) in the same axis
                        diag_line_plot = ensure_list(diagonal_plots.get(key_i, ((), ()))[0])
                        if len(diag_line_plot) in [1, 2]:
                            if len(diag_line_plot) == 1:
                                diag_line_plot = confirm_y_coordinate(diag_line_plot, y_max)
                            else:
                                diag_line_plot[1] = diag_line_plot[1] / numpy.max(diag_line_plot[1]) * y_max
                            if len(ensure_list(diag_line_plot[0])) == 1:
                                axes[i, j].plot(diag_line_plot[0], diag_line_plot[1], "o", mfc="k", mec="k",
                                                markersize=10)
                            else:
                                axes[i, j].plot(diag_line_plot[0], diag_line_plot[1], color='k',
                                                linestyle="dashed", linewidth=1)
                        # Plot a marker in the same axis
                        diag_marker_plot = ensure_list(diagonal_plots.get(key_i, ((), ()))[1])
                        if len(diag_marker_plot) in [1, 2]:
                            if len(diag_marker_plot) == 1:
                                diag_marker_plot = confirm_y_coordinate(diag_marker_plot, y_max)
                            axes[i, j].plot(diag_marker_plot[0], diag_marker_plot[1], "*", color='k', markersize=10)
                        axes[i, j].autoscale()
                        axes[i, j].set_ylim([0, 1.1 * y_max])

                    else:
                        if transpose:
                            dj = datai[key_j].T[skip:]
                        else:
                            dj = datai[key_j][skip:]
                        axes[i, j].plot(dj, di, '.')
                if i == 0:
                    axes[i, j].set_title(subtitles[j])
                if j == 0:
                    axes[i, j].set_ylabel(key_i)
        fig.tight_layout()
        self._save_figure(fig, figure_name)
        self._check_show()
        return fig, axes

    def plot_bars(self, data, ax=None, fig=None, title="", group_names=[], legend_prefix="", figsize=None):

        def barlabel(ax, rects, positions):
            """
            Attach a text label on each bar displaying its height
            """
            for rect, pos in zip(rects, positions):
                height = rect.get_height()
                if pos < 0:
                    y = -height
                    pos = 0.75 * pos
                else:
                    y = height
                    pos = 0.25 * pos
                ax.text(rect.get_x() + rect.get_width() / 2., pos, '%0.2f' % y,
                        color="k", ha='center', va='bottom', rotation=90)

        if fig is None:
            if not isinstance(figsize, (tuple, list)):
                figsize = self.config.VERY_LARGE_SIZE
            fig, ax = pyplot.subplots(1, 1, figsize=figsize)
            show_and_save = True
        else:
            show_and_save = False
            if ax is None:
                ax = pyplot.gca()
        if isinstance(data, (list, tuple)):  # If, there are many groups, data is a list:
            # Fill in with nan in case that not all groups have the same number of elements
            data = numpy.array(list(zip_longest(*ensure_list(data), fillvalue=numpy.nan))).T
        elif data.ndim == 1:  # This is the case where there is only one group...
            data = numpy.expand_dims(data, axis=1).T
        n_groups, n_elements = data.shape
        posmax = numpy.nanmax(data)
        negmax = numpy.nanmax(-(-data))
        n_groups_names = len(group_names)
        if n_groups_names != n_groups:
            if n_groups_names != 0:
                self.logger.warning("Ignoring group_names because their number (" + str(n_groups_names) +
                                    ") is not equal to the number of groups (" + str(n_groups) + ")!")
            group_names = n_groups * [""]
        colorcycle = pyplot.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = len(colorcycle)
        x_inds = numpy.arange(n_groups)
        width = 0.9 / n_elements
        elements = []
        for iE in range(n_elements):
            elements.append(ax.bar(x_inds + iE * width, data[:, iE], width, color=colorcycle[iE % n_colors]))
            positions = numpy.array([negmax if d < 0 else posmax for d in data[:, iE]])
            positions[numpy.logical_or(numpy.isnan(positions), numpy.isinf(numpy.abs(positions)))] = 0.0
            barlabel(ax, elements[-1], positions)
        if n_elements > 1:
            legend = [legend_prefix + str(ii) for ii in range(1, n_elements + 1)]
            ax.legend(tuple([element[0] for element in elements]), tuple(legend))
        ax.set_xticks(x_inds + n_elements * width / 2)
        ax.set_xticklabels(tuple(group_names))
        ax.set_title(title)
        ax.autoscale()  # tight=True
        ax.set_xlim([-1.05 * width, n_groups * 1.05])
        if show_and_save:
            fig.tight_layout()
            self._save_figure(fig)
            self._check_show()
        return fig, ax

    def tvb_plot(self, plot_fun_name, *args, **kwargs):
        getattr(TVB_plot_tools, plot_fun_name)(*args, **kwargs)
        fig = pyplot.gcf()
        self._save_figure(fig)
        self._check_show()
        return fig
