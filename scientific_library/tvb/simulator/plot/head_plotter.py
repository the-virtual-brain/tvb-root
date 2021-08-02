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

import numpy
from matplotlib import pyplot
from tvb.datatypes.projections import ProjectionMatrix
from tvb.datatypes.sensors import Sensors
from tvb.simulator.plot.base_plotter import BasePlotter
from tvb.simulator.plot.utils import compute_in_degree


class HeadPlotter(BasePlotter):

    def __init__(self, config=None):
        super(HeadPlotter, self).__init__(config)

    def _plot_connectivity(self, connectivity, figure_name='Connectivity'):
        pyplot.figure(figure_name + str(connectivity.number_of_regions), self.config.VERY_LARGE_SIZE)
        axes = []
        axes.append(self.plot_regions2regions(connectivity.weights,
                                              connectivity.region_labels, 121, "weights"))
        axes.append(self.plot_regions2regions(connectivity.tract_lengths,
                                              connectivity.region_labels, 122, "tract lengths"))
        self._save_figure(None, figure_name.replace(" ", "_").replace("\t", "_"))
        self._check_show()
        return pyplot.gcf(), tuple(axes)

    def _plot_connectivity_stats(self, connectivity, figsize=None, figure_name='HeadStats '):
        if not isinstance(figsize, (list, tuple)):
            figsize = self.config.VERY_LARGE_SIZE
        pyplot.figure("Head stats " + str(connectivity.number_of_regions), figsize=figsize)
        areas_flag = len(connectivity.areas) == len(connectivity.region_labels)
        axes = []
        axes.append(self.plot_vector(compute_in_degree(connectivity.weights), connectivity.region_labels,
                                     111 + 10 * areas_flag, "w in-degree"))
        if len(connectivity.areas) == len(connectivity.region_labels):
            axes.append(self.plot_vector(connectivity.areas, connectivity.region_labels, 122, "region areas"))
        self._save_figure(None, figure_name.replace(" ", "").replace("\t", ""))
        self._check_show()
        return pyplot.gcf(), tuple(axes)

    def _plot_sensors(self, sensors, projection, region_labels):
        figure, ax, cax = self._plot_projection(sensors, projection, region_labels,
                                                title="%s - Projection" % (sensors.sensors_type))
        return figure, ax, cax

    def _plot_projection(self, sensors, projection, region_labels, figure=None, title="Projection",
                         show_x_labels=True, show_y_labels=True, x_ticks=numpy.array([]), y_ticks=numpy.array([]),
                         figsize=None):
        if not isinstance(figsize, (list, tuple)):
            figsize = self.config.VERY_LARGE_SIZE
        if not (isinstance(figure, pyplot.Figure)):
            figure = pyplot.figure(title, figsize=figsize)
        ax, cax1 = self._plot_matrix(projection, sensors.labels, region_labels, 111, title,
                                     show_x_labels, show_y_labels, x_ticks, y_ticks)
        self._save_figure(None, title)
        self._check_show()
        return figure, ax, cax1

    def plot_head(self, connectivity, sensors_set, plot_stats=False, plot_sensors=True):
        output = []
        output.append(self._plot_connectivity(connectivity))
        if plot_stats:
            output.append(self._plot_connectivity_stats(connectivity))
        if plot_sensors:
            for sensor, projection in sensors_set.items():
                if isinstance(sensor, Sensors) and isinstance(projection, ProjectionMatrix):
                    figure, ax, cax = self._plot_sensors(sensor, projection.projection_data,
                                                         connectivity.region_labels)
                    output.append((figure, ax, cax))
        return tuple(output)

    def plot_tvb_connectivity(self, connectivity, num="weights", order_by=None, plot_hinton=False, plot_tracts=True,
                              **kwargs):
        """
        A 2D plot for visualizing the Connectivity.weights matrix
        """
        figsize = kwargs.pop("figsize", self.config.LARGE_SIZE)
        fontsize = kwargs.pop("fontsize", self.config.SMALL_FONTSIZE)

        labels = connectivity.region_labels
        plot_title = connectivity.__class__.__name__

        if order_by is None:
            order = numpy.arange(connectivity.number_of_regions)
        else:
            order = numpy.argsort(order_by)
            if order.shape[0] != connectivity.number_of_regions:
                self.logger.error("Ordering vector doesn't have length number_of_regions")
                self.logger.error("Check ordering length and that connectivity is configured")
                return

        # Assumes order is shape (number_of_regions, )
        order_rows = order[:, numpy.newaxis]
        order_columns = order_rows.T

        if plot_hinton:
            from tvb.simulator.plot.tools import hinton_diagram
            weights_axes = hinton_diagram(connectivity.weights[order_rows, order_columns], num)
            weights_figure = None
        else:
            # weights matrix
            weights_figure = pyplot.figure(num="Connectivity weights", figsize=figsize)
            weights_axes = weights_figure.gca()
            wimg = weights_axes.matshow(connectivity.weights[order_rows, order_columns])
            weights_figure.colorbar(wimg)

        weights_axes.set_title(plot_title)

        weights_axes.set_yticks(numpy.arange(connectivity.number_of_regions))
        weights_axes.set_yticklabels(list(labels[order]), fontsize=fontsize)

        weights_axes.set_xticks(numpy.arange(connectivity.number_of_regions))
        weights_axes.set_xticklabels(list(labels[order]), fontsize=fontsize, rotation=90)

        self._save_figure(weights_figure, plot_title)
        self._check_show()

        if plot_tracts:
            # tract lengths matrix
            tracts_figure = pyplot.figure(num="Tracts' lengths", figsize=figsize)
            tracts_axes = tracts_figure.gca()
            timg = tracts_axes.matshow(connectivity.tract_lengths[order_rows, order_columns])
            tracts_axes.set_title("Tracts' lengths")
            tracts_figure.colorbar(timg)
            tracts_axes.set_yticks(numpy.arange(connectivity.number_of_regions))
            tracts_axes.set_yticklabels(list(labels[order]), fontsize=fontsize)

            tracts_axes.set_xticks(numpy.arange(connectivity.number_of_regions))
            tracts_axes.set_xticklabels(list(labels[order]), fontsize=fontsize, rotation=90)

            self._save_figure(tracts_figure)
            self._check_show()
            return weights_figure, weights_axes, tracts_figure, tracts_axes

        else:
            return weights_figure, weights_axes
