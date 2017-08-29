# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Dan Pop <dan.pop@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import numpy
import json
from scipy.optimize import leastsq
from scipy.interpolate import griddata
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.adapters.exceptions import LaunchException
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.basic.filters.chain import FilterChain


class BaseTopography():
    """
    Base class for topographic visualizers.
    """
    # the following fields aren't changed from GUI
    plotsensors = False
    plothead = True
    masked = True

    # dictionaries that contains processed data
    head_contour = None
    sensor_locations = None
    topography_data = None

    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        # Don't know how much memory is needed.
        return -1

    def init_topography(self, sensor_locations):
        """
        Initialize entities for topographic computation.
        """
        self.topography_data = self.prepare_sensors(sensor_locations)

    def _get_topography_array(self, topography, topography_data):
        """
        Trim data, to make sure everything is inside the head contour.
        """
        x_arr = topography_data["x_arr"]
        y_arr = topography_data["y_arr"]
        try:
            points = numpy.vstack((topography_data["sproj"][:, 0], topography_data["sproj"][:, 1])).T
            topo = griddata(points, numpy.ravel(numpy.array(topography)), (x_arr, y_arr), method='linear')
            topo = self.extend_with_nans(topo)
            topo = self.spiral(topo)
            topo = self.remove_outer_nans(topo)
        except KeyError as err:
            self.log.exception("Could not execute matplotlib.mlab.griddata...")
            raise LaunchException("The measure points location is not compatible with this viewer "
                                  "(maybe all on one line)!", err)
        return topo

    @staticmethod
    def fit_circle(data_matrix):

        nx, ny = data_matrix.shape
        radius = nx / 2
        y, x = numpy.ogrid[-nx / 2:nx - nx / 2, -ny / 2:ny - ny / 2]
        mask = x * x + y * y >= radius * radius
        data_matrix[mask] = -1
        return data_matrix

    def spiral(self, array):
        X = array.shape[0] - 2
        Y = array.shape[1] - 2
        r = X / 2
        x = y = 0
        dx = 0
        dy = -1
        nx = [-1, -1, -1, 0, 0, 1, 1, 1]
        ny = [-1, 0, 1, -1, 1, -1, 0, 1]
        for i in range(max(X, Y) ** 2):
            if (-X / 2 < x <= X / 2) and (-Y / 2 < y <= Y / 2):
                if (numpy.isnan(array[x + r][y + r])):
                    neighbors = []
                    for j in range(0, 8):
                        neighbors.append(array[x + r + nx[j]][y + r + ny[j]])
                    array[x + r][y + r] = self.compute_avg(neighbors)
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            x, y = x + dx, y + dy
        return array

    @staticmethod
    def compute_avg(array):
        sum = 0
        dimension = 0
        for x in array:
            if (not numpy.isnan(x)):
                sum += x
                dimension += 1

        return sum / float(dimension)

    def extend_with_nans(self, data_matrix):
        n = data_matrix.shape[0]
        extended = numpy.empty((n + 2, n + 2,))
        extended[:] = numpy.nan
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                extended[i][j] = data_matrix[i - 1][j - 1]
        return extended

    def remove_outer_nans(self, data_matrix):
        n = data_matrix.shape[0]
        reduced = numpy.empty((n - 2, n - 2,))
        reduced[:] = numpy.nan
        for i in range(0, n - 2):
            for j in range(0, n - 2):
                reduced[i][j] = data_matrix[i + 1][j + 1]
        return reduced

    @staticmethod
    def prepare_sensors(sensor_locations, resolution=100):
        """
        Common method, to pre-process sensors before display (project them in 2D).
        """

        def sphere_fit(params):
            """Function to fit the sensor locations to a sphere"""
            return ((sensor_locations[:, 0] - params[1]) ** 2 + (sensor_locations[:, 1] - params[2]) ** 2
                    + (sensor_locations[:, 2] - params[3]) ** 2 - params[0] ** 2)

        (radius, circle_x, circle_y, circle_z) = leastsq(sphere_fit, (1, 0, 0, 0))[0]
        # size of each square
        ssh = float(radius) / resolution  # half-size
        # Generate a grid and interpolate using the gridData module
        x_arr = numpy.arange(circle_x - radius, circle_x + radius, ssh * 2.0) + ssh
        y_arr = numpy.arange(circle_y - radius, circle_y + radius, ssh * 2.0) + ssh
        x_arr, y_arr = numpy.meshgrid(x_arr, y_arr)

        # project the sensor locations onto the sphere
        sproj = sensor_locations - numpy.array((circle_x, circle_y, circle_z))
        sproj = radius * sproj / numpy.c_[numpy.sqrt(numpy.sum(sproj ** 2, axis=1))]
        sproj += numpy.array((circle_x, circle_y, circle_z))
        return dict(sproj=sproj, x_arr=x_arr, y_arr=y_arr,
                    circle_x=circle_x, circle_y=circle_y, rad=radius)

    @classmethod
    def _normalize(cls, points_positions):
        """Centers the brain."""
        steps = []
        for column_idx in range(3):
            column = [row[column_idx] for row in points_positions]
            step = (max(column) + min(column)) / 2.0
            steps.append(step)
        step = numpy.array(steps)
        return points_positions - step


class TopographicViewer(BaseTopography, ABCDisplayer):
    """
    Interface between TVB Framework and web display of a topography viewer.
    """

    _ui_name = "Topographic Visualizer"
    _ui_subsection = "topography"

    def get_input_tree(self):
        return [{'name': 'data_0', 'label': 'Connectivity Measures 1',
                 'type': ConnectivityMeasure, 'required': True,
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1]),
                 'description': 'Punctual values for each node in the connectivity matrix. '
                                'This will give the colors of the resulting topographic image.'},
                {'name': 'data_1', 'label': 'Connectivity Measures 2', 'type': ConnectivityMeasure,
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1]),
                 'description': 'Comparative values'},
                {'name': 'data_2', 'label': 'Connectivity Measures 3', 'type': ConnectivityMeasure,
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1]),
                 'description': 'Comparative values'}]

    def launch(self, data_0, data_1=None, data_2=None):

        connectivity = data_0.connectivity
        sensor_locations = BaseTopography._normalize(connectivity.centres)
        sensor_number = len(sensor_locations)

        arrays = []
        titles = []
        min_vals = []
        max_vals = []
        data_array = []
        data_arrays = []
        for measure in [data_0, data_1, data_2]:
            if measure is not None:
                if len(measure.connectivity.centres) != sensor_number:
                    raise Exception("Use the same connectivity!!!")
                arrays.append(measure.array_data.tolist())
                titles.append(measure.title)
                min_vals.append(measure.array_data.min())
                max_vals.append(measure.array_data.max())

        color_bar_min = min(min_vals)
        color_bar_max = max(max_vals)

        self.init_topography(sensor_locations)

        for i, array_data in enumerate(arrays):
            data_array = self._get_topography_array(array_data, self.topography_data)
            data_array = self.fit_circle(data_array)
            has_nan = numpy.any(numpy.isnan(data_array))
            titles[i] = titles[i] + ("\n - Topography contains nan -" if has_nan else "")
            data_arrays.append(ABCDisplayer.dump_with_precision(data_array.flat))

        params = dict(matrix_datas=data_arrays,
                      matrix_shape=json.dumps(data_array.squeeze().shape),
                      titles=titles,
                      vmin=color_bar_min,
                      vmax=color_bar_max)
        return self.build_display_result("topographic/view", params)
