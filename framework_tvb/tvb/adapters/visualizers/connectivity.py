# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import json
import math
import numpy
import pylab
from copy import copy
from tvb.basic.profile import TvbProfile
from tvb.config import CONNECTIVITY_CREATOR_MODULE, CONNECTIVITY_CREATOR_CLASS
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.adapters.exceptions import LaunchException
from tvb.basic.filters.chain import FilterChain
from tvb.core.services.flow_service import FlowService
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.datatypes.surfaces import CorticalSurface


class ConnectivityViewer(ABCDisplayer):
    """ 
    Given a Connectivity Matrix and a Surface data the viewer will display the matrix 'inside' the surface data. 
    The surface is only displayed as a shadow.
    """

    _ui_name = "Connectivity Visualizer"


    def get_input_tree(self):
        """
        Take as Input a Connectivity Object.
        """
        return [{'name': 'input_data', 'label': 'Connectivity Matrix', 'type': Connectivity, 'required': True},
                {'name': 'surface_data', 'label': 'Brain Surface', 'type': CorticalSurface,
                 'description': 'The Brain Surface is used to give you an idea of the connectivity position relative '
                                'to the full brain cortical surface.  This surface will be displayed as a shadow '
                                '(only used in 3D Edges tab).'},
                {'name': 'colors', 'label': 'Node Colors', 'type': ConnectivityMeasure,
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1]),
                 'description': 'A ConnectivityMeasure DataType that establishes a colormap for the nodes '
                                'displayed in the 2D Connectivity tabs.'},
                {'name': 'step', 'label': 'Color Threshold', 'type': 'float',
                 'description': 'All nodes with a value greater or equal (>=) than this threshold will be displayed '
                                'as red discs, otherwise (<) they will be yellow. (This applies to 2D Connectivity  '
                                'tabs and the threshold will depend on the metric used to set the Node Color)'},
                {'name': 'rays', 'label': 'Shapes Dimensions', 'type': ConnectivityMeasure,
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1]),
                 'description': 'A ConnectivityMeasure datatype used to establish the size of the spheres representing '
                                'each node. (It only applies to 3D Nodes tab).'}]


    def get_required_memory_size(self, input_data, surface_data, colors, rays, step):
        """
        Return the required memory to run this algorithm.
        """
        if surface_data is not None:
            # Nr of triangles * sizeOf(uint16) + (nr of vertices + nr of normals) * sizeOf(float)
            return surface_data.number_of_vertices * 6 * 4 + surface_data.number_of_vertices * 6 * 8
            # If no surface pass, assume enough memory should be available.
        return -1


    def launch(self, input_data, surface_data=None, colors=None, rays=None, step=None):
        """
        Given the input connectivity data and the surface data, 
        build the HTML response to be displayed.

        :param input_data: the `Connectivity` object which will be displayed
        :param surface_data: if provided, it is displayed as a shadow to give an idea of the connectivity \
                             position relative to the full brain cortical surface
        :type surface_data: `CorticalSurface`
        :param colors: used to establish a colormap for the nodes displayed in 2D Connectivity viewers
        :type colors:  `ConnectivityMeasure`
        :param rays: used to establish the size of the spheres representing each node in 3D Nodes viewer
        :type rays:  `ConnectivityMeasure`
        :param step: a threshold applied to the 2D Connectivity Viewers to differentiate 2 types of nodes \
                     the ones with a value greater that this will be displayed as red discs, instead of yellow
        :type step:  float
        """
        global_params, global_pages = self.compute_connectivity_global_params(input_data, surface_data)
        global_params['isSingleMode'] = False

        result_params, result_pages = Connectivity2DViewer().compute_parameters(input_data, colors, rays, step)
        result_params.update(global_params)
        result_pages.update(global_pages)
        _params, _pages = Connectivity3DViewer().compute_parameters(input_data, colors, rays)
        result_params.update(_params)
        result_pages.update(_pages)
        _params, _pages = MPLH5Connectivity().compute_parameters(input_data)
        result_params.update(_params)
        result_pages.update(_pages)

        return self.build_display_result("connectivity/main_connectivity", result_params, result_pages)


    def generate_preview(self, input_data, figure_size=None, surface_data=None, colors=None, rays=None, step=None):
        """
        Generate the preview for the BURST cockpit.

        see `launch_`
        """
        parameters, _ = Connectivity2DViewer().compute_preview_parameters(input_data, figure_size[0], figure_size[1],
                                                                          colors, rays, step)
        return self.build_display_result("connectivity/portlet_preview", parameters)


    @staticmethod
    def _compute_matrix_extrema(m):
        """Returns the min max and the minimal nonzero value from ``m``"""
        minv = float('inf')
        min_nonzero = float('inf')
        maxv = - float('inf')

        for data in m:
            for d in data:
                minv = min(minv, d)
                maxv = max(maxv, d)
                if d != 0:
                    min_nonzero = min(min_nonzero, d)

        return minv, maxv, min_nonzero


    def compute_connectivity_global_params(self, input_data, surface_data=None):
        """
        Returns a dictionary which contains the data needed for drawing a connectivity.

        :param input_data: the `Connectivity` object
        :param surface_data: if provided, it is displayed as a shadow to give an idea of the connectivity \
                             position relative to the full brain cortical surface
        :type surface_data: `CorticalSurface`
        """
        path_weights = self.paths2url(input_data, 'ordered_weights')
        path_pos = self.paths2url(input_data, 'ordered_centres')
        path_tracts = self.paths2url(input_data, 'ordered_tracts')
        path_labels = self.paths2url(input_data, 'ordered_labels')
        path_hemisphere_order_indices = self.paths2url(input_data, 'hemisphere_order_indices')

        if surface_data:
            url_vertices, url_normals, _, url_triangles = surface_data.get_urls_for_rendering()
        else:
            url_vertices, url_normals, url_triangles = [], [], []

        algo, group = FlowService().get_algorithm_by_module_and_class(CONNECTIVITY_CREATOR_MODULE, CONNECTIVITY_CREATOR_CLASS)

        submit_url = '/flow/%d/%d' % (group.fk_category, group.id)
        global_pages = dict(controlPage="connectivity/top_right_controls")

        minimum, maximum, minimum_non_zero = self._compute_matrix_extrema(input_data.ordered_weights)
        minimum_t, maximum_t, minimum_non_zero_t = self._compute_matrix_extrema(input_data.ordered_tracts)

        global_params = dict(urlWeights=path_weights, urlPositions=path_pos,
                             urlTracts=path_tracts, urlLabels=path_labels,
                             originalConnectivity=input_data.gid, title="Connectivity Control",
                             submitURL=submit_url,
                             positions=input_data.ordered_centres,
                             tractsMin=minimum_t, tractsMax=maximum_t,
                             weightsMin=minimum, weightsMax=maximum,
                             tractsNonZeroMin=minimum_non_zero_t, weightsNonZeroMin=minimum_non_zero,
                             pointsLabels=input_data.ordered_labels, conductionSpeed=input_data.speed or 1,
                             urlVertices=json.dumps(url_vertices), urlTriangles=json.dumps(url_triangles),
                             urlNormals=json.dumps(url_normals),
                             connectivity_entity=input_data, surface_entity=surface_data,
                             algo_group=self.get_algo_group(),
                             base_selection=input_data.saved_selection_labels,
                             hemisphereOrderUrl=path_hemisphere_order_indices)
        global_params.update(self.build_template_params_for_subselectable_datatype(input_data))
        return global_params, global_pages


    @staticmethod
    def get_connectivity_parameters(input_connectivity, surface_data=None):
        """
        Returns a dictionary which contains all the needed data for drawing a connectivity.
        """
        viewer = ConnectivityViewer()
        global_params, global_pages = viewer.compute_connectivity_global_params(input_connectivity, surface_data)
        global_params.update(global_pages)
        global_params['selectedConnectivityGid'] = input_connectivity.gid
        return global_params

#    
# -------------------- Connectivity 3D code starting -------------------


class Connectivity3DViewer(object):
    """
    Behavior for the HTML/JS 3D representation of the connectivity matrix.
    """


    @staticmethod
    def compute_parameters(input_data, colors=None, rays=None):
        """
        Having as inputs a Connectivity matrix(required) and two arrays that 
        represent the rays and colors of the nodes from the matrix(optional) 
        this method will build the required parameter dictionary that will be 
        sent to the HTML/JS 3D representation of the connectivity matrix.
        """
        if colors is not None:
            color_list = colors.array_data.tolist()
            color_list = ABCDisplayer.get_one_dimensional_list(color_list, input_data.number_of_regions,
                                                               "Invalid input size for Sphere Colors")
            color_list = numpy.nan_to_num(numpy.array(color_list, dtype=numpy.float64)).tolist()
        else:
            color_list = [1.0] * input_data.number_of_regions

        if rays is not None:
            rays_list = rays.array_data.tolist()
            rays_list = ABCDisplayer.get_one_dimensional_list(rays_list, input_data.number_of_regions,
                                                              "Invalid input size for Sphere Sizes")
            rays_list = numpy.nan_to_num(numpy.array(rays_list, dtype=numpy.float64)).tolist()
        else:
            rays_list = [1.0] * input_data.number_of_regions

        params = dict(raysArray=json.dumps(rays_list), rayMin=min(rays_list), rayMax=max(rays_list),
                      colorsArray=json.dumps(color_list), colorMin=min(color_list), colorMax=max(color_list))
        return params, {}


# -------------------- Connectivity 2D code starting  ------------------
X_CANVAS_SMALL = 280
Y_CANVAS_SMALL = 145
X_CANVAS_FULL = 280
Y_CANVAS_FULL = 300



class Connectivity2DViewer(object):
    """
    Having as inputs a Connectivity matrix(required) and two arrays that 
    represent the colors and shapes of the nodes from the matrix(optional) 
    the viewer will build the required parameter dictionary that will be 
    sent to the HTML/JS 2D representation of the connectivity matrix.
    """
    DEFAULT_COLOR = '#d73027'
    OTHER_COLOR = '#1a9850'
    MIN_RAY = 4
    MAX_RAY = 40
    MIN_WEIGHT_VALUE = 0.0
    MAX_WEIGHT_VALUE = 0.6


    def compute_parameters(self, input_data, colors=None, rays=None, step=None):
        """
        Build the required HTML response to be displayed.

        :raises LaunchException: when number of regions in input_data is less than 3
        """
        if input_data.number_of_regions <= 3:
            raise LaunchException('The connectivity matrix you selected has fewer nodes than acceptable for display!')

        half = input_data.number_of_regions / 2
        normalized_weights = self._normalize_weights(input_data.ordered_weights)
        weights = Connectivity2DViewer._get_weights(normalized_weights)

        ## Compute shapes and colors ad adjacent data
        norm_rays, min_ray, max_ray = self._normalize_rays(rays, input_data.number_of_regions)
        colors, step = self._prepare_colors(colors, input_data.number_of_regions, step)

        right_json = self._get_json(input_data.ordered_labels[half:], input_data.ordered_centres[half:], weights[1],
                                    math.pi, 1, 2, norm_rays[half:], colors[half:], X_CANVAS_SMALL, Y_CANVAS_SMALL)
        left_json = self._get_json(input_data.ordered_labels[:half], input_data.ordered_centres[:half], weights[0],
                                   math.pi, 1, 2, norm_rays[:half], colors[:half], X_CANVAS_SMALL, Y_CANVAS_SMALL)
        full_json = self._get_json(input_data.ordered_labels, input_data.ordered_centres, normalized_weights,
                                   math.pi, 0, 1, norm_rays, colors, X_CANVAS_FULL, Y_CANVAS_FULL)

        params = dict(bothHemisphereJson=full_json, rightHemisphereJson=right_json, leftHemisphereJson=left_json,
                      stepValue=step or max_ray, firstColor=self.DEFAULT_COLOR,
                      secondColor=self.OTHER_COLOR, minRay=min_ray, maxRay=max_ray)
        return params, {}


    def compute_preview_parameters(self, input_data, width, height, colors=None, rays=None, step=None):
        """
        Build the required HTML response to be displayed in the BURST preview iFrame.
        """
        if input_data.number_of_regions <= 3:
            raise LaunchException('The connectivity matrix you selected has fewer nodes than acceptable for display!')
        norm_rays, min_ray, max_ray = self._normalize_rays(rays, input_data.number_of_regions)
        colors, step = self._prepare_colors(colors, input_data.number_of_regions, step)
        normalizer_size_coeficient = width / 600.0
        if height / 700 < normalizer_size_coeficient:
            normalizer_size_coeficient = (height * 0.8) / 700.0
        x_size = X_CANVAS_FULL * normalizer_size_coeficient
        y_size = Y_CANVAS_FULL * normalizer_size_coeficient
        full_json = self._get_json(input_data.ordered_labels, input_data.ordered_centres, input_data.ordered_weights,
                                   math.pi, 0, 1, norm_rays, colors, x_size, y_size)
        params = dict(bothHemisphereJson=full_json, stepValue=step or max_ray, firstColor=self.DEFAULT_COLOR,
                      secondColor=self.OTHER_COLOR, minRay=min_ray, maxRay=max_ray)
        return params, {}


    def _get_json(self, labels, positions, weights, rotate_angle, coord_idx1,
                  coord_idx2, dimensions_list, colors_list, x_canvas, y_canvas):
        """
        Method used for creating a valid JSON for an entire chart.
        """
        max_y = max(positions[:, coord_idx2])
        min_y = min(positions[:, coord_idx2])
        max_x = max(positions[:, coord_idx1])
        min_x = min(positions[:, coord_idx1])
        y_scale = 2 * y_canvas / (max_y - min_y)
        x_scale = 2 * x_canvas / (max_x - min_x)
        mid_x_value = (max_x + min_x) / 2
        mid_y_value = (max_y + min_y) / 2

        result_json = []

        for i in xrange(len(positions)):
            x_coord = (positions[i][coord_idx1] - mid_x_value) * x_scale
            y_coord = (positions[i][coord_idx2] - mid_y_value) * y_scale
            adjacencies = Connectivity2DViewer._get_adjacencies_json(weights[i], labels)

            r = self.point2json(labels[i], x_coord, y_coord, adjacencies,
                                rotate_angle, dimensions_list[i], colors_list[i])
            result_json.append(r)

        return json.dumps(result_json)

    @staticmethod
    def _get_weights(weights):
        """
        Method used for calculating the weights for the right and for the 
        left hemispheres. Those matrixes are obtained from
        a weights matrix which contains data related to both hemispheres.
        """
        half = len(weights) / 2
        l_aux, r_aux = weights[:half], weights[half:]
        r_weights = []
        l_weights = []
        for i in range(half):
            l_weights.append(l_aux[i][:half])
        for i in range(half, len(weights)):
            r_weights.append(r_aux[i - half][half:])
        return l_weights, r_weights


    def point2json(self, node_lbl, x_coord, y_coord, adjacencies, angle, shape_dimension, shape_color):
        """
        Method used for creating a valid JSON for a certain point.
        """
        form = "circle"
        default_dimension = 6
        angle += math.atan2(y_coord, x_coord)
        radius = math.sqrt(math.pow(x_coord, 2) + math.pow(y_coord, 2))

        return {
            "id": node_lbl, "name": node_lbl,
            "data": {
                "$dim": default_dimension, "$type": form,
                "$color": self.DEFAULT_COLOR, "customShapeDimension": shape_dimension,
                "customShapeColor": shape_color, "angle": angle,
                "radius": radius
            },
            "adjacencies": adjacencies
        }

    @staticmethod
    def _get_adjacencies_json(point_weights, points_labels):
        """
        Method used for obtaining a valid JSON which will contain all the edges of a certain node.
        """
        adjacencies = []
        for weight, label in zip(point_weights, points_labels):
            if weight:
                adjacencies.append({"nodeTo": label, "data": {"weight": weight}})
        return adjacencies


    def _prepare_colors(self, colors, expected_size, step=None):
        """
        From the input array, all values smaller than step will get a different color
        """
        if colors is None:
            return [self.DEFAULT_COLOR] * expected_size, None
        colors = numpy.nan_to_num(numpy.array(colors.array_data, dtype=numpy.float64)).tolist()
        colors = ABCDisplayer.get_one_dimensional_list(colors, expected_size, "Invalid size for colors array!")
        result = []
        if step is None:
            step = (max(colors) + min(colors)) / 2
        for val in colors:
            if val < step:
                result.append(self.OTHER_COLOR)
            else:
                result.append(self.DEFAULT_COLOR)
        return result, step


    def _normalize_rays(self, rays, expected_size):
        """
        Make sure all rays are in the interval [self.MIN_RAY, self.MAX_RAY]
        """
        if rays is None:
            value = (self.MAX_RAY + self.MIN_RAY) / 2
            return [value] * expected_size, 0.0, 0.0
        rays = rays.array_data.tolist()
        rays = ABCDisplayer.get_one_dimensional_list(rays, expected_size, "Invalid size for rays array.")
        min_x = min(rays)
        max_x = max(rays)
        if min_x >= self.MIN_RAY and max_x <= self.MAX_RAY:
            # No need to normalize
            return rays, min_x, max_x
        result = []
        diff = max_x - min_x
        if min_x == max_x:
            diff = self.MAX_RAY - self.MIN_RAY
        for ray in rays:
            result.append(self.MIN_RAY + self.MAX_RAY * (ray - min_x) / diff)
        result = numpy.nan_to_num(numpy.array(result, dtype=numpy.float64)).tolist()
        return result, min(rays), max(rays)


    def _normalize_weights(self, weights):
        """
        Normalize the weights matrix. The values should be between 
        MIN_WEIGHT_VALUE and MAX_WEIGHT_VALUE
        """
        weights = copy(weights)
        min_value = numpy.min(weights)
        max_value = numpy.max(weights)
        if min_value < self.MIN_WEIGHT_VALUE or max_value > self.MAX_WEIGHT_VALUE:
            for i, row in enumerate(weights):
                for j in xrange(len(row)):
                    if min_value == max_value:
                        weights[i][j] = self.MAX_WEIGHT_VALUE
                    else:
                        weights[i][j] = (self.MIN_WEIGHT_VALUE + ((weights[i][j] - min_value) / (max_value - min_value))
                                         * (self.MAX_WEIGHT_VALUE - self.MIN_WEIGHT_VALUE))
        return weights


# -------------------- Connectivity MPLH5 code starting  ------------------

class MPLH5Connectivity():
    """
    MPLH5 tab page, where the entire connectivity matrix can be seen.
    """


    @staticmethod
    def compute_parameters(input_data):
        """
        Build the required HTML response to be displayed.
        """
        figure = pylab.figure(figsize=(7, 8))
        pylab.clf()

        matrix = input_data.ordered_weights
        matrix_size = input_data.number_of_regions
        labels = input_data.ordered_labels
        plot_title = "Connection Strength"

        order = numpy.arange(matrix_size)
        # Assumes order is shape (number_of_regions, )
        order_rows = order[:, numpy.newaxis]
        order_columns = order_rows.T

        axes = figure.gca()
        img = axes.matshow(matrix[order_rows, order_columns])
        axes.set_title(plot_title)
        figure.colorbar(img)

        if labels is None:
            return

        axes.set_yticks(numpy.arange(matrix_size))
        axes.set_yticklabels(list(labels[order]), fontsize=8)
        axes.set_xticks(numpy.arange(matrix_size))
        axes.set_xticklabels(list(labels[order]), fontsize=8, rotation=90)

        figure.canvas.draw()
        parameters = dict(mplh5ServerURL=TvbProfile.current.web.MPLH5_SERVER_URL,
                          figureNumber=figure.number, showFullToolbar=False)
        return parameters, {}
    
    
    