# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
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

The Pattern datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy

from tvb.datatypes import surfaces, volumes, connectivity, equations
from tvb.basic.neotraits.api import HasTraits, NArray, Attr


class SpatialPattern(HasTraits):
    """
    Equation for space variation.
    """

    spatial =  Attr(field_type=equations.FiniteSupportEquation, label="Spatial Equation")

    space = None
    _spatial_pattern = None

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance of this DataType.
        """
        return {"Type": self.__class__.__name__,
                "Spatial equation": self.spatial.__class__.__name__,
                "Spatial parameters": self.spatial.parameters}

    @property
    def spatial_pattern(self):
        """
        Return a discrete representation of the spatial pattern.
        """
        return self._spatial_pattern

    def configure_space(self, distance):
        """
        Stores the distance vector as an attribute of the spatiotemporal pattern
        and uses it to generate the spatial pattern vector.

        Depending on equations used and interpretation distance can be an actual
        physical distance, on a surface,  geodesic distance (along the surface)
        away for some focal point, or a per node weighting...
        """
        # Set the discrete representation of space.
        self.space = distance
        # Generate a discrete representation of the spatial pattern.
        # The argument x represents a distance, or effective distance, for each node in the space.
        self._spatial_pattern = numpy.sum(self.spatial.evaluate(self.space), axis=1)[:, numpy.newaxis]


class SpatioTemporalPattern(SpatialPattern):
    """
    Combine space and time equations.
    """

    temporal = Attr(field_type=equations.TemporalApplicableEquation, label="Temporal Equation")
    # space must be shape (x, 1); time must be shape (1, t)
    time = None
    _temporal_pattern = None

    def summary_info(self):
        """ Extend the base class's summary dictionary. """
        summary = super(SpatioTemporalPattern, self).summary_info()
        summary["Temporal equation"] = self.temporal.__class__.__name__
        summary["Temporal parameters"] = self.temporal.parameters
        return summary

    @property
    def temporal_pattern(self):
        """
        Return a discrete representation of the temporal pattern.
        """
        return self._temporal_pattern

    def configure_time(self, time):
        """
        Stores the time vector, physical units (ms), as an attribute of the
        spatio-temporal pattern and uses it to generate the temporal pattern
        vector.
        """
        self.time = time
        # Generate a discrete representation of the temporal pattern.
        self._temporal_pattern = numpy.reshape(self.temporal.evaluate(self.time), (1, -1))

    def __call__(self, temporal_indices=None, spatial_indices=None):
        """
        The temporal pattern vector, set by the configure_time method, is
        combined with the spatial pattern vector, set by the configure_space
        method, to form a spatiotemporal pattern.

        Called with a single time index as an argument, the spatial pattern at
        that point in time is returned. This is the standard usage within a
        simulation where the current simulation time point is retrieved.

        Called without any arguments, by default a big array representing the
        entire spatio-temporal pattern is returned. While this may be useful for
        visualisation, say of region level spatio-temporal patterns, care should
        be taken as when surfaces are considered the returned array can be
        potentially quite large.
        """
        pattern = None
        if temporal_indices is not None and spatial_indices is None:
            pattern = self._spatial_pattern * self._temporal_pattern[0, temporal_indices]
        elif temporal_indices is None and spatial_indices is None:
            pattern = self._spatial_pattern * self._temporal_pattern
        elif temporal_indices is not None and spatial_indices is not None:
            pattern = self._spatial_pattern[spatial_indices, 0] * self._temporal_pattern[0, temporal_indices]
        elif temporal_indices is None and spatial_indices is not None:
            pattern = self._spatial_pattern[spatial_indices, 0] * self._temporal_pattern
        else:
            self.log.error("%s: Well, that shouldn't be possible..." % repr(self))
        return pattern


class StimuliRegion(SpatioTemporalPattern):
    """
    A class that bundles the temporal profile of the stimulus, together with the
    list of scaling weights of the regions where it will applied.
    """
    connectivity = Attr(field_type=connectivity.Connectivity, label="Connectivity")

    spatial = Attr(field_type=equations.DiscreteEquation,
                   label="Spatial Equation",
                   default=equations.DiscreteEquation())  # fixed_type=True, order=-1)

    weight = NArray(label="scaling")  # , locked=True, order=4)

    @staticmethod
    def get_default_weights(number_of_regions):
        """
        Returns a list with a number of elements
        equal to the given number of regions.
        """
        return numpy.array([0.0] * number_of_regions)

    @property
    def weight_array(self):
        """
        Wrap weight List into a Numpy array, as it is requested by the simulator.
        """
        return numpy.array(self.weight)[:, numpy.newaxis]

    def configure_space(self, region_mapping=None):
        """
        Do necessary preparations in order to use this stimulus.
        NOTE: this was previously done in simulator configure_stimuli() method.
        It no needs to be used in stimulus viewer also.
        """
        if region_mapping is not None:
            # TODO: smooth at surface region boundaries
            distance = self.weight_array[region_mapping, :]
        else:
            distance = self.weight_array
        super(StimuliRegion, self).configure_space(distance)


class StimuliSurface(SpatioTemporalPattern):
    """
    A spatio-temporal pattern defined in a Surface DataType.
    It includes the list of focal points.
    """

    surface = Attr(field_type=surfaces.CorticalSurface, label="Surface")

    focal_points_triangles = NArray(dtype=int, label="Focal points triangles")  # , locked=True, order=4)

    @property
    def focal_points_surface(self):
        focal_points = []

        if self.surface is None or self.surface.triangles is None:
            self.log.warning('Focal points list will be empty. Load the surface triangles before accessing this property!')
            return numpy.array(focal_points)

        for triangle_index in self.focal_points_triangles:
            focal_points.append(int(self.surface.triangles[triangle_index][0]))
        return numpy.array(focal_points)

    def configure_space(self, region_mapping=None):
        """
        Do necessary preparations in order to use this stimulus.
        NOTE: this was previously done in simulator configure_stimuli() method.
        It no needs to be used in stimulus viewer also.
        """
        dis_shp = (self.surface.number_of_vertices, numpy.size(self.focal_points_surface))
        # TODO: When this was in Simulator it was number of nodes, using surface vertices
        # breaks surface simulations which include non-cortical regions.

        distance = numpy.zeros(dis_shp)
        k = -1
        for focal_point in self.focal_points_surface:
            k += 1
            foci = numpy.array([focal_point], dtype=numpy.int32)
            distance[:, k] = self.surface.geodesic_distance(foci)
        super(StimuliSurface, self).configure_space(distance)


class SpatialPatternVolume(SpatialPattern):
    """ A spatio-temporal pattern defined in a volume. """

    volume = Attr(volumes.Volume, label="Volume")

    focal_points_volume = NArray(dtype=int, label="Focal points")
