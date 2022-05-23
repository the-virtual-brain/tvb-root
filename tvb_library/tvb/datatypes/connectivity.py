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

The Connectivity datatype.

.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

import numpy
import scipy.stats
from copy import copy
from io import BytesIO
from tvb.basic.exceptions import ValidationException
from tvb.basic.neotraits.api import Attr, NArray, List, HasTraits, Int, narray_summary_info
from tvb.basic.readers import ZipReader, H5Reader, try_get_absolute_path


class Connectivity(HasTraits):
    region_labels = NArray(
        dtype='U128',
        label="Region labels",
        doc="""Short strings, 'labels', for the regions represented by the connectivity matrix.""")

    weights = NArray(
        label="Connection strengths",
        doc="""Matrix of values representing the strength of connections between regions, arbitrary units.""")

    undirected = Attr(
        field_type=bool,
        default=False, required=False,
        doc="1, when the weights matrix is square and symmetric over the main diagonal, 0 when directed graph.")

    tract_lengths = NArray(
        label="Tract lengths",
        doc="""The length of myelinated fibre tracts between regions.
            If not provided Euclidean distance between region centres is used.""")

    speed = NArray(
        label="Conduction speed",
        default=numpy.array([3.0]),
        doc="""A single number or matrix of conduction speeds for the myelinated fibre tracts between regions.""")

    centres = NArray(
        label="Region centres",
        doc="An array specifying the location of the centre of each region.")

    cortical = NArray(
        dtype=bool,
        label="Cortical",
        required=False,
        doc="""A boolean vector specifying whether or not a region is part of the cortex.""")

    hemispheres = NArray(
        dtype=bool,
        label="Hemispheres (True for Right and False for Left Hemisphere",
        required=False,
        doc="""A boolean vector specifying whether or not a region is part of the right hemisphere""")

    orientations = NArray(
        label="Average region orientation",
        required=False,
        doc="""Unit vectors of the average orientation of the regions represented in the connectivity matrix.
            NOTE: Unknown data should be zeros.""")

    areas = NArray(
        label="Area of regions",
        required=False,
        doc="""Estimated area represented by the regions in the connectivity matrix.
            NOTE: Unknown data should be zeros.""")

    idelays = NArray(
        dtype=int,
        label="Conduction delay indices",
        required=False,
        doc="An array of time delays between regions in integration steps.")

    delays = NArray(
        label="Conduction delay",
        required=False,
        doc="""Matrix of time delays between regions in physical units, setting conduction speed automatically
            combines with tract lengths to update this matrix, i.e. don't try and change it manually.""")

    number_of_regions = Int(
        field_type=int,
        label="Number of regions",
        doc="""The number of regions represented in this Connectivity """)

    number_of_connections = Int(
        field_type=int,
        label="Number of connections",
        doc="""The number of non-zero entries represented in this Connectivity """)

    # Original Connectivity, from which current connectivity was edited.
    parent_connectivity = Attr(field_type=str, required=False)

    # In case of edited Connectivity, this are the nodes left in interest area,
    # the rest were part of a lesion, so they were removed.
    saved_selection = List(of=int)

    @property
    def subcortical_indices(self):
        subcortical_indices = numpy.flatnonzero(self.cortical == 0)
        return subcortical_indices

    @property
    def saved_selection_labels(self):
        """
        Taking the entity field saved_selection, convert indexes in that array
        into labels.
        """
        if self.saved_selection:
            idxs = [int(i) for i in self.saved_selection]
            result = ''
            for i in idxs:
                result += self.region_labels[i] + ','
            return result[:-1]
        else:
            return ''

    def is_right_hemisphere(self, idx):
        """
        :param idx:  Region IDX
        :return: True when hemispheres information is present and it shows that the current node is in the right
        hemisphere. When hemispheres info is not present, return True for the second half of the indices and
        False otherwise.
        """
        if self.hemispheres is not None and self.hemispheres.size:
            return self.hemispheres[idx]
        return idx >= self.number_of_regions / 2

    @property
    def hemisphere_order_indices(self):
        """
        A sequence of indices of rows/colums.
        These permute rows/columns so that the first half would belong to the first hemisphere
        If there is no hemisphere information returns the identity permutation
        """
        if self.hemispheres is not None and self.hemispheres.size:
            li, ri = [], []
            for i, is_right in enumerate(self.hemispheres):
                if is_right:
                    ri.append(i)
                else:
                    li.append(i)
            return numpy.array(li + ri)
        else:
            return numpy.arange(self.number_of_regions)

    @property
    def ordered_weights(self):
        """
        This view of the weights matrix lists all left hemisphere nodes before the right ones.
        It is used by viewers of the connectivity.
        """
        permutation = self.hemisphere_order_indices
        # how this works:
        # w[permutation, :] selects all rows at the indices present in the permutation array thus permuting the rows
        # [:, permutation] does the same to columns. See numpy index arrays
        return self.weights[permutation, :][:, permutation]

    @property
    def ordered_tracts(self):
        """
        Similar to :meth:`ordered_weights`
        """
        permutation = self.hemisphere_order_indices
        return self.tract_lengths[permutation, :][:, permutation]

    @property
    def ordered_labels(self):
        """
        Similar to :meth:`ordered_weights`
        """
        permutation = self.hemisphere_order_indices
        return self.region_labels[permutation]

    @property
    def ordered_centres(self):
        """
        Similar to :method:`ordered_weights`
        """
        permutation = self.hemisphere_order_indices
        return self.centres[permutation]

    def get_grouped_space_labels(self):
        """
        :return: A list [('left', [lh_labels)], ('right': [rh_labels])]
        """
        if self.hemispheres is not None and self.hemispheres.size:
            l, r = [], []

            for i, (is_right, label) in enumerate(zip(self.hemispheres, self.region_labels)):
                if is_right:
                    r.append((i, label))
                else:
                    l.append((i, label))
            return [('Left', l), ('Right', r)]
        else:
            return [('', list(enumerate(self.region_labels)))]

    def get_default_selection(self):
        # should this be sub-selection or all always?
        sel = self.saved_selection
        if sel is not None and len(sel) > 0:
            return sel
        else:
            return list(range(len(self.region_labels)))

    @property
    def binarized_weights(self):
        """
        :return: a matrix of he same size as weights, with 1 where weight > 0, and 0 in rest
        """
        result = numpy.zeros_like(self.weights)
        result = numpy.where(self.weights > 0, 1, result)
        return result

    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialization.
        """

        self.number_of_regions = int(self.weights.shape[0])
        # NOTE: In numpy 1.8 there is a function called count_zeros
        self.number_of_connections = int(self.weights.nonzero()[0].shape[0])

        if self.tract_lengths is None or self.tract_lengths.size == 0:
            self.compute_tract_lengths()
        if self.region_labels is None or self.region_labels.size == 0:
            self.compute_region_labels()
        if self.hemispheres is None or self.hemispheres.size == 0:
            self.try_compute_hemispheres()

        # This can not go into compute, as it is too complex reference
        # if self.delays.size == 0:
        # TODO: Because delays are stored and loaded the size was never 0.0 and
        #      so this wasn't being run, making the conduction_speed hack on the
        #      simulator non-functional. Inn the longer run it'll probably be
        #      necessary for delays to never be stored but always calculated
        #      from tract-lengths and speed...
        if self.speed is None:
            self.log.warning("Connectivity.speed attribute not initialized properly, setting it to 3.0...")
            self.speed = numpy.array([3.0])

        # NOTE: Because of the conduction_speed hack for UI this must be evaluated here, even if delays
        # already has a value, otherwise setting speed in the UI has no effect...
        self.delays = self.tract_lengths / self.speed

        if (self.weights.transpose() == self.weights).all():
            self.undirected = True

        self.validate()

    def summary_info(self):
        summary = {
            "Number of regions": self.number_of_regions,
            "Number of connections": self.number_of_connections,
            "Undirected": self.undirected,
        }
        summary.update(narray_summary_info(self.areas, ar_name='areas'))
        summary.update(narray_summary_info(self.weights, ar_name='weights'))
        summary.update(narray_summary_info(
            self.weights[self.weights.nonzero()],
            ar_name='weights-non-zero',
            omit_shape=True))
        summary.update(narray_summary_info(
            self.tract_lengths,
            ar_name='tract_lengths',
            omit_shape=True))
        summary.update(narray_summary_info(
            self.tract_lengths[self.tract_lengths.nonzero()],
            ar_name='tract_lengths-non-zero',
            omit_shape=True))
        summary.update(narray_summary_info(
            self.tract_lengths[self.weights.nonzero()],
            ar_name='tract_lengths (connections)',
            omit_shape=True))
        return summary

    def set_idelays(self, dt):
        """
        Convert the time delays between regions in physical units into an array
        of linear indices into the simulator's history attribute.

        args:
            ``dt (float64)``: Length of integration time step...

        Updates attribute:
            ``idelays (numpy.array)``: Transmission delay between brain regions
            in integration steps.
        """
        # Express delays in integration steps
        self.idelays = numpy.rint(self.delays / dt).astype(numpy.int32)
        self.has_delays = self.idelays.any()
        self._horizon = self.idelays.max() + 1
        nn = self.idelays.shape[0]
        self.inodes = numpy.tile(numpy.r_[:nn], (nn, 1))
        self.delay_indices = self.idelays * nn + self.inodes

    def compute_tract_lengths(self):
        """
        If no tract lengths data are available, this can be used to calculate
        the Euclidean distance between region centres to use as a proxy.

        """
        nor = self.number_of_regions
        tract_lengths = numpy.zeros((nor, nor))
        # Suggestion for optimization: redundant by half, do half triangle then flip...
        for region in range(nor):
            temp = self.centres - self.centres[region, :][numpy.newaxis, :]
            tract_lengths[region, :] = numpy.sqrt(numpy.sum(temp ** 2, axis=1))

        self.tract_lengths = tract_lengths

    def compute_region_labels(self):
        """
        Compute some labers, if missing
        """
        labels = ["region_%03d" % n for n in range(self.number_of_regions)]
        self.region_labels = numpy.array(labels, dtype="128a")

    def try_compute_hemispheres(self):
        """
        If all region labels are prefixed with L or R, then compute hemisphere side with that.
        """
        if self.region_labels is not None and self.region_labels.size > 0:
            hemispheres = []
            # Check if all labels are prefixed with R / L
            for label in self.region_labels:
                if label is not None and label.lower().startswith('r'):
                    hemispheres.append(True)
                elif label is not None and label.lower().startswith('l'):
                    hemispheres.append(False)
                else:
                    hemispheres = None
                    break
            # Check if all labels are sufixed with R / L
            if hemispheres is None:
                hemispheres = []
                for label in self.region_labels:
                    if label is not None and label.lower().endswith('r'):
                        hemispheres.append(True)
                    elif label is not None and label.lower().endswith('l'):
                        hemispheres.append(False)
                    else:
                        hemispheres = None
                        break
            if hemispheres is not None:
                self.hemispheres = numpy.array(hemispheres, dtype=numpy.bool_)

    def transform_remove_self_connections(self):
        """
        Remove the values from the main diagonal (self-connections)
        """
        nor = self.number_of_regions
        result = copy(self.weights)
        result = result - result * numpy.eye(nor, nor)
        return result

    def scaled_weights(self, mode='tract'):
        """
        Scale the connection strengths (weights) and return the scaled matrix.
        Three simple types of scaling are supported.
        The ``scaling_mode``  is one of the following:

            'tract': Scale by a value such that the maximum absolute value of a single
                connection is 1.0. (Global scaling)

            'region': Scale by a value such that the maximum absolute value of the
                cumulative input to any region is 1.0. (Global-wise scaling)

            None: does nothing.

        NOTE: Currently multiple 'tract' and/or 'region' scalings without
            intermediate 'none' scaling mode destroy the ability to recover
            the original un-scaled weights matrix.

        """
        # NOTE: It is not yet clear how or if we will integrate this functinality
        #      into the UI. Currently the same effect can be achieved manually
        #      by using the coupling functions, it is just that, in certain
        #      situations, things are simplified by starting from a normalised
        #      weights matrix. However, in other situations it is not desirable
        #      to have a simple normalisation of this sort.
        # NOTE: We should probably separate the two cases implemented here into
        #      'scaling' and 'normalisation'. Normalisation implies that the norm
        #      of the samples is equal to 1, while here it is only scaling by a factor.

        self.log.info("Starting to normalize to mode: %s" % str(mode))

        normalisation_factor = None
        if mode in ("tract", "edge"):
            # global scaling
            normalisation_factor = numpy.abs(self.weights).max()
        elif mode in ("region", "node"):
            # node-wise scaling
            normalisation_factor = numpy.max(numpy.abs(self.weights.sum(axis=1)))
        elif mode in (None, "none"):
            normalisation_factor = 1.0
        else:
            self.log.error("Bad weights normalisation mode, must be one of:")
            self.log.error("('tract', 'edge', 'region', 'node', 'none')")
            raise ValidationException("Bad weights normalisation mode")

        self.log.debug("Normalization factor is: %s" % str(normalisation_factor))
        mask = self.weights != 0.0
        result = copy(self.weights)
        result[mask] = self.weights[mask] / normalisation_factor
        return result

    def transform_binarize_matrix(self):
        """
        Transforms the weights matrix into a binary (unweighted) matrix
        """
        self.log.info("Transforming weighted matrix into unweighted matrix")

        result = copy(self.weights)
        result = numpy.where(result > 0, 1, result)
        return result

    def motif_linear_directed(self, number_of_regions=4, max_radius=100., return_type=None):
        """
        Generates a linear (open chain) unweighted directed graph with equidistant nodes.
        """

        iu1 = numpy.triu_indices(number_of_regions, 1)
        iu2 = numpy.triu_indices(number_of_regions, 2)

        self.weights = numpy.zeros((number_of_regions, number_of_regions))
        self.weights[iu1] = 1.0
        self.weights[iu2] = 0.0

        self.tract_lengths = max_radius * copy(self.weights)
        self.number_of_regions = number_of_regions
        self.create_region_labels(mode='numeric')

        if return_type is not None:
            return self.weights, self.tract_lengths
        else:
            pass

    def motif_linear_undirected(self, number_of_regions=4, max_radius=42.):
        """
        Generates a linear (open chain) unweighted undirected graph with equidistant nodes.
        """

        self.weights, self.tract_lengths = self.motif_linear_directed(number_of_regions=number_of_regions,
                                                                      max_radius=max_radius,
                                                                      return_type=True)

        self.weights += self.weights.T
        self.tract_lengths += self.tract_lengths.T
        self.number_of_regions = number_of_regions
        self.create_region_labels(mode='numeric')

    def motif_chain_directed(self, number_of_regions=4, max_radius=42., return_type=None):
        """
        Generates a closed unweighted directed graph with equidistant nodes.
        Depending on the centres it could be a box or a ring.
        """

        self.weights, self.tract_lengths = self.motif_linear_directed(number_of_regions=number_of_regions,
                                                                      max_radius=max_radius,
                                                                      return_type=True)

        self.weights[-1, 0] = 1.0
        self.tract_lengths[-1, 0] = max_radius
        self.number_of_regions = number_of_regions
        self.create_region_labels(mode='numeric')

        if return_type is not None:
            return self.weights, self.tract_lengths
        else:
            pass

    def motif_chain_undirected(self, number_of_regions=4, max_radius=42.):
        """
        Generates a closed unweighted undirected graph with equidistant nodes.
        Depending on the centres it could be a box or a ring.
        """

        self.weights, self.tract_lengths = self.motif_chain_directed(number_of_regions=number_of_regions,
                                                                     max_radius=max_radius,
                                                                     return_type=True)

        self.weights[0, -1] = 1.0
        self.tract_lengths[0, -1] = max_radius
        self.number_of_regions = number_of_regions
        self.create_region_labels(mode='numeric')

    def motif_all_to_all(self, number_of_regions=4, max_radius=42.):
        """
        Generates an all-to-all closed unweighted undirected graph with equidistant nodes.
        Self-connections are not included.
        """

        diagonal_elements = numpy.diag_indices(number_of_regions)

        self.weights = numpy.ones((number_of_regions, number_of_regions))
        self.weights[diagonal_elements] = 0.0
        self.tract_lengths = max_radius * copy(self.weights)
        self.number_of_regions = number_of_regions
        self.create_region_labels(mode='numeric')

    def centres_spherical(self, number_of_regions=4, max_radius=42., flat=False):
        """
        The nodes positions are distributed on a sphere.
        See: http://mathworld.wolfram.com/SphericalCoordinates.html

        If flat is true, then theta=0.0, the nodes are lying inside a circle.

        r    : radial
        theta: azimuthal
        polar: phi
        """

        # azimuth
        theta = numpy.random.uniform(low=-numpy.pi, high=numpy.pi, size=number_of_regions)

        # side of the cube
        u = numpy.random.uniform(low=0.0, high=1.0, size=number_of_regions)

        if flat:
            cosphi = 0.0
        else:
            # cos(elevation)
            cosphi = numpy.random.uniform(low=-1.0, high=1.0, size=number_of_regions)

        phi = numpy.arccos(cosphi)
        r = max_radius * pow(u, 1 / 3.0)

        # To Cartesian coordinates
        x = r * numpy.sin(phi) * numpy.cos(theta)
        y = r * numpy.sin(phi) * numpy.sin(theta)
        z = r * numpy.cos(phi)

        self.centres = numpy.array([x, y, z]).T
        norm_xyz = numpy.sqrt(numpy.sum(self.centres ** 2, axis=0))
        self.orientations = self.centres / norm_xyz[numpy.newaxis, :]

    def centres_toroidal(self, number_of_regions=4, max_radius=77., min_radius=13., mu=numpy.pi, kappa=numpy.pi / 6):
        """
        The nodes are lying on  a torus.
        See: http://mathworld.wolfram.com/Torus.html

        """

        u = scipy.stats.vonmises.rvs(kappa, loc=mu, size=number_of_regions)
        v = scipy.stats.vonmises.rvs(kappa, loc=mu, size=number_of_regions)

        # To cartesian coordinates
        x = (max_radius + min_radius * numpy.cos(v)) * numpy.cos(u)
        y = (max_radius + min_radius * numpy.cos(v)) * numpy.sin(u)
        z = min_radius * numpy.sin(v)

        # Tangent vector with respect to max_radius
        tx = -numpy.sin(u)
        ty = -numpy.cos(u)
        tz = 0

        # Tangent vector with respect to min_radius
        sx = -numpy.cos(u) * (-numpy.sin(v))
        sy = numpy.sin(u) * (-numpy.sin(v))
        sz = numpy.cos(v)

        # Normal vector
        nx = ty * sz - tz * sy
        ny = tz * sx - tx * sz
        nz = tx * sy - ty * sx

        # Normalize normal vectors
        norm = numpy.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        nx /= norm
        ny /= norm
        nz /= norm

        self.orientations = numpy.array([nx, ny, nz]).T
        self.centres = numpy.array([x, y, z]).T

    def centres_annular(self, number_of_regions=4, max_radius=77., min_radius=13., mu=numpy.pi, kappa=numpy.pi / 6):
        """
        The nodes are lying inside an annulus.

        """

        r = numpy.random.uniform(low=min_radius, high=max_radius, size=number_of_regions)
        theta = scipy.stats.vonmises.rvs(kappa, loc=mu, size=number_of_regions)

        # To cartesian coordinates
        x = r * numpy.cos(theta)
        y = r * numpy.sin(theta)
        z = numpy.zeros(number_of_regions)

        self.centres = numpy.array([x, y, z]).T

    def centres_cubic(self, number_of_regions=4, max_radius=42., flat=False):
        """
        The nodes are positioined in a 3D grid inside the cube centred at the origin and
        with edges parallel to the axes, with an edge length of 2*max_radius.

        """

        # To cartesian coordinates
        x = numpy.linspace(-max_radius, max_radius, number_of_regions)
        y = numpy.linspace(-max_radius, max_radius, number_of_regions)

        if flat:
            z = numpy.zeros(number_of_regions)
        else:
            z = numpy.linspace(-max_radius, max_radius, number_of_regions)

        self.centres = numpy.array([x, y, z]).T

    def generate_surrogate_connectivity(self, number_of_regions, motif='chain', undirected=True,
                                        these_centres='spherical'):
        """
        This one generates some defaults.
        For more specific motifs, generate invoking each method separetly.

        """

        # NOTE: Luckily I went for 5 motifs ...
        if motif == 'chain' and undirected:
            self.motif_chain_undirected(number_of_regions=number_of_regions)
        elif motif == "chain" and not undirected:
            self.motif_chain_directed(number_of_regions=number_of_regions)
        elif motif == 'linear' and undirected:
            self.motif_linear_undirected(number_of_regions=number_of_regions)
        elif motif == 'linear' and not undirected:
            self.motif_linear_directed(number_of_regions=number_of_regions)
        else:
            self.log.info("Generating all-to-all connectivity \\")
            self.motif_all_to_all(number_of_regions=number_of_regions)

        # centres
        if these_centres in ("spherical", "annular", "toroidal", "cubic"):
            eval("self.centres_" + these_centres + "(number_of_regions=number_of_regions)")
        else:
            raise ValidationException("Bad centres geometry")

    def create_region_labels(self, mode="numeric"):

        """
        Assumes weights already exists
        """

        self.log.info("Create labels: %s" % str(mode))

        if mode in ("numeric", "num"):
            region_labels = [n for n in range(self.number_of_regions)]
            self.region_labels = numpy.array(region_labels).astype(str)
        elif mode in ("alphabetic", "alpha"):
            if self.number_of_regions < 26:
                self.region_labels = numpy.array(list(map(chr, list(range(65, 65 + self.number_of_regions))))).astype(
                    str)
            else:
                self.log.info("I'm too lazy to create several strategies to label regions. \\")
                self.log.info("Please choose mode 'numeric' or set your own labels\\")
        else:
            self.log.error("Bad region labels mode, must be one of:")
            self.log.error("('numeric', 'num', 'alphabetic', 'alpha')")
            raise ValidationException("Bad region labels mode")

    def unmapped_indices(self, region_mapping):
        """
        Compute vector of indices of regions in connectivity which are not in the given
        region mapping.

        """

        return numpy.setdiff1d(numpy.r_[:self.number_of_regions], region_mapping)

    @classmethod
    def _read(cls, reader):
        result = Connectivity()
        result.weights = reader.read_array_from_file("weights")
        if reader.has_file_like("centres"):
            result.centres = reader.read_array_from_file("centres", use_cols=(1, 2, 3))
            result.region_labels = reader.read_array_from_file("centres", dtype=numpy.str_, use_cols=(0,))
        else:
            result.centres = reader.read_array_from_file("centers", use_cols=(1, 2, 3))
            result.region_labels = reader.read_array_from_file("centers", dtype=numpy.str_, use_cols=(0,))
        result.orientations = reader.read_optional_array_from_file("average_orientations")
        result.cortical = reader.read_optional_array_from_file("cortical", dtype=numpy.bool_)
        result.hemispheres = reader.read_optional_array_from_file("hemispheres", dtype=numpy.bool_)
        result.areas = reader.read_optional_array_from_file("areas")
        result.tract_lengths = reader.read_array_from_file("tract_lengths")
        return result

    @classmethod
    def from_file(cls, source_file="connectivity_76.zip"):

        result = Connectivity()
        source_full_path = try_get_absolute_path("tvb_data.connectivity", source_file)

        if source_file.endswith(".h5"):
            reader = H5Reader(source_full_path)

            result.weights = reader.read_field("weights")
            result.centres = reader.read_field("centres")
            result.region_labels = reader.read_field("region_labels")
            result.orientations = reader.read_optional_field("orientations")
            result.cortical = reader.read_optional_field("cortical")
            result.hemispheres = reader.read_field("hemispheres")
            result.areas = reader.read_optional_field("areas")
            result.tract_lengths = reader.read_field("tract_lengths")

        else:
            reader = ZipReader(source_full_path)
            result = cls._read(reader)

        return result

    @classmethod
    def from_bytes_stream(cls, bytes_stream):
        """Construct a Connectivity from a stream of bytes."""
        reader = ZipReader(BytesIO(bytes_stream))
        return cls._read(reader)

    @property
    def horizon(self):
        """The horizon is the maximum number of steps required in memory for simulation."""
        return self._horizon

    def set_centres(self, centres, expected_number_of_nodes):
        """Fill positions"""
        if centres is None:
            raise ValidationException("Region centres are required for Connectivity Regions! "
                                      "We expect a file that contains *centres* inside the uploaded ZIP.")
        if expected_number_of_nodes < 2:
            raise ValidationException("A connectivity with at least 2 nodes is expected")

        self.centres = centres

    def set_region_labels(self, region_labels):
        if region_labels is not None:
            self.region_labels = region_labels

    def set_weights(self, weights, expected_number_of_nodes):
        if weights is not None:
            if weights.shape != (expected_number_of_nodes, expected_number_of_nodes):
                raise ValidationException("Unexpected shape for weights matrix!  Should be %d x %d " % (
                    expected_number_of_nodes, expected_number_of_nodes))
            self.weights = weights

    def set_tract_lengths(self, tract_lengths, expected_number_of_nodes):
        """Fill and check tracts. Allow empty files for tracts, they will be computed by tvb-library."""
        if tract_lengths is not None and tract_lengths.size != 0:
            if numpy.any([x < 0 for x in tract_lengths.flatten()]):
                raise ValidationException("Negative values are not accepted in tracts matrix! "
                                          "Please check your file, and use values >= 0")
            if tract_lengths.shape != (expected_number_of_nodes, expected_number_of_nodes):
                raise ValidationException("Unexpected shape for tracts matrix!  Should be %d x %d " % (
                    expected_number_of_nodes, expected_number_of_nodes))
        self.tract_lengths = tract_lengths

    def set_orientations(self, orientations, expected_number_of_nodes):
        if orientations is not None:
            if len(orientations) != expected_number_of_nodes:
                raise ValidationException("Invalid size for vector orientation. "
                                          "Expected the same as region-centers number %d" % expected_number_of_nodes)
            self.orientations = orientations

    def set_areas(self, areas, expected_number_of_nodes):
        if areas is not None:
            if len(areas) != expected_number_of_nodes:
                raise ValidationException("Invalid size for vector areas. "
                                          "Expected the same as region-centers number %d" % expected_number_of_nodes)
            self.areas = areas

    def set_cortical(self, cortical, expected_number_of_nodes):
        if cortical is not None:
            if len(cortical) != expected_number_of_nodes:
                raise ValidationException("Invalid size for vector cortical. "
                                          "Expected the same as region-centers number %d" % expected_number_of_nodes)
            self.cortical = cortical

    def set_hemispheres(self, hemispheres, expected_number_of_nodes):
        if hemispheres is not None:
            if len(hemispheres) != expected_number_of_nodes:
                raise ValidationException("Invalid size for vector hemispheres. "
                                          "Expected the same as region-centers number %d" % expected_number_of_nodes)
            self.hemispheres = hemispheres
