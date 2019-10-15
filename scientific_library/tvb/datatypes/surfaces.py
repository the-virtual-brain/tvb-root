# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
Surface relates DataTypes. This brings together the scientific and framework 
methods that are associated with the surfaces data.

.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>
.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""
import scipy.sparse
import warnings
import json
import numpy
from tvb.basic import exceptions
from tvb.basic.logger.builder import get_logger
from tvb.basic.profile import TvbProfile
from tvb.basic.readers import ZipReader, try_get_absolute_path
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Final, Int, Float, narray_describe

try:
    import gdist
except ImportError:
    class ExceptionRaisingGdistModule:
        msg = "Geodesic distance module is unavailable, cannot compute gdist matrix."

        def local_gdist_matrix(self, *args, **kwds):
            raise RuntimeError(self.msg)

        def compute_gdist(self, *args, **kwds):
            raise RuntimeError(self.msg)

    gdist = ExceptionRaisingGdistModule()
    msg = "Geodesic distance module is unavailable; some functionality for surfaces will be unavailable."
    warnings.warn(msg)


LOG = get_logger(__name__)


OUTER_SKIN = "Skin Air"
OUTER_SKULL = "Skull Skin"
INNER_SKULL = "Brain Skull"
CORTICAL = "Cortical Surface"
WHITE_MATTER = "White Matter"
EEG_CAP = "EEG Cap"
FACE = "Face"

# Slices are for vertices [0.....SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE]
# [SPLIT_MAX_SIZE ..... 2 * SPLIT_BUFFER_SIZE + SPLIT_BUFFER_SIZE]
# Triangles [0, 1, 2], [3, 4, 5], [6, 7, 8].....
# Vertices -  no of triangles * 3

SPLIT_MAX_SIZE = 40000
SPLIT_BUFFER_SIZE = 15000
SPLIT_PICK_MAX_TRIANGLE = 20000

KEY_TRIANGLES = "triangles"
KEY_VERTICES = "vertices"
KEY_HEMISPHERE = "hemisphere"
KEY_START = "start_idx"
KEY_END = "end_idx"

HEMISPHERE_LEFT = "LEFT"
HEMISPHERE_RIGHT = "RIGHT"
HEMISPHERE_UNKNOWN = "NONE"


class ValidationResult(object):
    """
    Used by surface validate methods to report non-fatal failed validations
    """
    def __init__(self):
        self.warnings = []

    def add_warning(self, msg, data):
        self.warnings.append((msg, data))
        self._log(msg, data)

    def _log(self, msg, data):
        LOG.warn(msg)
        if data:
            LOG.debug(data)

    def merge(self, other):
        r = ValidationResult()
        r.warnings = self.warnings + other.warnings
        return r

    def summary(self):
        return '  |  '.join(msg for msg, _ in self.warnings)


class Surface(HasTraits):
    """A base class for other surfaces."""

    vertices = NArray(
        label="Vertex positions",
        doc="""An array specifying coordinates for the surface vertices.""")

    triangles = NArray(
        dtype=int,
        label="Triangles",
        doc="""Array of indices into the vertices, specifying the triangles which define the surface.""")

    vertex_normals = NArray(
        label="Vertex normal vectors",
        required=False,
        doc="""An array of unit normal vectors for the surfaces vertices.""")

    triangle_normals = NArray(
        label="Triangle normal vectors",
        required=False,
        doc="""An array of unit normal vectors for the surfaces triangles.""")

    geodesic_distance_matrix = Attr(
        field_type=scipy.sparse.csc_matrix,
        label="Geodesic distance matrix",
        required=False,
        # file_storage=FILE_STORAGE_NONE,
        doc="""A sparse matrix of truncated geodesic distances""")  # 'CS'

    number_of_vertices = Int(
        field_type=int,
        label="Number of vertices",
        doc="""The number of vertices making up this surface.""")

    number_of_triangles = Int(
        field_type=int,
        label="Number of triangles",
        doc="""The number of triangles making up this surface.""")

    edge_mean_length = Float()

    edge_min_length = Float()

    edge_max_length = Float()

    ##--------------------- FRAMEWORK ATTRIBUTES -----------------------------##

    hemisphere_mask = NArray(
        dtype=bool,
        label="An array specifying if a vertex belongs to the right hemisphere",
        # file_storage=FILE_STORAGE_NONE,
        required=False)

    zero_based_triangles = Attr(field_type=bool)

    split_triangles = NArray(dtype=int, required=False)

    number_of_split_slices = Int(required=False)

    split_slices = Attr(field_type=dict, required=False)

    bi_hemispheric = Attr(field_type=bool, default=False)

    surface_type = Attr(field_type=str)

    valid_for_simulations = Attr(field_type=bool, default=True)

    @classmethod
    def from_file(cls, source_file="cortex_16384.zip", instance=None):
        """Construct a Surface from source_file."""

        if instance is None:
            result = cls()
        else:
            result = instance

        source_full_path = try_get_absolute_path("tvb_data.surfaceData", source_file)
        reader = ZipReader(source_full_path)

        result.vertices = reader.read_array_from_file("vertices.txt")
        result.vertex_normals = reader.read_array_from_file("normals.txt")
        result.triangles = reader.read_array_from_file("triangles.txt", dtype=numpy.int32)

        return result


    def configure(self):
        "Compute additional attributes on surface data required for full functionality."

        self.number_of_vertices = int(self.vertices.shape[0])
        self.number_of_triangles = int(self.triangles.shape[0])

        if self.triangle_normals is None or self.triangle_normals.size == 0:
            LOG.debug("Triangle normals not available. Start to compute them.")
            self.compute_triangle_normals()
            LOG.debug("End computing triangles normals")

        if self.vertex_normals is None or self.vertex_normals.size == 0:
            LOG.debug("Vertex normals not available. Start to compute them.")
            self.compute_vertex_normals()
            LOG.debug("End computing vertex normals")

        if self._edge_lengths is None:
            self._find_edge_lengths()

        self.framework_configure()


    def validate(self):
        """
        Combines scientific and framework surface validations.
        """
        result_sci = self.scientific_validate()
        result_fr = self.framework_validate()

        validation_result = result_sci.merge(result_fr)

        self.user_tag_3 = validation_result.summary()
        return validation_result

    # from scientific surfaces
    _vertex_neighbours = None
    _vertex_triangles = None
    _triangle_centres = None
    _triangle_angles = None
    _triangle_areas = None
    _edges = None
    _number_of_edges = None
    _edge_lengths = None
    _edge_length_mean = None
    _edge_length_min = None
    _edge_length_max = None
    _edge_triangles = None

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        return {
            "Surface type": self.__class__.__name__,
            "Valid for simulations": self.valid_for_simulations,
            "Number of vertices": self.number_of_vertices,
            "Number of triangles": self.number_of_triangles,
            "Number of edges": self.number_of_edges,
            "Has two hemispheres": self.bi_hemispheric,
            "Edge lengths, mean (mm)": self.edge_length_mean,
            "Edge lengths, shortest (mm)": self.edge_length_min,
            "Edge lengths, longest (mm)": self.edge_length_max
        }

    def geodesic_distance(self, sources, max_dist=None, targets=None):
        """
        Calculate the geodesic distance between vertices of the surface,

        ``sources``: one or more indices into vertices, these are required,
            they specify the vertices from which the distance is calculated.
            NOTE: if multiple sources are provided then the distance returned
            is the shortest from the closest source.
        ``max_dist``: find the distance to vertices out as far as max_dist.
        ``targets``: one or more indices into vertices,.

        NOTE: Either ``targets`` or ``max_dist`` should be specified, but not
            both, specifying neither is equivalent to max_dist=1e100.

        NOTE: when max_dist is specifed, distances > max_dist are returned as
            numpy.inf

        If end_vertex is omitted the distance from the starting vertex to all
        vertices within max_dist will be returned, if max_dist is also omitted
        the distance to all vertices on the surface will be returned.

        """
        if max_dist is not None and targets is not None:
            raise ValueError("Specifying both targets and max_dist doesn't work.")

        # Cython expects data with specific dtype
        verts = self.vertices.astype(numpy.float64)
        tris = self.triangles.astype(numpy.int32)
        srcs = sources.astype(numpy.int32)
        kwd = {}

        # handle custom args
        if targets is not None:
            kwd['target_indices'] = targets.astype(numpy.int32)

        if max_dist is not None:
            kwd['max_distance'] = max_dist

        dist = gdist.compute_gdist(verts, tris, source_indices=srcs, **kwd)
        return dist

    # TODO why two methods for this?
    def compute_geodesic_distance_matrix(self, max_dist):
        """
        Calculate a sparse matrix of the geodesic distance from each vertex to
        all vertices within max_dist of them on the surface,

        ``max_dist``: find the distance to vertices out as far as max_dist.

        NOTE: Compute time increases rapidly with max_dist and the memory
        efficiency of the sparse matrices decreases, so, don't use too large a
        value for max_dist...

        """
        # TODO: Probably should check that max_dist isn't "too" large or too
        #      small, min should probably be max edge length...

        # if NO_GEODESIC_DISTANCE:
        #    LOG.error("%s: The geodesic distance library didn't load" % repr(self))
        #    return

        dist = gdist.local_gdist_matrix(self.vertices.astype(numpy.float64),
                                        self.triangles.astype(numpy.int32),
                                        max_distance=max_dist)

        self.geodesic_distance_matrix = dist

    @property
    def vertex_neighbours(self):
        """
        List of the set of neighbours for each vertex.
        """
        if self._vertex_neighbours is None:
            self._vertex_neighbours = self._find_vertex_neighbours()
        return self._vertex_neighbours

    def _find_vertex_neighbours(self):
        """
        .
        """
        neighbours = [[] for _ in range(self.number_of_vertices)]
        for k in range(self.number_of_triangles):
            neighbours[self.triangles[k, 0]].append(self.triangles[k, 1])
            neighbours[self.triangles[k, 0]].append(self.triangles[k, 2])
            neighbours[self.triangles[k, 1]].append(self.triangles[k, 0])
            neighbours[self.triangles[k, 1]].append(self.triangles[k, 2])
            neighbours[self.triangles[k, 2]].append(self.triangles[k, 0])
            neighbours[self.triangles[k, 2]].append(self.triangles[k, 1])

        neighbours = list(map(frozenset, neighbours))

        return neighbours

    @property
    def vertex_triangles(self):
        """
        List of the set of triangles surrounding each vertex.
        """
        if self._vertex_triangles is None:
            self._vertex_triangles = self._find_vertex_triangles()
        return self._vertex_triangles

    def _find_vertex_triangles(self):
        # self.attr calls __get__@type_mapped which is performance sensitive here
        triangles = self.triangles

        vertex_triangles = [[] for _ in range(self.number_of_vertices)]
        for k in range(self.number_of_triangles):
            vertex_triangles[triangles[k, 0]].append(k)
            vertex_triangles[triangles[k, 1]].append(k)
            vertex_triangles[triangles[k, 2]].append(k)

        vertex_triangles = list(map(frozenset, vertex_triangles))

        return vertex_triangles

    def nth_ring(self, vertex, neighbourhood=2, contains=False):
        """
        Return the vertices of the nth ring around a given vertex, defaults to
        neighbourhood=2. NOTE: if you want neighbourhood=1 then you should
        directly access the property vertex_neighbours, ie use
        surf_obj.vertex_neighbours[vertex] setting contains=True returns all
        vertices from rings 1 to n inclusive.
        """

        ring = set([vertex])
        local_vertices = set([vertex])

        for _ in range(neighbourhood):
            neighbours = [self.vertex_neighbours[indx] for indx in ring]
            neighbours = set(vert for subset in neighbours for vert in subset)
            ring = neighbours.difference(local_vertices)
            local_vertices.update(ring)

        if contains:
            local_vertices.discard(vertex)
            return frozenset(local_vertices)
        return frozenset(ring)

    def compute_triangle_normals(self):
        """Calculates triangle normals."""
        tri_u = self.vertices[self.triangles[:, 1], :] - self.vertices[self.triangles[:, 0], :]
        tri_v = self.vertices[self.triangles[:, 2], :] - self.vertices[self.triangles[:, 0], :]

        tri_norm = numpy.cross(tri_u, tri_v)

        try:
            self.triangle_normals = tri_norm / numpy.sqrt(numpy.sum(tri_norm ** 2, axis=1))[:, numpy.newaxis]
        except FloatingPointError:
            # TODO: NaN generation would stop execution, however for normals this case could maybe be
            # handled in a better way.
            self.triangle_normals = tri_norm
        LOG.debug("triangle_normals")
        LOG.debug(narray_describe(self.triangle_normals))

    def compute_vertex_normals(self):
        """
        Estimates vertex normals, based on triangle normals weighted by the
        angle they subtend at each vertex...
        """
        vert_norms = numpy.zeros((self.number_of_vertices, 3))
        bad_normal_count = 0
        for k in range(self.number_of_vertices):
            try:
                tri_list = list(self.vertex_triangles[k])
                angle_mask = self.triangles[tri_list, :] == k
                angles = self.triangle_angles[tri_list, :]
                angles = angles[angle_mask][:, numpy.newaxis]
                angle_scaling = angles / numpy.sum(angles, axis=0)
                vert_norms[k, :] = numpy.mean(angle_scaling * self.triangle_normals[tri_list, :], axis=0)
                # Scale by angle subtended.
                vert_norms[k, :] = vert_norms[k, :] / numpy.sqrt(numpy.sum(vert_norms[k, :] ** 2, axis=0))
                # Normalise to unit vectors.
            except (ValueError, FloatingPointError):
                # If normals are bad, default to position vector
                # A nicer solution would be to detect degenerate triangles and ignore their
                # contribution to the vertex normal
                vert_norms[k, :] = self.vertices[k] / numpy.sqrt(self.vertices[k].dot(self.vertices[k]))
                bad_normal_count += 1
        if bad_normal_count:
            self.logger.warn(" %d vertices have bad normals" % bad_normal_count)
        self.vertex_normals = vert_norms
        LOG.debug("vertex_normals")
        LOG.debug(narray_describe(self.vertex_normals))

    @property
    def triangle_areas(self):
        """An array specifying the area of the triangles making up a surface."""
        if self._triangle_areas is None:
            self._triangle_areas = self._find_triangle_areas()
        return self._triangle_areas

    def _find_triangle_areas(self):
        """Calculates the area of triangles making up a surface."""
        tri_u = self.vertices[self.triangles[:, 1], :] - self.vertices[self.triangles[:, 0], :]
        tri_v = self.vertices[self.triangles[:, 2], :] - self.vertices[self.triangles[:, 0], :]

        tri_norm = numpy.cross(tri_u, tri_v)
        triangle_areas = numpy.sqrt(numpy.sum(tri_norm ** 2, axis=1)) / 2.0
        triangle_areas = triangle_areas[:, numpy.newaxis]
        LOG.debug("triangle_areas")
        LOG.debug(narray_describe(triangle_areas))

        return triangle_areas

    @property
    def triangle_centres(self):
        """
        An array specifying the location of triangle centres.
        """
        if self._triangle_centres is None:
            self._triangle_centres = self._find_triangle_centres()
        return self._triangle_centres

    def _find_triangle_centres(self):
        """
        Calculate the location of the centre of all triangles comprising the mesh surface.
        """
        tri_verts = self.vertices[self.triangles, :]
        tri_centres = numpy.mean(tri_verts, axis=1)
        LOG.debug("tri_centres")
        LOG.debug(narray_describe(tri_centres))
        return tri_centres

    @property
    def triangle_angles(self):
        """
        An array containing the inner angles for each triangle, same shape as triangles.
        """
        if self._triangle_angles is None:
            self._triangle_angles = self._find_triangle_angles()
        return self._triangle_angles

    def _normalized_edge_vectors(self):
        """ for triangle abc computes the normalized vector edges b-a c-a c-b """
        tri_verts = self.vertices[self.triangles]
        tri_verts[:, 2, :] -= tri_verts[:, 0, :]
        tri_verts[:, 1, :] -= tri_verts[:, 0, :]
        tri_verts[:, 0, :] = tri_verts[:, 2, :] - tri_verts[:, 1, :]
        # normalize
        tri_verts /= numpy.sqrt(numpy.sum(tri_verts ** 2, axis=2, keepdims=True))
        return tri_verts

    def _find_triangle_angles(self):
        """
        Calculates the inner angles of all the triangles which make up a surface
        """

        def _angle(a, b):
            """ Angle between normalized vectors. <a|b> = cos(alpha)"""
            return numpy.arccos(numpy.sum(a * b, axis=1, keepdims=True))

        edges = self._normalized_edge_vectors()
        a0 = _angle(edges[:, 1, :], edges[:, 2, :])
        a1 = _angle(edges[:, 0, :], - edges[:, 1, :])
        a2 = 2 * numpy.pi - a0 - a1
        angles = numpy.hstack([a0, a1, a2])
        LOG.debug("triangle_angles")
        LOG.debug(narray_describe(angles))

        return angles

    @property
    def edges(self):
        """
        A sorted list of the two element tuples(vertex_0, vertex_1) representing
        the edges of the mesh.
        """
        if self._edges is None:
            self._edges = self._find_edges()
        return self._edges

    def _find_edges(self):
        """
        Find all the edges of the mesh surface, return them sorted as a list of
        two element tuple, where the elements are vertex indices.
        """
        v0 = numpy.vstack((self.triangles[:, 0][:, numpy.newaxis],
                           self.triangles[:, 0][:, numpy.newaxis],
                           self.triangles[:, 1][:, numpy.newaxis]))
        v1 = numpy.vstack((self.triangles[:, 1][:, numpy.newaxis],
                           self.triangles[:, 2][:, numpy.newaxis],
                           self.triangles[:, 2][:, numpy.newaxis]))
        edges = numpy.hstack((v0, v1))
        edges.sort(axis=1)
        edges = set(tuple(edges[k]) for k in range(edges.shape[0]))
        edges = sorted(edges)
        return edges

    @property
    def number_of_edges(self):
        """
        The number of edges making up the mesh surface.
        """
        if self._number_of_edges is None:
            self._number_of_edges = len(self.edges)
        return self._number_of_edges

    @property
    def edge_lengths(self):
        """
        The length of the edges defined in the ``edges`` attribute.
        """
        if self._edge_lengths is None:
            self._edge_lengths = self._find_edge_lengths()
        return self._edge_lengths

    def _find_edge_lengths(self):
        """
        Calculate the Euclidean distance between the pair of vertices that
        define the edges in the ``edges`` attribute.
        """
        # TODO: Would a Sparse matrix be a more useful data structure for these???
        elem = numpy.sqrt(((self.vertices[self.edges, :][:, 0, :] -
                            self.vertices[self.edges, :][:, 1, :]) ** 2).sum(axis=1))

        self.edge_mean_length = float(elem.mean())
        self.edge_min_length = float(elem.min())
        self.edge_max_length = float(elem.max())

        return elem

    @property
    def edge_length_mean(self):
        """The mean length of the edges of the mesh."""
        if self.edge_mean_length is None:
            self._find_edge_lengths()
        return self.edge_mean_length

    @property
    def edge_length_min(self):
        """The length of the shortest edge in the mesh."""
        if self.edge_min_length is None:
            self._find_edge_lengths()
        return self.edge_min_length

    @property
    def edge_length_max(self):
        """The length of the longest edge in the mesh."""
        if self.edge_max_length is None:
            self._find_edge_lengths()
        return self.edge_max_length

    @property
    def edge_triangles(self):
        """
        List of the pairs of triangles sharing an edge.
        """
        if self._edge_triangles is None:
            self._edge_triangles = self._find_edge_triangles()
        return self._edge_triangles

    def _find_edge_triangles(self):
        triangles = [None] * self.number_of_edges
        for k in range(self.number_of_edges):
            triangles[k] = (frozenset(self.vertex_triangles[self.edges[k][0]]) &
                            frozenset(self.vertex_triangles[self.edges[k][1]]))
        return triangles

    def compute_topological_constants(self):
        """
        Returns a 4 tuple:
         * the Euler characteristic number
         * indices for any isolated vertices
         * indices of edges where the surface is pinched
         * indices of edges that border holes in the surface
        We call isolated vertices those who do not belong to at least 3 triangles.
        """
        euler = self.number_of_vertices + self.number_of_triangles - self.number_of_edges
        triangles_per_vertex = numpy.array(list(map(len, self.vertex_triangles)))
        isolated = numpy.nonzero(triangles_per_vertex < 3)
        triangles_per_edge = numpy.array(list(map(len, self.edge_triangles)))
        pinched_off = numpy.nonzero(triangles_per_edge > 2)
        holes = numpy.nonzero(triangles_per_edge < 2)
        return euler, isolated[0], pinched_off[0], holes[0]

    def validate_topology_for_simulations(self):
        """
        Validates if this surface can be used in simulations.
        The surface should be topologically equivalent to one or two closed spheres.
        It should not contain isolated vertices.
        It should not be pinched or have holes: all edges must belong to 2 triangles.
        The allowance for one or two closed surfaces is because the skull/etc
        should be represented by a single closed surface and we typically
        represent the cortex as one closed surface per hemisphere.

        :return: a ValidationResult
        """
        r = ValidationResult()

        euler, isolated, pinched_off, holes = self.compute_topological_constants()

        # The Euler characteristic for a 2D sphere embedded in a 3D space is 2.
        # This should be 2 or 4 -- meaning one or two closed topologically spherical surfaces
        if euler not in (2, 4):
            r.add_warning("Topologically not 1 or 2 spheres.", "Euler characteristic: " + str(euler))

        if len(isolated):
            r.add_warning("Has isolated vertices.", "Offending indices: \n" + str(isolated))

        if len(pinched_off):
            r.add_warning("Surface is pinched off.",
                          "These are edges with more than 2 triangles: \n" + str(pinched_off))

        if len(holes):
            r.add_warning("Has holes.", "Free boundaries: \n" + str(holes))

        return r

    def laplace_beltrami(self, fv, h=1.0):
        """
        Evaluates the discrete Laplace-Beltrami operator for a given vertex-wise function
        and geodesic distance matrix. From Belkin 2008:

        Let K be a mesh in R^3, with V its set of vertices. For face t in K, the number of vertices in t is #t, and
        V(t) is the set of vertices in t. For a function f : V -> R, this produces another function L_K^h f : V -> R,
        and L_K^h is computed for any w in V,

          L_K^h f (w) = 1 / (4 pi h^2) sum_{t in K} area(t) / #t sum_{p in V(t)} exp(-||p - w||^2/(4*h)) (f(p) - f(w))


        :param fv: a function evaluated on each vertex, shape (n, )
        :return: matrix of evaluated L-B operator

        """

        from math import exp, pi

        assert fv.shape[0] == self.vertices.shape[0]
        assert hasattr(self, 'geodesic_distance_matrix')

        lbo = numpy.zeros_like(fv)
        gd = self.geodesic_distance_matrix
        for w in range(len(self.vertices)):
            mesh_sum = 0.0
            for t in range(len(self.triangles)):
                face_sum = 0.0
                for p in self.triangles[t]:
                    face_sum += exp(-gd[w, p] ** 2 / (4 * h)) * (fv[p] - fv[w])
                face_sum *= self.triangle_areas[t] / 3.0
                mesh_sum += face_sum
            lbo[w] = mesh_sum / (4.0 * pi * h ** 2)

        return lbo

    def scientific_validate(self):
        self.number_of_vertices = self.vertices.shape[0]
        self.number_of_triangles = self.triangles.shape[0]

        if self.triangles.max() >= self.number_of_vertices:
            raise exceptions.ValidationException("There are triangles that index nonexistent vertices.")

        validation_result = self.validate_topology_for_simulations()

        self.valid_for_simulations = len(validation_result.warnings) == 0

        return validation_result

    def compute_equation(self, focal_points, equation):
        """
        focal_points - a list of focal points. Used for specifying the vertices
        from which the distance is calculated.
        equation - the equation which should be evaluated
        """
        focal_points = numpy.array(focal_points, dtype=numpy.int32)
        dist = self.geodesic_distance(focal_points)
        return equation.evaluate(dist)

    # framework methods
    def load_from_metadata(self, meta_dictionary):
        self.edge_mean_length = 0
        self.edge_min_length = 0
        self.edge_max_length = 0
        self.valid_for_simulations = True
        super(Surface, self).load_from_metadata(meta_dictionary)

    @staticmethod
    def _triangles_to_lines(triangles):
        lines_array = []
        for a, b, c in triangles:
            lines_array.extend([a, b, b, c, c, a])
        return numpy.array(lines_array)

    def framework_configure(self):
        """
        Before storing Surface in DB, make sure vertices/triangles are split in
        slices that are readable by WebGL.
        WebGL only supports triangle indices in interval [0.... 2^16]
        """
        # super(SurfaceFramework, self).configure()

        self.number_of_vertices = int(self.vertices.shape[0])
        self.number_of_triangles = int(self.triangles.shape[0])

        ### Do not split again, if split-data is already computed:
        if 1 < self.number_of_split_slices == len(self.split_slices):
            return

        ### Do not split when size is conveniently small:
        self.bi_hemispheric = self.hemisphere_mask is not None and numpy.unique(self.hemisphere_mask).size > 1
        if self.number_of_vertices <= SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE and not self.bi_hemispheric:
            self.number_of_split_slices = 1
            self.split_slices = {0: {KEY_TRIANGLES: {KEY_START: 0, KEY_END: self.number_of_triangles},
                                     KEY_VERTICES: {KEY_START: 0, KEY_END: self.number_of_vertices},
                                     KEY_HEMISPHERE: HEMISPHERE_UNKNOWN}}
            return

        ### Compute the number of split slices:
        left_hemisphere_slices = 0
        left_hemisphere_vertices_no = 0
        if self.bi_hemispheric:
            ## when more than one hemisphere
            right_hemisphere_vertices_no = numpy.count_nonzero(self.hemisphere_mask)
            left_hemisphere_vertices_no = self.number_of_vertices - right_hemisphere_vertices_no
            LOG.debug("Right %d Left %d" % (right_hemisphere_vertices_no, left_hemisphere_vertices_no))
            left_hemisphere_slices = self._get_slices_number(left_hemisphere_vertices_no)
            self.number_of_split_slices = left_hemisphere_slices
            self.number_of_split_slices += self._get_slices_number(right_hemisphere_vertices_no)
            LOG.debug("Hemispheres Total %d Left %d" % (self.number_of_split_slices, left_hemisphere_slices))
        else:
            ## when a single hemisphere
            self.number_of_split_slices = self._get_slices_number(self.number_of_vertices)

        LOG.debug("Start to compute surface split triangles and vertices")
        split_triangles = []
        ignored_triangles_counter = 0
        self.split_slices = {}

        for i in range(self.number_of_split_slices):
            split_triangles.append([])
            if not self.bi_hemispheric:
                self.split_slices[i] = {KEY_VERTICES: {KEY_START: i * SPLIT_MAX_SIZE,
                                                       KEY_END: min(self.number_of_vertices,
                                                                    (i + 1) * SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE)},
                                        KEY_HEMISPHERE: HEMISPHERE_UNKNOWN}
            else:
                if i < left_hemisphere_slices:
                    self.split_slices[i] = {KEY_VERTICES: {KEY_START: i * SPLIT_MAX_SIZE,
                                                           KEY_END: min(left_hemisphere_vertices_no,
                                                                        (i + 1) * SPLIT_MAX_SIZE + SPLIT_BUFFER_SIZE)},
                                            KEY_HEMISPHERE: HEMISPHERE_LEFT}
                else:
                    self.split_slices[i] = {KEY_VERTICES: {KEY_START: left_hemisphere_vertices_no +
                                                                      (i - left_hemisphere_slices) * SPLIT_MAX_SIZE,
                                                           KEY_END: min(self.number_of_vertices,
                                                                        left_hemisphere_vertices_no + SPLIT_MAX_SIZE *
                                                                        (i + 1 - left_hemisphere_slices)
                                                                        + SPLIT_BUFFER_SIZE)},
                                            KEY_HEMISPHERE: HEMISPHERE_RIGHT}

        ### Iterate Triangles and find the slice where it fits best, based on its vertices indexes:
        for i in range(self.number_of_triangles):
            current_triangle = [self.triangles[i][j] for j in range(3)]
            fit_slice, transformed_triangle = self._find_slice(current_triangle)

            if fit_slice is not None:
                split_triangles[fit_slice].append(transformed_triangle)
            else:
                # triangle ignored, as it has vertices over multiple slices.
                ignored_triangles_counter += 1
                continue

        final_split_triangles = []
        last_triangles_idx = 0

        ### Concatenate triangles, to be stored in a single HDF5 array.
        for slice_idx, split_ in enumerate(split_triangles):
            self.split_slices[slice_idx][KEY_TRIANGLES] = {KEY_START: last_triangles_idx,
                                                           KEY_END: last_triangles_idx + len(split_)}
            final_split_triangles.extend(split_)
            last_triangles_idx += len(split_)
        self.split_triangles = numpy.array(final_split_triangles, dtype=numpy.int32)

        if ignored_triangles_counter > 0:
            LOG.warning("Ignored triangles from multiple hemispheres: " + str(ignored_triangles_counter))
        LOG.debug("End compute surface split triangles and vertices " + str(self.split_slices))

    def framework_validate(self):
        # First check if the surface has a valid number of vertices
        self.number_of_vertices = self.vertices.shape[0]
        self.number_of_triangles = self.triangles.shape[0]

        if self.number_of_vertices > TvbProfile.current.MAX_SURFACE_VERTICES_NUMBER:
            msg = "This surface has too many vertices (max: %d)." % TvbProfile.current.MAX_SURFACE_VERTICES_NUMBER
            msg += " Please upload a new surface or change max number in application settings."
            raise exceptions.ValidationException(msg)
        return ValidationResult()

    def _get_slice_vertex_boundaries(self, slice_idx):
        if str(slice_idx) in self.split_slices:
            start_idx = max(0, self.split_slices[str(slice_idx)][KEY_VERTICES][KEY_START])
            end_idx = min(self.split_slices[str(slice_idx)][KEY_VERTICES][KEY_END], self.number_of_vertices)
            return start_idx, end_idx
        else:
            LOG.warn("Could not access slice indices, possibly due to an incompatibility with code update!")
            return 0, min(SPLIT_BUFFER_SIZE, self.number_of_vertices)

    def _get_slice_triangle_boundaries(self, slice_idx):
        if str(slice_idx) in self.split_slices:
            start_idx = max(0, self.split_slices[str(slice_idx)][KEY_TRIANGLES][KEY_START])
            end_idx = min(self.split_slices[str(slice_idx)][KEY_TRIANGLES][KEY_END], self.number_of_triangles)
            return start_idx, end_idx
        else:
            LOG.warn("Could not access slice indices, possibly due to an incompatibility with code update!")
            return 0, self.number_of_triangles

    @staticmethod
    def _get_slices_number(vertices_number):
        """
        Slices are for vertices [SPLIT_MAX_SIZE * i ... SPLIT_MAX_SIZE * (i + 1) + SPLIT_BUFFER_SIZE]
        Slices will overlap :
        |........SPLIT_MAX_SIZE|...SPLIT_BUFFER_SIZE|                           <-- split 1
                               |......... SPLIT_MAX_SIZE|...SPLIT_BUFFER_SIZE|  <-- split 2
        If we have trailing data smaller than the SPLIT_BUFFER_SIZE,
        then we no longer split but we need to have at least 1 slice.
        """
        slices_number, trailing = divmod(vertices_number, SPLIT_MAX_SIZE)
        if trailing > SPLIT_BUFFER_SIZE or (slices_number == 0 and trailing > 0):
            slices_number += 1
        return slices_number

    def _find_slice(self, triangle):
        split_slices = self.split_slices  # because of performance: 1.5 times slower without this
        mn = min(triangle)
        mx = max(triangle)
        for i in range(self.number_of_split_slices):
            v = split_slices[i][KEY_VERTICES]  # extracted for performance
            slice_start = v[KEY_START]
            if slice_start <= mn and mx < v[KEY_END]:
                return i, [triangle[j] - slice_start for j in range(3)]
        return None, triangle

    ####################################### Split for Picking
    #######################################


    def center(self):
        """
        Compute the center of the surface as the mean spot on all the three axes.
        """
        # is this different from return numpy.mean(self.vertices, axis=0) ?
        return [float(numpy.mean(self.vertices[:, 0])),
                float(numpy.mean(self.vertices[:, 1])),
                float(numpy.mean(self.vertices[:, 2]))]

    @staticmethod
    def _process_triangle(triangle, reg_idx1, reg_idx2, dangling_idx, indices_offset,
                          region_mapping_array, vertices, normals):
        """
        Process a triangle and generate the required data for a region separation.
        :param triangle: the actual triangle as a 3 element vector
        :param reg_idx1: the first vertex that is in a 'conflicting' region
        :param reg_idx2: the second vertex that is in a 'conflicting' region
        :param dangling_idx: the third vector for which we know nothing yet.
                    Depending on this we might generate a line, or a 3 star centered in the triangle
        :param indices_offset: to take into account the slicing
        :param region_mapping_array: the region mapping raw array for which the regions are computed
        :param vertices: the current vertex slice
        :param normals: the current normals slice
        """

        def _star_triangle(point0, point1, point2, result_array):
            """
            Helper function that for a given triangle generates a 3-way star centered in the triangle center
            """
            center_vertex = [(point0[i] + point1[i] + point2[i]) / 3 for i in range(3)]
            mid_line1 = [(point0[i] + point1[i]) / 2 for i in range(3)]
            mid_line2 = [(point1[i] + point2[i]) / 2 for i in range(3)]
            mid_line3 = [(point2[i] + point0[i]) / 2 for i in range(3)]
            result_array.extend(center_vertex)
            result_array.extend(mid_line1)
            result_array.extend(mid_line2)
            result_array.extend(mid_line3)

        def _slice_triangle(point0, point1, point2, result_array):
            """
            Helper function that for a given triangle generates a line cutting thtough the middle of two edges.
            """
            mid_line1 = [(point0[i] + point1[i]) / 2 for i in range(3)]
            mid_line2 = [(point0[i] + point2[i]) / 2 for i in range(3)]
            result_array.extend(mid_line1)
            result_array.extend(mid_line2)

        # performance opportunity: we are computing some values available in caller

        p0 = vertices[triangle[reg_idx1] - indices_offset]
        p1 = vertices[triangle[reg_idx2] - indices_offset]
        p2 = vertices[triangle[dangling_idx] - indices_offset]
        n0 = normals[triangle[reg_idx1] - indices_offset]
        n1 = normals[triangle[reg_idx2] - indices_offset]
        n2 = normals[triangle[dangling_idx] - indices_offset]
        result_vertices = []
        result_normals = []

        dangling_reg = region_mapping_array[triangle[dangling_idx]]
        reg_1 = region_mapping_array[triangle[reg_idx1]]
        reg_2 = region_mapping_array[triangle[reg_idx2]]

        if dangling_reg != reg_1 and dangling_reg != reg_2:
            # Triangle is actually spanning 3 regions. Create a vertex in the center of the triangle, which connects to
            # the middle of each edge
            _star_triangle(p0, p1, p2, result_vertices)
            _star_triangle(n0, n1, n2, result_normals)
            result_lines = [0, 1, 0, 2, 0, 3]
        elif dangling_reg == reg_1:
            # Triangle spanning only 2 regions, draw a line through the middle of the triangle
            _slice_triangle(p1, p0, p2, result_vertices)
            _slice_triangle(n1, n0, n2, result_normals)
            result_lines = [0, 1]
        else:
            # Triangle spanning only 2 regions, draw a line through the middle of the triangle
            _slice_triangle(p0, p1, p2, result_vertices)
            _slice_triangle(n0, n1, n2, result_normals)
            result_lines = [0, 1]
        return result_vertices, result_lines, result_normals



class WhiteMatterSurface(Surface):
    """White matter - gray matter interface surface."""
    _ui_name = "A white matter - gray  surface"
    surface_type = Final(WHITE_MATTER)


class CorticalSurface(Surface):
    """Cortical or pial surface."""
    _ui_name = "A cortical surface"
    surface_type = Attr(field_type=str, default=CORTICAL)


class SkinAir(Surface):
    """Skin - air interface surface."""
    _ui_name = "Skin"
    surface_type = Final(OUTER_SKIN)

    @classmethod
    def from_file(cls, source_file="outer_skin_4096.zip", instance=None):
        return super(SkinAir, cls).from_file(source_file, instance)


class BrainSkull(Surface):
    """Brain - inner skull interface surface."""
    _ui_name = "Brain - inner skull interface surface."
    surface_type = Final(INNER_SKULL)

    @classmethod
    def from_file(cls, source_file="inner_skull_4096.zip", instance=None):
        return super(BrainSkull, cls).from_file(source_file, instance)



class SkullSkin(Surface):
    """Outer-skull - scalp interface surface."""
    _ui_name = "Outer-skull - scalp interface surface"
    surface_type = Final(OUTER_SKULL)

    @classmethod
    def from_file(cls, source_file="outer_skull_4096.zip", instance=None):
        return super(SkullSkin, cls).from_file(source_file, instance)


class OpenSurface(Surface):
    """Base class for open surfaces."""


class EEGCap(OpenSurface):
    """EEG cap surface."""
    _ui_name = "EEG Cap"
    surface_type = Final(EEG_CAP)

    @classmethod
    def from_file(cls, source_file="scalp_1082.zip", instance=None):
        return super(EEGCap, cls).from_file(source_file, instance)


class FaceSurface(OpenSurface):
    """Face surface."""
    _ui_name = "Face surface"
    surface_type = Final(FACE)

    @classmethod
    def from_file(cls, source_file="face_8614.zip", instance=None):
        return super(FaceSurface, cls).from_file(source_file, instance)


def make_surface(surface_type):
    """
    Build a Surface instance, based on an input type
    :param surface_type: one of the supported surface types
    :return: Instance of the corresponding surface lass, or None
    """
    if surface_type in [CORTICAL, "Pial"] or surface_type.startswith("Cortex"):
        return CorticalSurface()
    elif surface_type == INNER_SKULL:
        return BrainSkull()
    elif surface_type == OUTER_SKULL:
        return SkullSkin()
    elif surface_type in [OUTER_SKIN, "SkinAir"]:
        return SkinAir()
    elif surface_type == EEG_CAP:
        return EEGCap()
    elif surface_type == FACE:
        return FaceSurface()
    elif surface_type == WHITE_MATTER:
        return WhiteMatterSurface()

    return None


def center_vertices(vertices):
    """
    Centres the vertices using means along axes.
    :param vertices: a numpy array of shape (n, 3)
    :returns: the centered array
    """
    return vertices - numpy.mean(vertices, axis=0).reshape((1, 3))


ALL_SURFACES_SELECTION = {'Cortical Surface': CORTICAL,
                          'Brain Skull': INNER_SKULL,
                          'Skull Skin': OUTER_SKULL,
                          'Skin Air': OUTER_SKIN,
                          'EEG Cap': EEG_CAP,
                          'Face Surface': FACE,
                          'White Matter Surface': WHITE_MATTER}
