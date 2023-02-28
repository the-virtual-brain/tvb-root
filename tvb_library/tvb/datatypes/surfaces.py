# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
Surface relates DataTypes.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""
import warnings
import numpy
import scipy.sparse
from io import BytesIO

from tvb.basic import exceptions
from tvb.basic.neotraits.api import TVBEnum
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Final, Int, Float, narray_describe
from tvb.basic.readers import ZipReader, try_get_absolute_path

try:
    import gdist
except Exception:
    class ExceptionRaisingGdistModule(object):
        msg = "Geodesic distance module is unavailable, cannot compute gdist matrix."

        def local_gdist_matrix(self, *args, **kwds):
            raise RuntimeError(self.msg)

        def compute_gdist(self, *args, **kwds):
            raise RuntimeError(self.msg)


    gdist = ExceptionRaisingGdistModule()
    msg = "Geodesic distance module is unavailable; some functionality for surfaces will be unavailable."
    warnings.warn(msg)


class ValidationResult(object):
    """
    Used by surface validate methods to report non-fatal failed validations
    """

    def __init__(self, logger):
        self.warnings = []
        self.log = logger

    def add_warning(self, message, data):
        self.warnings.append((message, data))
        self.log.warning(message)
        if data:
            self.log.debug(data)

    def merge(self, other):
        r = ValidationResult(self.log)
        r.warnings = self.warnings + other.warnings
        return r

    def summary(self):
        return '  |  '.join(message for message, _ in self.warnings)


class SurfaceTypesEnum(TVBEnum):
    CORTICAL_SURFACE = "Cortical Surface"
    BRAIN_SKULL_SURFACE = "Brain Skull"
    SKULL_SKIN_SURFACE = "Skull Skin"
    SKIN_AIR_SURFACE = "Skin Air"
    EEG_CAP_SURFACE = "EEG Cap"
    FACE_SURFACE = "Face"
    WHITE_MATTER_SURFACE = "White Matter"
    KEY_OPTION_READ_METADATA = 'Specified in the file metadata'  # This last option will be displayed only for gifti
    # surface importer


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
        doc="""A sparse matrix of truncated geodesic distances""")

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

    hemisphere_mask = NArray(dtype=bool, required=False,
                             label="An array specifying if a vertex belongs to the right hemisphere")

    zero_based_triangles = Attr(field_type=bool)

    bi_hemispheric = Attr(field_type=bool, default=False)

    surface_type = Final(field_type=str)

    valid_for_simulations = Attr(field_type=bool, default=True)

    @classmethod
    def _read(cls, reader):
        result = cls()
        result.vertices = reader.read_array_from_file("vertices.txt")
        result.vertex_normals = reader.read_array_from_file("normals.txt")
        result.triangles = reader.read_array_from_file("triangles.txt", dtype=numpy.int32)
        return result

    @classmethod
    def from_file(cls, source_file="cortex_16384.zip"):
        """Construct a Surface from source_file."""
        source_full_path = try_get_absolute_path("tvb_data.surfaceData", source_file)
        reader = ZipReader(source_full_path)

        return cls._read(reader)

    @classmethod
    def from_bytes_stream(cls, bytes_stream, content_type='.zip'):
        """Construct a Surface from a stream of bytes."""

        reader = ZipReader(BytesIO(bytes_stream))
        return cls._read(reader)

    def set_scaled_vertices(self, new_vertices):
        # The vertex values can be too small to be seen with the viewers or widgets, thus we scale them a bit
        vertices_mean = numpy.mean(new_vertices)
        if vertices_mean < 0.1:
            new_vertices = new_vertices * 1000
        elif vertices_mean < 1:
            new_vertices = new_vertices * 100
        elif vertices_mean < 10:
            new_vertices = new_vertices * 10
        self.vertices = new_vertices

    def configure(self):
        """Compute additional attributes on surface data required for full functionality."""

        self.number_of_vertices = int(self.vertices.shape[0])
        self.number_of_triangles = int(self.triangles.shape[0])
        self.bi_hemispheric = self.hemisphere_mask is not None and numpy.unique(self.hemisphere_mask).size > 1

        if self.triangle_normals is None or self.triangle_normals.size == 0:
            self.log.debug("Triangle normals not available. Start to compute them.")
            self.compute_triangle_normals()
            self.log.debug("End computing triangles normals")

        if self.vertex_normals is None or self.vertex_normals.size == 0:
            self.log.debug("Vertex normals not available. Start to compute them.")
            self.compute_vertex_normals()
            self.log.debug("End computing vertex normals")

        if self._edge_lengths is None:
            self._edge_lengths = self.edge_lengths

    # from scientific surfaces
    _vertex_neighbours = None
    _vertex_triangles = None
    _triangle_centres = None
    _triangle_angles = None
    _triangle_areas = None
    _edges = None
    _number_of_edges = None
    _edge_lengths = None
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
            "Edge lengths, mean (mm)": self.edge_mean_length,
            "Edge lengths, shortest (mm)": self.edge_min_length,
            "Edge lengths, longest (mm)": self.edge_max_length
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
        ring = {vertex}
        local_vertices = {vertex}

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
        self.log.debug("triangle_normals")
        self.log.debug(narray_describe(self.triangle_normals))

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
            self.log.warning(" %d vertices have bad normals" % bad_normal_count)
        self.vertex_normals = vert_norms
        self.log.debug("vertex_normals")
        self.log.debug(narray_describe(self.vertex_normals))

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
        self.log.debug("triangle_areas")
        self.log.debug(narray_describe(triangle_areas))

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
        self.log.debug("tri_centres")
        self.log.debug(narray_describe(tri_centres))
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
        self.log.debug("triangle_angles")
        self.log.debug(narray_describe(angles))

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
            try:
                self._number_of_edges = len(self.edges)
            except Exception:
                return 0
        return self._number_of_edges

    @property
    def edge_lengths(self):
        """
        The length of the edges defined in the ``edges`` attribute.

        Calculate the Euclidean distance between the pair of vertices that
        define the edges in the ``edges`` attribute, when missing.
        """
        if self._edge_lengths is None:
            elem = numpy.sqrt(((self.vertices[self.edges, :][:, 0, :] -
                                self.vertices[self.edges, :][:, 1, :]) ** 2).sum(axis=1))

            self.edge_mean_length = float(elem.mean())
            self.edge_min_length = float(elem.min())
            self.edge_max_length = float(elem.max())

            self._edge_lengths = elem
        return self._edge_lengths

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
        r = ValidationResult(self.log)

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
        :param h: default 1.0
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

    def validate(self):
        self.number_of_vertices = self.vertices.shape[0]
        self.number_of_triangles = self.triangles.shape[0]

        if self.triangles.max() >= self.number_of_vertices:
            raise exceptions.ValidationException("There are triangles that index nonexistent vertices.")

        validation_result = self.validate_topology_for_simulations()

        self.valid_for_simulations = len(validation_result.warnings) == 0

        return validation_result

    @staticmethod
    def _triangles_to_lines(triangles):
        lines_array = []
        for a, b, c in triangles:
            lines_array.extend([a, b, b, c, c, a])
        return numpy.array(lines_array)

    def center(self):
        """
        Compute the center of the surface as the mean spot on all the three axes.
        """
        # is this different from return numpy.mean(self.vertices, axis=0) ?
        return [float(numpy.mean(self.vertices[:, 0])),
                float(numpy.mean(self.vertices[:, 1])),
                float(numpy.mean(self.vertices[:, 2]))]

    def compute_equation(self, focal_points, equation):
        """
        focal_points - a list of focal points. Used for specifying the vertices
        from which the distance is calculated.
        equation - the equation which should be evaluated
        """
        focal_points = numpy.array(focal_points, dtype=numpy.int32)
        dist = self.geodesic_distance(focal_points)
        return equation.evaluate(dist)


class WhiteMatterSurface(Surface):
    """White matter - gray matter interface surface."""
    surface_type = Final(field_type=str, default=SurfaceTypesEnum.WHITE_MATTER_SURFACE.value)


class CorticalSurface(Surface):
    """Cortical or pial surface."""
    surface_type = Final(field_type=str, default=SurfaceTypesEnum.CORTICAL_SURFACE.value)


class SkinAir(Surface):
    """Skin - air interface surface."""
    surface_type = Final(field_type=str, default=SurfaceTypesEnum.SKIN_AIR_SURFACE.value)

    @classmethod
    def from_file(cls, source_file="outer_skin_4096.zip"):
        return super(SkinAir, cls).from_file(source_file)


class BrainSkull(Surface):
    """Brain - inner skull interface surface."""
    surface_type = Final(field_type=str, default=SurfaceTypesEnum.BRAIN_SKULL_SURFACE.value)

    @classmethod
    def from_file(cls, source_file="inner_skull_4096.zip"):
        return super(BrainSkull, cls).from_file(source_file)


class SkullSkin(Surface):
    """Outer-skull - scalp interface surface."""
    surface_type = Final(field_type=str, default=SurfaceTypesEnum.SKULL_SKIN_SURFACE.value)

    @classmethod
    def from_file(cls, source_file="outer_skull_4096.zip"):
        return super(SkullSkin, cls).from_file(source_file)


class EEGCap(Surface):
    """EEG cap surface."""
    surface_type = Final(field_type=str, default=SurfaceTypesEnum.EEG_CAP_SURFACE.value)

    @classmethod
    def from_file(cls, source_file="scalp_1082.zip"):
        return super(EEGCap, cls).from_file(source_file)


class FaceSurface(Surface):
    """Face surface."""
    surface_type = Final(field_type=str, default=SurfaceTypesEnum.FACE_SURFACE.value)

    @classmethod
    def from_file(cls, source_file="face_8614.zip"):
        return super(FaceSurface, cls).from_file(source_file)


def make_surface(surface_type):
    """
    Build a Surface instance, based on an input type
    :param surface_type: one of the supported surface types
    :return: Instance of the corresponding surface lass, or None
    """
    if surface_type in [SurfaceTypesEnum.CORTICAL_SURFACE.value, "Pial"] or surface_type.startswith("Cortex"):
        return CorticalSurface()
    elif surface_type == SurfaceTypesEnum.BRAIN_SKULL_SURFACE.value:
        return BrainSkull()
    elif surface_type == SurfaceTypesEnum.SKULL_SKIN_SURFACE.value:
        return SkullSkin()
    elif surface_type in [SurfaceTypesEnum.SKIN_AIR_SURFACE.value, "SkinAir"]:
        return SkinAir()
    elif surface_type == SurfaceTypesEnum.EEG_CAP_SURFACE.value:
        return EEGCap()
    elif surface_type == SurfaceTypesEnum.FACE_SURFACE.value:
        return FaceSurface()
    elif surface_type == SurfaceTypesEnum.WHITE_MATTER_SURFACE.value:
        return WhiteMatterSurface()

    return None


def center_vertices(vertices):
    """
    Centres the vertices using means along axes.
    :param vertices: a numpy array of shape (n, 3)
    :returns: the centered array
    """
    return vertices - numpy.mean(vertices, axis=0).reshape((1, 3))
