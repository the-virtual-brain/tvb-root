# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

Scientific methods for the Surfaces datatype.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia@tvb.invalid>

"""

import numpy
import scipy.sparse as sparse
import tvb.datatypes.surfaces_data as surfaces_data
import tvb.basic.traits.util as util
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)

try:  # externals.geodesic_distance.
    import gdist
    #NO_GEODESIC_DISTANCE = False
except ImportError:
    #NO_GEODESIC_DISTANCE = True
    LOG.error("Failed to import geodesic distance package from externals...")
    LOG.error("Check it's configured, externals/geodesic_distance/setup.py")



class SurfaceScientific(surfaces_data.SurfaceData):
    """ This class exists to add scientific methods to Surface """

    __tablename__ = None
    _vertex_neighbours = None
    _vertex_triangles = None
    _vertex_mapping = None
    _region_sum = None
    _region_average = None
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


    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialization.
        """
        super(SurfaceScientific, self).configure()

        self.number_of_vertices = self.vertices.shape[0]
        self.number_of_triangles = self.triangles.shape[0]

        if self.triangle_normals.size == 0:
            LOG.debug("Triangle normals not available. Start to compute them.")
            self.compute_triangle_normals()
            LOG.debug("End computing triangles normals")

        if self.vertex_normals.size == 0:
            LOG.debug("Vertex normals not available. Start to compute them.")
            self.compute_vertex_normals()
            LOG.debug("End computing vertex normals")


    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        summary = {"Surface type": self.__class__.__name__,
                   "Number of vertices": self.number_of_vertices,
                   "Number of triangles": self.number_of_triangles,
                   "Number of edges": self.number_of_edges,
                   "Has two hemispheres": self.bi_hemispheric,
                   "Edge lengths, mean (mm)": self.edge_length_mean,
                   "Edge lengths, shortest (mm)": self.edge_length_min,
                   "Edge lengths, longest (mm)": self.edge_length_max}
        return summary


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
        #TODO: Probably should check that targets and start_vertex are less than
        #      number of vertices, etc...
        #if NO_GEODESIC_DISTANCE:
        #    LOG.error("%s: The geodesic distance library didn't load" % repr(self))
        #    return

        if (max_dist is None) and (targets is None):
            dist = gdist.compute_gdist(self.vertices.astype(numpy.float64),
                                       self.triangles.astype(numpy.int32),
                                       source_indices=sources.astype(numpy.int32))
        elif (max_dist is None) and (targets is not None):
            dist = gdist.compute_gdist(self.vertices.astype(numpy.float64),
                                       self.triangles.astype(numpy.int32),
                                       source_indices=sources.astype(numpy.int32),
                                       target_indices=targets.astype(numpy.int32))
        elif (max_dist is not None) and (targets is None):
            dist = gdist.compute_gdist(self.vertices.astype(numpy.float64),
                                       self.triangles.astype(numpy.int32),
                                       source_indices=sources.astype(numpy.int32),
                                       max_distance=max_dist)
        else:
            LOG.error("%s: Specifying both targets and max_dist doesn't work." % str(self))
            dist = None

        return dist


    def compute_geodesic_distance_matrix(self, max_dist):
        """
        Calculate a sparse matrix of the geodesic distance from each vertex to 
        all vertices within max_dist of them on the surface, 

        ``max_dist``: find the distance to vertices out as far as max_dist.

        NOTE: Compute time increases rapidly with max_dist and the memory 
        efficiency of the sparse matrices decreases, so, don't use too large a 
        value for max_dist...

        """
        #TODO: Probably should check that max_dist isn't "too" large or too 
        #      small, min should probably be max edge length...

        #if NO_GEODESIC_DISTANCE:
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
        neighbours = [[] for _ in xrange(self.number_of_vertices)]
        for k in xrange(self.number_of_triangles):
            neighbours[self.triangles[k, 0]].append(self.triangles[k, 1])
            neighbours[self.triangles[k, 0]].append(self.triangles[k, 2])
            neighbours[self.triangles[k, 1]].append(self.triangles[k, 0])
            neighbours[self.triangles[k, 1]].append(self.triangles[k, 2])
            neighbours[self.triangles[k, 2]].append(self.triangles[k, 0])
            neighbours[self.triangles[k, 2]].append(self.triangles[k, 1])

        neighbours = map(frozenset, neighbours)

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
        """
        .
        """
        triangles = [[] for _ in xrange(self.number_of_vertices)]
        for k in xrange(self.number_of_triangles):
            triangles[self.triangles[k, 0]].append(k)
            triangles[self.triangles[k, 1]].append(k)
            triangles[self.triangles[k, 2]].append(k)

        triangles = map(frozenset, triangles)

        return triangles


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
            #TODO: NaN generation would stop execution, however for normals this case could maybe be 
            # handled in a better way.
            self.triangle_normals = tri_norm
        util.log_debug_array(LOG, self.triangle_normals, "triangle_normals", owner=self.__class__.__name__)


    def compute_vertex_normals(self):
        """
        Estimates vertex normals, based on triangle normals weighted by the 
        angle they subtend at each vertex...
        """
        vert_norms = numpy.zeros((self.number_of_vertices, 3))
        bad_normal_count = 0
        for k in xrange(self.number_of_vertices):
            try:
                tri_list = list(self.vertex_triangles[k])
                angle_mask = self.triangles[tri_list, :] == k
                angles = self.triangle_angles[tri_list, :]
                angles = angles[angle_mask][:, numpy.newaxis]
                angle_scaling = angles / numpy.sum(angles, axis=0)
                vert_norms[k, :] = numpy.mean(angle_scaling * self.triangle_normals[tri_list, :], axis=0)
                #Scale by angle subtended.
                vert_norms[k, :] = vert_norms[k, :] / numpy.sqrt(numpy.sum(vert_norms[k, :] ** 2, axis=0))
                #Normalise to unit vectors.
            except (ValueError, FloatingPointError):
                # If normals are bad, default to position vector
                # A nicer solution would be to detect degenerate triangles and ignore their
                # contribution to the vertex normal
                vert_norms[k, :] = self.vertices[k] / numpy.sqrt(self.vertices[k].dot(self.vertices[k]))
                bad_normal_count += 1
        if bad_normal_count:
            self.logger.warn(" %d vertices have bad normals" % bad_normal_count)
        util.log_debug_array(LOG, vert_norms, "vertex_normals", owner=self.__class__.__name__)
        self.vertex_normals = vert_norms


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
        util.log_debug_array(LOG, triangle_areas, "triangle_areas", owner=self.__class__.__name__)

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
        util.log_debug_array(LOG, tri_centres, "tri_centres")
        return tri_centres


    @property
    def triangle_angles(self):
        """
        An array containing the inner angles for each triangle, same shape as triangles.
        """
        if self._triangle_angles is None:
            self._triangle_angles = self._find_triangle_angles()
        return self._triangle_angles


    def _find_triangle_angles(self):
        """
        Calculates the inner angles of all the triangles which make up a surface
        """
        verts = self.vertices
        # TODO: Should be possible with arrays, ie not nested loops...
        # A short profile indicates this function takes 95% of the time to compute normals
        # (this was a direct translation of some old matlab code)
        angles = numpy.zeros((self.number_of_triangles, 3))
        for tt in xrange(self.number_of_triangles):
            triangle = self.triangles[tt, :]
            for ta in xrange(3):
                ang = numpy.roll(triangle, -ta)
                angles[tt, ta] = numpy.arccos(numpy.dot(
                    (verts[ang[1], :] - verts[ang[0], :]) /
                    numpy.sqrt(numpy.sum((verts[ang[1], :] - verts[ang[0], :]) ** 2, axis=0)),
                    (verts[ang[2], :] - verts[ang[0], :]) /
                    numpy.sqrt(numpy.sum((verts[ang[2], :] - verts[ang[0], :]) ** 2, axis=0))))

        util.log_debug_array(LOG, angles, "triangle_angles", owner=self.__class__.__name__)
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
        edges = set(tuple(edges[k]) for k in xrange(edges.shape[0]))
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
        #TODO: Would a Sparse matrix be a more useful data structure for these??? 
        elem = numpy.sqrt(((self.vertices[self.edges, :][:, 0, :] -
                            self.vertices[self.edges, :][:, 1, :]) ** 2).sum(axis=1))
        return elem


    @property
    def edge_length_mean(self):
        """The mean length of the edges of the mesh."""
        if self._edge_length_mean is None:
            self._edge_length_mean = self.edge_lengths.mean()
        return self._edge_length_mean


    @property
    def edge_length_min(self):
        """The length of the shortest edge in the mesh."""
        if self._edge_length_min is None:
            self._edge_length_min = self.edge_lengths.min()
        return self._edge_length_min


    @property
    def edge_length_max(self):
        """The length of the longest edge in the mesh."""
        if self._edge_length_max is None:
            self._edge_length_max = self.edge_lengths.max()
        return self._edge_length_max


    @property
    def edge_triangles(self):
        """
        List of the pairs of triangles sharing an edge.
        """
        if self._edge_triangles is None:
            self._edge_triangles = self._find_edge_triangles()
        return self._edge_triangles


    def _find_edge_triangles(self):
        """
        .
        """
        triangles = [[] for _ in xrange(self.number_of_edges)]
        for k in xrange(self.number_of_edges):
            tris = set.intersection(set(self.vertex_triangles[self.edges[k][0]]),
                                    set(self.vertex_triangles[self.edges[k][1]]))
            triangles[k] = tris

        triangles = map(frozenset, triangles)

        return triangles


    def check(self):
        """
        Check the surface under the assumption of topologically spherical closed
        triangular mesh. Returns a 5 element tuple: 1) A boolean, True if no 
        checks failed False otherwise; 2) The Euler characteristic number for
        the mesh surface, this should be 2 or 4 -- meaning one or two closed
        topologically spherical surfaces; 3) a list of indices for any isolated
        vertices; 4) a list of indices of edges where the surface is pinched; 
        and 5) a list of indices of edges that border holes in the surface.

        The allowance for one or two closed surfaces is because the skull/etc
        should be represented by a single closed surface and we typically 
        represent the cortex as one closed surface per hemisphere.

        """
        is_good = True
        isolated = []
        pinched_off = []
        holes = []
        error_message = ""

        #The Euler characteristic for a 2D sphere embedded in a 3D space is 2.
        euler = self.number_of_vertices + self.number_of_triangles - self.number_of_edges
        if euler not in (2, 4):
            error_message = "The surface is expected to be 1 or 2 closed spheres."
            LOG.error(error_message)
            is_good = False

        if self.triangles.max() >= self.number_of_vertices:
            error_message = "There are triangles that index nonexistent vertices."
            LOG.error(error_message)
            is_good = False

        triangles_per_vertex = numpy.array(map(len, self.vertex_triangles))
        if numpy.any(triangles_per_vertex < 3):
            error_message = "The surface contains isolated vertices."
            LOG.error(error_message)
            is_good = False
            isolated = numpy.nonzero(triangles_per_vertex < 3)

        triangles_per_edge = numpy.array(map(len, self.edge_triangles))
        if numpy.any(triangles_per_edge > 2):
            error_message = "There are edges with more than 2 triangles, part of the surface is pinched off."
            LOG.error(error_message)
            is_good = False
            pinched_off = numpy.nonzero(triangles_per_edge > 2)

        if numpy.any(triangles_per_edge < 2):
            error_message = "Free boundaries, there are holes in the surface."
            LOG.error(error_message)
            is_good = False
            holes = numpy.nonzero(triangles_per_edge < 2)

        return is_good, euler, isolated, pinched_off, holes, error_message


    def compute_equation(self, focal_points, equation):
        """
        focal_points - a list of focal points. Used for specifying the vertices
        from which the distance is calculated.
        equation - the equation which should be evaluated
        """
        focal_points = numpy.array(focal_points, dtype=numpy.int32)
        dist = self.geodesic_distance(focal_points)
        equation.pattern = dist
        return equation.pattern



class CorticalSurfaceScientific(surfaces_data.CorticalSurfaceData, SurfaceScientific):
    """ This class exists to add scientific methods to CorticalSurface """
    pass



class SkinAirScientific(surfaces_data.SkinAirData, SurfaceScientific):
    """ This class exists to add scientific methods to SkinAir """

    __tablename__ = None



class BrainSkullScientific(surfaces_data.BrainSkullData, SurfaceScientific):
    """ This class exists to add scientific methods to BrainSkull """
    pass



class SkullSkinScientific(surfaces_data.SkullSkinData, SurfaceScientific):
    """ This class exists to add scientific methods to SkullSkin """
    pass

##--------------------- CLOSE SURFACES End Here---------------------------------------##

##--------------------- OPEN SURFACES Start Here---------------------------------------##


class OpenSurfaceScientific(surfaces_data.OpenSurfaceData, SurfaceScientific):
    """ This class exists to add scientific methods to OpenSurface """
    pass



class EEGCapScientific(surfaces_data.EEGCapData, OpenSurfaceScientific):
    """ This class exists to add scientific methods to EEGCap """
    pass



class FaceSurfaceScientific(surfaces_data.FaceSurfaceData, OpenSurfaceScientific):
    """ This class exists to add scientific methods to FaceSurface """
    pass

##--------------------- OPEN SURFACES End Here---------------------------------------##

##--------------------- SURFACES ADJIACENT classes start Here---------------------------------------##


class RegionMappingScientific(surfaces_data.RegionMappingData):
    """ 
    Scientific methods regarding RegionMapping DataType.
    """
    __tablename__ = None



class LocalConnectivityScientific(surfaces_data.LocalConnectivityData):
    """ This class exists to add scientific methods to LocalConnectivity """
    __tablename__ = None


    def compute_sparse_matrix(self):
        """
        NOTE: Before calling this method, the surface field
        should already be set on the local connectivity.

        Computes the sparse matrix for this local connectivity.
        """
        if self.surface is None:
            msg = " Before calling 'compute_sparse_matrix' method, the surface field should already be set."
            LOG.error(msg)
            raise Exception(msg)

        self.matrix = gdist.local_gdist_matrix(self.surface.vertices.astype(numpy.float64),
                                               self.surface.triangles.astype(numpy.int32), max_distance=self.cutoff)



class CortexScientific(surfaces_data.CortexData, SurfaceScientific):
    """ This class exists to add scientific methods to Cortex """
    __tablename__ = None

    #TODO: Prob. should implement these in the @property way...
    region_areas = None
    region_orientation = None


    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialisation.
        """
        super(CortexScientific, self).configure()
        self.region_average = self.region_mapping

        if self.region_orientation is None:
            self.compute_region_orientation()

        if self.region_areas is None:
            self.compute_region_areas()

        if self.local_connectivity is None:
            # TODO: Switch to degree weighted nearest neighbour, or store a default,
            # computing this every time I want a configured Cortex is a pain...
            self.local_connectivity = surfaces_data.LocalConnectivityData(cutoff=40.0, use_storage=False)

        if self.local_connectivity.cutoff == 0:
            self.local_connectivity.cutoff = 40.0
            #TODO: Temporary hack

        if self.local_connectivity.matrix.size == 0:
            self.compute_local_connectivity()


    def _find_summary_info(self):
        """
        Extend the base class's scientific summary information dictionary.
        """
        summary = super(CortexScientific, self)._find_summary_info()
        summary["Number of regions"] = numpy.sum(self.region_areas > 0.0)
        summary["Region area, mean (mm:math:`^2`)"] = self.region_areas.mean()
        summary["Region area, minimum (mm:math:`^2`)"] = self.region_areas.min()
        summary["Region area, maximum (mm:math:`^2`)"] = self.region_areas.max()
        return summary


    def compute_local_connectivity(self):
        """
        """
        LOG.info("Computing local connectivity matrix")
        #TODO: Better to put this in configure, then callers job, if change cutoff, to explicit call compute...matrix 
        loc_con_cutoff = self.local_connectivity.cutoff
        self.compute_geodesic_distance_matrix(max_dist=loc_con_cutoff)

        self.local_connectivity.matrix = self.geodesic_distance_matrix.copy()
        self.local_connectivity.compute()  # Evaluate equation based distance
        self.local_connectivity.trait["matrix"].log_debug(owner=self.__class__.__name__ + ".local_connectivity")

        #HACK FOR DEBUGGING CAUSE TRAITS REPORTS self.local_connectivity.trait["matrix"] AS BEING EMPTY...
        lcmat = self.local_connectivity.matrix
        sts = str(lcmat.__class__)
        name = ".".join((self.__class__.__name__ + ".local_connectivity", self.local_connectivity.trait.name))
        shape = str(lcmat.shape)
        sparse_format = str(lcmat.format)
        nnz = str(lcmat.nnz)
        dtype = str(lcmat.dtype)
        if lcmat.data.any() and lcmat.data.size > 0:
            array_max = lcmat.data.max()
            array_min = lcmat.data.min()
        else:
            array_max = array_min = 0.0
        LOG.debug("%s: %s shape: %s" % (sts, name, shape))
        LOG.debug("%s: %s format: %s" % (sts, name, sparse_format))
        LOG.debug("%s: %s number of non-zeros: %s" % (sts, name, nnz))
        LOG.debug("%s: %s dtype: %s" % (sts, name, dtype))
        LOG.debug("%s: %s maximum: %s" % (sts, name, array_max))
        LOG.debug("%s: %s minimum: %s" % (sts, name, array_min))

        # Pad the local connectivity matrix with zeros when non-cortical regions
        # are included in the long range connectivity...
        if self.local_connectivity.matrix.shape[0] < self.region_mapping.shape[0]:
            LOG.info("There are non-cortical regions, will pad local connectivity")
            padding = sparse.csc_matrix((self.local_connectivity.matrix.shape[0],
                                         self.region_mapping.shape[0] - self.local_connectivity.matrix.shape[0]))
            self.local_connectivity.matrix = sparse.hstack([self.local_connectivity.matrix, padding])

            padding = sparse.csc_matrix((self.region_mapping.shape[0] - self.local_connectivity.matrix.shape[0],
                                         self.local_connectivity.matrix.shape[1]))
            self.local_connectivity.matrix = sparse.vstack([self.local_connectivity.matrix, padding])


    #----------------------------- region_average -----------------------------#
    def _get_region_average(self):
        """
        An array that ...
        """
        return self._region_average


    def _set_region_average(self, spatial_mask):
        """ 
        .d..
        """

        self.region_sum = spatial_mask

        nodes_per_area = numpy.sum(self.region_sum, axis=1)[:, numpy.newaxis]
        self._region_average = self.region_sum / nodes_per_area
        #import pdb; pdb.set_trace()

        util.log_debug_array(LOG, self._region_average, "region_average", owner=self.__class__.__name__)


    region_average = property(fget=_get_region_average, fset=_set_region_average)
    #--------------------------------------------------------------------------#


    #------------------------------- region_sum -------------------------------#
    def _get_region_sum(self):
        """
        An array that ...
        """
        return self._region_sum


    def _set_region_sum(self, spatial_mask):
        """ 
         Set self._region_average attribute based on region mapping...
        """

        self.vertex_mapping = spatial_mask

        self._region_sum = self.vertex_mapping.T

        util.log_debug_array(LOG, self._region_sum, "region_sum", owner=self.__class__.__name__)


    region_sum = property(fget=_get_region_sum, fset=_set_region_sum)
    #--------------------------------------------------------------------------#


    #----------------------------- vertex_mapping -----------------------------#
    def _get_vertex_mapping(self):
        """
        A two dimensional array that can be used, via matrix multiplication, to
        map a vector of length number_of_regions to a vector of length 
        number_of_vertices.
        """
        return self._vertex_mapping


    def _set_vertex_mapping(self, spatial_mask):
        """ 
        Set self._region_average attribute based on region mapping...
        """
        number_of_nodes = self.region_mapping.shape[0]     # TODO: need to support non-cortical regions here
        number_of_areas = len(numpy.unique(spatial_mask))  # TODO: need to support non-cortical regions here
        #import pdb; pdb.set_trace()
        vertex_mapping = numpy.zeros((number_of_nodes, number_of_areas))
        vertex_mapping[numpy.arange(number_of_nodes), spatial_mask] = 1

        self._vertex_mapping = vertex_mapping

        util.log_debug_array(LOG, self._vertex_mapping, "vertex_mapping", owner=self.__class__.__name__)


    vertex_mapping = property(fget=_get_vertex_mapping, fset=_set_vertex_mapping)
    #--------------------------------------------------------------------------#

    #TODO: May be better to have these return values for assignment to the
    #      associated Connectivity...
    #TODO: These will need to do something sensible with non-cortical regions.
    def compute_region_areas(self):
        """
        """
        regions = numpy.unique(self.region_mapping)
        region_surface_area = numpy.zeros((len(regions), 1))
        avt = numpy.array(self.vertex_triangles)
        #NOTE: Slightly overestimates as it counts overlapping border triangles,
        #      but, not really a problem provided triangle-size << region-size.
        for k in regions:
            regs = map(set, avt[self.region_mapping == k])
            region_triangles = set.union(*regs)
            region_surface_area[k] = self.triangle_areas[list(region_triangles)].sum()

        util.log_debug_array(LOG, region_surface_area, "region_areas", owner=self.__class__.__name__)
        self.region_areas = region_surface_area


    def compute_region_orientation(self):
        """
        """
        regions = numpy.unique(self.region_mapping)
        average_orientation = numpy.zeros((len(regions), 3))
        #Average orientation of the region
        for k in regions:
            orient = self.vertex_normals[self.region_mapping == k, :]
            avg_orient = numpy.mean(orient, axis=0)
            average_orientation[k, :] = avg_orient / numpy.sqrt(numpy.sum(avg_orient ** 2))

        util.log_debug_array(LOG, average_orientation, "region_orientation", owner=self.__class__.__name__)
        self.region_orientation = average_orientation


