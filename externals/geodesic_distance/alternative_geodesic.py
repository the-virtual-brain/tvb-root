
"""
Translation of the C++ geodesic library to Python 

mw 18/10/2013

"""

import time
import numpy

# geodesic_constants_and_simple_functions.h
GEODESIC_INF = 1e100
SMALLEST_INTERVAL_RATIO = 1e-6

def cos_from_edges(a, b, c):
    assert all(a > 1e-50) and all(b > 1e-50) and all(c > 1e-50)
    return numpy.clip((b*b + c*c - a*a) / (2.0 * b * c), -1.0, 1.0)
    
def angle_from_edges(a, b, c):
    return numpy.arccos(cos_from_edges(a, b, c))

def read_mesh_from(filename):
    raise NotImplemented

# geodesic_mesh_elements.h
class MeshElement(object):
    def __init__(self):
        self.adjacent_vertices = []
        self.adjacent_faces = []
        self.adjacent_edges = []

class Point3D(object):
    x, y, z = 0., 0., 0.
    def distance(self, v):
        x, y, z = v
        dx, dy, dz =  self.x - x, self.y - y, self.z - z
        return numpy.sqrt(dx*dx + dy*dy + dz*dz)
    def set(self, x, y, z):
        self.x, self.y, self.z = 0., 0., 0.
    def iadd(self, v):
        self.x += v[0]
        self.y += v[1]
        self.z += v[2]
    def imul(self, v):
        self.x *= v
        self.y *= v
        self.z *= v

class Vertex(MeshElement, Point3D):
    saddle_or_boundary = None

class Face(MeshElement):
    corner_angles = [None]*3
    def opposite_edge(vertex):
        raise NotImplemented
    def opposite_vertex(edge):
        raise NotImplemented
    def next_edge(edge, vertex):
        raise NotImplemented
    def vertex_angle(vertex):
        for v, a in zip(self.adjacent_vertices, self.corner_angles):
            if v == vertex:
                return a

class Edge(MeshElement):
    length = 0.0
    def opposite_face(self, face):
        raise NotImplemented
    def opposite_vertex(self, vertex):
        raise NotImplemented
    def belongs(self, vertex):
        raise NotImplemented
    def is_boundary(self):
        return 1 == len(self.adjacent_faces)
    def local_coordinates(point, x, y):
        raise NotImplemented

class SurfacePoint(Point3D):
    """
    A point lying anywhere on the surface of the mesh

    """

    def __init__(self, p3, a=0.5):
        if isinstance(p3, Vertex):
            self.p = p3
        elif isinstance(p3, Face):
            self.set(0., 0., 0.)
            [self.iadd(vi) for vi in p3.adjacent_vertices]
            self.imul(1./3)
        elif isinstance(p3, Edge):
            b = 1 - a
            v0 = p3.adjacent_vertices[0]
            v1 = p3.adjacent_vertices[1]
            self.x = b*v0.x + a*v1.x
            self.y = b*v0.y + a*v1.y
            self.z = b*v0.z + a*v1.z        

class HalfEdge(object):
    # ints in C++
    face_id, vertex_0, vertex_1 = None, None, None

    def __lt__(x, y):
        if (x.vertex_0 == y.vertex_0):
            return x.vertex_1 < y.vertex_1
        else:
            return x.vertex_0 < y.vertex_0

    def __ne__(x, y):
      return x.vertex_0 != y.vertex_0 or x.vertex_1 != y.vertex_1;

    def __eq__(x, y):
      return x.vertex_0 == y.vertex_0 and x.vertex_1 == y.vertex_1;


class SurfacePath(object):
    # std::vector<SurfacePoint>& path
    path = [] 
    def length(self):
        raise NotImplemented
    def print_info_about_path(self):
        raise NotImplemented



# geodesic_algorithm_base.h







class Base(object):
    """
    Base algorithm, from geodesic_algorithm_base.h

    """

    def __init__(self):
        self.max_propagation_distance = 1e100
        self.mesh = None

    def propagate(self, sources, max_propagation_distance, stop_points):
        raise NotImplemented

    def trace_back(self, destination, path):
        raise NotImplemented

    def geodesic(self, source, destination, path):
        raise NotImplemented

    def best_source(self, point, distance):
        raise NotImplemented

    def print_statistics(self):
        raise NotImplemented

    def set_stop_conditions(self, stop_points, stop_distance):
        raise NotImplemented

    def stop_distance(self):
        return max_propagation_distance


class Dijkstra(Base):
    pass

class Subdivision(Base):
    pass

class Exact(Base):
    pass



