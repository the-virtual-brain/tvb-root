# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
import numpy
import scipy.sparse
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.projections import ProjectionMatrix
from tvb.datatypes.surfaces import Surface, CorticalSurface
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import Sensors
from tvb.datatypes.volumes import Volume
from tvb.datatypes.local_connectivity import LocalConnectivity


connectivity = Connectivity(
    region_labels=numpy.array(["a", "b"]),
    weights=numpy.zeros((2, 2)),
    # undirected=False,
    tract_lengths=numpy.zeros((2, 2)),
    centres=numpy.zeros((2, 2)),
    cortical=numpy.array([True, True, True, True]),
    hemispheres=numpy.array([True, True, True, True]),
    orientations=numpy.zeros((2, 2)),
    areas=numpy.zeros((4, )),
    number_of_regions=2,
    number_of_connections=4,
    # parent_connectivity=""
    saved_selection=["a", "b", "C"]
)

surface = Surface(
    vertices=numpy.zeros((5, 3)),
    triangles=numpy.zeros((3, 3), dtype=int),
    vertex_normals=numpy.zeros((5, 3)),
    triangle_normals=numpy.zeros((3, 3)),
    number_of_vertices=5,
    number_of_triangles=3,
    edge_mean_length=1.0,
    edge_min_length=0.0,
    edge_max_length=2.0,
    zero_based_triangles=False,
    split_triangles=numpy.arange(0),
    number_of_split_slices=1,
    split_slices=dict(),
    bi_hemispheric=False,
    surface_type="surface",
    valid_for_simulations=True
)

region_mapping = RegionMapping(
    array_data=numpy.arange(5),
    connectivity=connectivity,
    surface=surface
)

cortical_surface = CorticalSurface(
    vertices=numpy.zeros((5, 3)),
    triangles=numpy.zeros((3, 3), dtype=int),
    vertex_normals=numpy.zeros((5, 3)),
    triangle_normals=numpy.zeros((3, 3)),
    number_of_vertices=5,
    number_of_triangles=3,
    edge_mean_length=1.0,
    edge_min_length=0.0,
    edge_max_length=2.0,
    zero_based_triangles=False,
    split_triangles=numpy.arange(0),
    number_of_split_slices=1,
    split_slices=dict(),
    bi_hemispheric=False,
    # surface_type="surface",
    valid_for_simulations=True
)

sensors = Sensors(
    sensors_type="SEEG",
    labels=numpy.array(["s1", "s2", "s3"]),
    locations=numpy.zeros((3, 3)),
    number_of_sensors=3,
    has_orientation=True,
    orientations=numpy.zeros((3, 3)),
    usable=numpy.array([True, False, True])
)

volume = Volume(
    origin=numpy.zeros((3, 3)),
    voxel_size=numpy.zeros((3, 3))
)

projection_matrix = ProjectionMatrix(
    projection_type="projSEEG",
    sources=cortical_surface,
    sensors=sensors,
    projection_data=numpy.zeros((5, 3))
)

local_connectivity = LocalConnectivity(
    surface=cortical_surface,
    matrix=scipy.sparse.csc_matrix(numpy.eye(8) + numpy.eye(8)[:, ::-1]),
    cutoff=12,
)
