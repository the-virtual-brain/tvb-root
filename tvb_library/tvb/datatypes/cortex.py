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
# CITATION:
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
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Range
from tvb.datatypes import local_connectivity, region_mapping, surfaces


class Cortex(HasTraits):
    """
    Wrapper Class to gather necessary entities for a surface-based simulation.
    To be used when preparing a simulation launch.
    """

    local_connectivity = Attr(
        field_type=local_connectivity.LocalConnectivity,
        label="Local Connectivity",
        required=False,
        doc="Define the interaction between neighboring network nodes. This is implicitly integrated in"
            " the definition of a given surface as an excitatory mean coupling of directly adjacent neighbors to"
            " the first state variable of each population model (since these typically represent the mean-neural"
            " membrane voltage). This coupling is instantaneous (no time delays).")

    region_mapping_data = Attr(
        field_type=region_mapping.RegionMapping,
        label="Region mapping",
        doc="""An index vector of length equal to the number_of_vertices + the
            number of non-cortical regions, with values that index into an
            associated connectivity matrix.""")  # 'CS'

    coupling_strength = NArray(
        label="Local coupling strength",
        domain=Range(lo=0.0, hi=20.0, step=1.0),
        default=numpy.array([1.0]),
        # file_storage=core.FILE_STORAGE_NONE,
        doc="""A factor that rescales local connectivity strengths.""")

    _regmap = None

    @property
    def region_mapping(self):
        """Generate a full region mapping vector."""
        if self._regmap is not None:
            return self._regmap
        rm = self.region_mapping_data.array_data
        unmapped = self.region_mapping_data.connectivity.unmapped_indices(rm)
        self._regmap = numpy.r_[rm, unmapped]
        return self._regmap

    @property
    def surface(self):
        """
        Define shortcut for retrieving the surface held by a RegionMapping.
        """
        return self.region_mapping_data.surface

    @property
    def number_of_vertices(self):
        """
        Define shortcut for retrieving the number of vertices of the surface held by a RegionMapping.
        """
        return self.region_mapping_data.surface.number_of_vertices

    @property
    def number_of_triangles(self):
        """
        Define shortcut for retrieving the number of triangles of the surface held by a RegionMapping.
        """
        return self.region_mapping_data.surface.number_of_triangles

    @property
    def triangles(self):
        """
        Define shortcut for retrieving the triangles of the surface held by a RegionMapping.
        """
        return self.region_mapping_data.surface.triangles

    @property
    def vertices(self):
        """
        Define shortcut for retrieving the vertices of the surface held by a RegionMapping.
        """
        return self.region_mapping_data.surface.vertices

    @property
    def vertex_normals(self):
        """
        Define shortcut for retrieving the vertex_normals of the surface held by a RegionMapping.
        """
        return self.region_mapping_data.surface.vertex_normals

    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialisation.
        """
        if self.local_connectivity is None:
            self.local_connectivity = local_connectivity.LocalConnectivity(cutoff=40.0,
                                                                           surface=self.region_mapping_data.surface)

        # mhtodo: review nullability of NArrays
        if self.local_connectivity.matrix is None or self.local_connectivity.matrix.size == 0:
            self.compute_local_connectivity()

        # Pad the local connectivity matrix with zeros when non-cortical regions
        # are included in the long range connectivity...
        if self.local_connectivity.matrix.shape[0] < self.region_mapping.shape[0]:
            self.log.info("There are non-cortical regions, will pad local connectivity")
            padding = scipy.sparse.csc_matrix((self.local_connectivity.matrix.shape[0],
                                               self.region_mapping.shape[0] - self.local_connectivity.matrix.shape[0]))
            self.local_connectivity.matrix = scipy.sparse.hstack([self.local_connectivity.matrix, padding])

            padding = scipy.sparse.csc_matrix((self.region_mapping.shape[0] - self.local_connectivity.matrix.shape[0],
                                               self.local_connectivity.matrix.shape[1]))
            self.local_connectivity.matrix = scipy.sparse.vstack([self.local_connectivity.matrix, padding])

    def compute_local_connectivity(self):
        """
        """
        self.log.info("Computing local connectivity matrix")
        loc_con_cutoff = self.local_connectivity.cutoff
        self.local_connectivity.surface.compute_geodesic_distance_matrix(max_dist=loc_con_cutoff)

        self.local_connectivity.matrix_gdist = self.local_connectivity.surface.geodesic_distance_matrix.copy()
        self.local_connectivity.compute()  # Evaluate equation based distance

        # HACK FOR DEBUGGING CAUSE TRAITS REPORTS self.local_connectivity.trait["matrix"] AS BEING EMPTY...
        lcmat = self.local_connectivity.matrix
        sts = str(lcmat.__class__)
        # mhtodo: the trait.name is the file name for mapped,
        name = self.__class__.__name__ + ".local_connectivity"  # self.local_connectivity.trait.name))
        shape = str(lcmat.shape)
        sparse_format = str(lcmat.format)
        nnz = str(lcmat.nnz)
        dtype = str(lcmat.dtype)
        if lcmat.data.any() and lcmat.data.size > 0:
            array_max = lcmat.data.max()
            array_min = lcmat.data.min()
        else:
            array_max = array_min = 0.0
        self.log.debug("%s: %s shape: %s" % (sts, name, shape))
        self.log.debug("%s: %s format: %s" % (sts, name, sparse_format))
        self.log.debug("%s: %s number of non-zeros: %s" % (sts, name, nnz))
        self.log.debug("%s: %s dtype: %s" % (sts, name, dtype))
        self.log.debug("%s: %s maximum: %s" % (sts, name, array_max))
        self.log.debug("%s: %s minimum: %s" % (sts, name, array_min))

    def prepare_local_coupling(self, number_of_nodes):
        """Prepare the concrete local coupling matrix used for simulation."""
        if self.coupling_strength.size == 1:
            local_coupling = (self.coupling_strength[0] *
                              self.local_connectivity.matrix)
        elif self.coupling_strength.size == self.number_of_vertices:
            ind = numpy.arange(number_of_nodes, dtype=numpy.intc)
            vec_cs = numpy.zeros((number_of_nodes,))
            vec_cs[:self.number_of_vertices] = self.coupling_strength
            sp_cs = scipy.sparse.csc_matrix((vec_cs, (ind, ind)),
                                            shape=(number_of_nodes, number_of_nodes))
            local_coupling = sp_cs * self.local_connectivity.matrix
        else:
            raise RuntimeError("cortex.coupling_strength must be size 1 or number_of_vertices")
        return local_coupling

    @classmethod
    def from_file(cls, source_file='cortex_16384.zip', region_mapping_file="regionMapping_16k_76.txt",
                  local_connectivity_file=None):

        result = Cortex()

        if region_mapping_file is not None:
            result.region_mapping_data = region_mapping.RegionMapping.from_file(region_mapping_file)

        if source_file is not None:
            result.region_mapping_data.surface = surfaces.CorticalSurface.from_file(source_file)

        if local_connectivity_file is not None:
            result.local_connectivity = local_connectivity.LocalConnectivity.from_file(local_connectivity_file)

        return result
