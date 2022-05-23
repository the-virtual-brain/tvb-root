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

from tvb.basic.neotraits.api import HasTraits, Attr, Float, narray_summary_info
from tvb.basic.readers import try_get_absolute_path, FileReader
from tvb.datatypes import equations, surfaces


class LocalConnectivity(HasTraits):
    """
    A sparse matrix for representing the local connectivity within the Cortex.
    """
    surface = Attr(field_type=surfaces.CorticalSurface, label="Surface")

    matrix = Attr(field_type=scipy.sparse.spmatrix, required=False)

    equation = Attr(
        field_type=equations.FiniteSupportEquation,
        label="Spatial",
        required=False,
        default=equations.Gaussian())

    cutoff = Float(
        label="Cutoff distance (mm)",
        default=40.0,
        doc="Distance at which to truncate the evaluation in mm.")

    # Temporary obj
    matrix_gdist = None

    def compute(self):
        """
        Compute current Matrix.
        """
        self.log.info("Mapping geodesic distance through the LocalConnectivity.")

        # Start with data being geodesic_distance_matrix, then map it through equation
        # Then replace original data with result...
        self.matrix_gdist.data = self.equation.evaluate(self.matrix_gdist.data)

        # Homogenise spatial discretisation effects across the surface
        nv = self.matrix_gdist.shape[0]
        ind = numpy.arange(nv, dtype=int)
        pos_mask = self.matrix_gdist.data > 0.0
        neg_mask = self.matrix_gdist.data < 0.0
        pos_con = self.matrix_gdist.copy()
        neg_con = self.matrix_gdist.copy()
        pos_con.data[neg_mask] = 0.0
        neg_con.data[pos_mask] = 0.0
        pos_contrib = pos_con.sum(axis=1)
        pos_contrib = numpy.array(pos_contrib).squeeze()
        neg_contrib = neg_con.sum(axis=1)
        neg_contrib = numpy.array(neg_contrib).squeeze()
        pos_mean = pos_contrib.mean()
        neg_mean = neg_contrib.mean()
        if ((pos_mean != 0.0 and any(pos_contrib == 0.0)) or
                (neg_mean != 0.0 and any(neg_contrib == 0.0))):
            msg = "Cortical mesh is too coarse for requested LocalConnectivity."
            self.log.warning(msg)
            bad_verts = ()
            if pos_mean != 0.0:
                bad_verts = bad_verts + numpy.nonzero(pos_contrib == 0.0)
            if neg_mean != 0.0:
                bad_verts = bad_verts + numpy.nonzero(neg_contrib == 0.0)
            self.log.debug("Problem vertices are: %s" % str(bad_verts))
        pos_hf = numpy.zeros(shape=pos_contrib.shape)
        pos_hf[pos_contrib != 0] = pos_mean / pos_contrib[pos_contrib != 0]
        neg_hf = numpy.zeros(shape=neg_contrib.shape)
        neg_hf[neg_contrib != 0] = neg_mean / neg_contrib[neg_contrib != 0]
        pos_hf_diag = scipy.sparse.csc_matrix((pos_hf, (ind, ind)), shape=(nv, nv))
        neg_hf_diag = scipy.sparse.csc_matrix((neg_hf, (ind, ind)), shape=(nv, nv))
        homogenious_conn = (pos_hf_diag * pos_con) + (neg_hf_diag * neg_con)

        # Then replace unhomogenised result with the spatially homogeneous one...
        if not homogenious_conn.has_sorted_indices:
            homogenious_conn.sort_indices()

        self.matrix = homogenious_conn

    @staticmethod
    def from_file(source_file="local_connectivity_16384.mat"):

        result = LocalConnectivity()

        source_full_path = try_get_absolute_path("tvb_data.local_connectivity", source_file)
        reader = FileReader(source_full_path)

        result.matrix = reader.read_array(matlab_data_name="LocalCoupling")
        return result

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        _, _, v = scipy.sparse.find(self.matrix)
        return narray_summary_info(v, ar_name='matrix-nonzero')

    def compute_sparse_matrix(self):
        """
        NOTE: Before calling this method, the surface field
        should already be set on the local connectivity.

        Computes the sparse matrix for this local connectivity.
        """
        if self.surface is None:
            raise AttributeError('Require surface to compute local connectivity.')

        self.matrix_gdist = surfaces.gdist.local_gdist_matrix(
            self.surface.vertices.astype(numpy.float64),
            self.surface.triangles.astype(numpy.int32),
            max_distance=self.cutoff)

        self.compute()
        # Avoid having a large data-set in memory.
        self.matrix_gdist = None
