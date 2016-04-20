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
from scipy import sparse as sparse
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits import types_basic as basic, exceptions
from tvb.basic.traits.types_mapped import MappedType, SparseMatrix
from tvb.datatypes import equations as equations
from tvb.datatypes.surfaces import CorticalSurface

LOG = get_logger(__name__)

class LocalConnectivityData(MappedType):
    """
    A sparse matrix for representing the local connectivity within the Cortex.
    """
    _ui_name = "Local connectivity"

    surface = CorticalSurface(label="Surface", order=1)

    matrix = SparseMatrix(order=-1)

    equation = equations.FiniteSupportEquation(
        label="Spatial",
        required=False,
        default=equations.Gaussian,
        order=2)

    cutoff = basic.Float(
        label="Cutoff distance (mm)",
        default=40.0,
        doc="Distance at which to truncate the evaluation in mm.",
        order=3)


    def compute(self):
        """
        Compute current Matrix.
        """
        LOG.info("Mapping geodesic distance through the LocalConnectivity.")

        #Start with data being geodesic_distance_matrix, then map it through equation
        self.equation.pattern = self.matrix_gdist.data

        #Then replace original data with result...
        self.matrix_gdist.data = self.equation.pattern

        #Homogenise spatial discretisation effects across the surface
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
            LOG.warning(msg)
            bad_verts = ()
            if pos_mean != 0.0:
                bad_verts = bad_verts + numpy.nonzero(pos_contrib == 0.0)
            if neg_mean != 0.0:
                bad_verts = bad_verts + numpy.nonzero(neg_contrib == 0.0)
            LOG.debug("Problem vertices are: %s" % str(bad_verts))
        pos_hf = numpy.zeros(shape=pos_contrib.shape)
        pos_hf[pos_contrib != 0] = pos_mean / pos_contrib[pos_contrib != 0]
        neg_hf = numpy.zeros(shape=neg_contrib.shape)
        neg_hf[neg_contrib != 0] = neg_mean / neg_contrib[neg_contrib != 0]
        pos_hf_diag = sparse.csc_matrix((pos_hf, (ind, ind)), shape=(nv, nv))
        neg_hf_diag = sparse.csc_matrix((neg_hf, (ind, ind)), shape=(nv, nv))
        homogenious_conn = (pos_hf_diag * pos_con) + (neg_hf_diag * neg_con)

        #Then replace unhomogenised result with the spatially homogeneous one...
        if not homogenious_conn.has_sorted_indices:
            homogenious_conn.sort_indices()

        self.matrix = homogenious_conn


    def _validate_before_store(self):
        """
        Overrides MappedType._validate_before_store to use a custom error for missing matrix.
        """
        # Sparse Matrix is required so we should check if there is any data stored for it
        if self.matrix is None:
            msg = ("LocalConnectivity can not be stored because it "
                   "has no SparseMatrix attached.")
            raise exceptions.ValidationException(msg)

        super(LocalConnectivityData, self)._validate_before_store()