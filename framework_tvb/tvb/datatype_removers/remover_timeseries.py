# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.entities.storage import dao
from tvb.datatypes.graph import Covariance
from tvb.datatypes.mode_decompositions import PrincipalComponents, IndependentComponents
from tvb.datatypes.temporal_correlations import CrossCorrelation
from tvb.datatypes.spectral import FourierSpectrum, WaveletCoefficients, CoherenceSpectrum
from tvb.datatypes.mapped_values import DatatypeMeasure
from tvb.core.services.exceptions import RemoveDataTypeException


class TimeseriesRemover(ABCRemover):

    def remove_datatype(self, skip_validation=False):
        """
        Called when a TimeSeries is removed.
        """
        if not skip_validation:
            associated_cv = dao.get_generic_entity(Covariance, self.handled_datatype.gid, '_source')
            associated_pca = dao.get_generic_entity(PrincipalComponents, self.handled_datatype.gid, '_source')
            associated_is = dao.get_generic_entity(IndependentComponents, self.handled_datatype.gid, '_source')
            associated_cc = dao.get_generic_entity(CrossCorrelation, self.handled_datatype.gid, '_source')
            associated_fr = dao.get_generic_entity(FourierSpectrum, self.handled_datatype.gid, '_source')
            associated_wv = dao.get_generic_entity(WaveletCoefficients, self.handled_datatype.gid, '_source')
            associated_cs = dao.get_generic_entity(CoherenceSpectrum, self.handled_datatype.gid, '_source')

            msg = "TimeSeries cannot be removed as it is used by at least one "

            if len(associated_cv) > 0:
                raise RemoveDataTypeException(msg + " Covariance.")
            if len(associated_pca) > 0:
                raise RemoveDataTypeException(msg + " PrincipalComponents.")
            if len(associated_is) > 0:
                raise RemoveDataTypeException(msg + " IndependentComponents.")
            if len(associated_cc) > 0:
                raise RemoveDataTypeException(msg + " CrossCorrelation.")
            if len(associated_fr) > 0:
                raise RemoveDataTypeException(msg + " FourierSpectrum.")
            if len(associated_wv) > 0:
                raise RemoveDataTypeException(msg + " WaveletCoefficients.")
            if len(associated_cs) > 0:
                raise RemoveDataTypeException(msg + " CoherenceSpectrum.")

        # todo: reconsider this. Possibly remove measures
        associated_dm = dao.get_generic_entity(DatatypeMeasure, self.handled_datatype.gid, '_analyzed_datatype')
        for datatype_measure in associated_dm:
            datatype_measure._analyed_datatype = None
            dao.store_entity(datatype_measure)

        ABCRemover.remove_datatype(self, skip_validation)
