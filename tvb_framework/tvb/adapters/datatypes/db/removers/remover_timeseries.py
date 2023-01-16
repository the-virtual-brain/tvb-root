# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
from tvb.adapters.datatypes.db.fcd import FcdIndex
from tvb.adapters.datatypes.db.graph import CovarianceIndex, CorrelationCoefficientsIndex
from tvb.adapters.datatypes.db.mapped_value import DatatypeMeasureIndex
from tvb.adapters.datatypes.db.mode_decompositions import PrincipalComponentsIndex, IndependentComponentsIndex
from tvb.adapters.datatypes.db.spectral import FourierSpectrumIndex, WaveletCoefficientsIndex, CoherenceSpectrumIndex, \
    ComplexCoherenceSpectrumIndex
from tvb.adapters.datatypes.db.temporal_correlations import CrossCorrelationIndex
from tvb.core.adapters.abcremover import ABCRemover
from tvb.core.entities.storage import dao
from tvb.core.services.exceptions import RemoveDataTypeException


class TimeseriesRemover(ABCRemover):

    def remove_datatype(self, skip_validation=False):
        """
        Called when a TimeSeries is removed.
        """
        if not skip_validation:
            key = 'fk_source_gid'

            associated_fcd = dao.get_generic_entity(FcdIndex, self.handled_datatype.gid, key)
            associated_cv = dao.get_generic_entity(CovarianceIndex, self.handled_datatype.gid, key)
            associated_pca = dao.get_generic_entity(PrincipalComponentsIndex, self.handled_datatype.gid, key)
            associated_is = dao.get_generic_entity(IndependentComponentsIndex, self.handled_datatype.gid, key)
            associated_cc = dao.get_generic_entity(CrossCorrelationIndex, self.handled_datatype.gid, key)
            associated_fr = dao.get_generic_entity(FourierSpectrumIndex, self.handled_datatype.gid, key)
            associated_wv = dao.get_generic_entity(WaveletCoefficientsIndex, self.handled_datatype.gid, key)
            associated_cs = dao.get_generic_entity(CoherenceSpectrumIndex, self.handled_datatype.gid, key)
            associated_coef = dao.get_generic_entity(CorrelationCoefficientsIndex, self.handled_datatype.gid, key)
            associated_dtm = dao.get_generic_entity(DatatypeMeasureIndex, self.handled_datatype.gid, key)
            associated_ccs = dao.get_generic_entity(ComplexCoherenceSpectrumIndex, self.handled_datatype.gid, key)

            msg = "TimeSeries cannot be removed as it is used by at least one "

            if len(associated_fcd) > 0:
                raise RemoveDataTypeException(msg + " FCD.")
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
            if len(associated_coef) > 0:
                raise RemoveDataTypeException(msg + " CorrelationCoefficient.")
            if len(associated_dtm) > 0:
                raise RemoveDataTypeException(msg + " DatatypeMeasure.")
            if len(associated_ccs) > 0:
                raise RemoveDataTypeException(msg + " ComplexCoherenceSpectrum.")

        ABCRemover.remove_datatype(self, skip_validation)
