from tvb.adapters.datatypes.db.graph import CovarianceIndex
from tvb.adapters.datatypes.db.mode_decompositions import PrincipalComponentsIndex, IndependentComponentsIndex
from tvb.adapters.datatypes.db.spectral import FourierSpectrumIndex, WaveletCoefficientsIndex, CoherenceSpectrumIndex
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
            associated_cv = dao.get_generic_entity(CovarianceIndex, self.handled_datatype.gid, 'fk_source_gid')
            associated_pca = dao.get_generic_entity(PrincipalComponentsIndex, self.handled_datatype.gid, 'fk_source_gid')
            associated_is = dao.get_generic_entity(IndependentComponentsIndex, self.handled_datatype.gid, 'fk_source_gid')
            associated_cc = dao.get_generic_entity(CrossCorrelationIndex, self.handled_datatype.gid, 'fk_source_gid')
            associated_fr = dao.get_generic_entity(FourierSpectrumIndex, self.handled_datatype.gid, 'fk_source_gid')
            associated_wv = dao.get_generic_entity(WaveletCoefficientsIndex, self.handled_datatype.gid, 'fk_source_gid')
            associated_cs = dao.get_generic_entity(CoherenceSpectrumIndex, self.handled_datatype.gid, 'fk_source_gid')

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

        # # reconsider this. Possibly remove measures?
        # associated_dm = dao.get_generic_entity(DatatypeMeasure, self.handled_datatype.gid, '_analyzed_datatype')
        # for datatype_measure in associated_dm:
        #     datatype_measure._analyed_datatype = None
        #     dao.store_entity(datatype_measure)

        ABCRemover.remove_datatype(self, skip_validation)