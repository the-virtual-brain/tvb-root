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

"""
Adapter that uses the traits model to generate interfaces for FCD Analyzer.

.. moduleauthor:: Francesca Melozzi <france.melozzi@gmail.com>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""
import json
import uuid

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.manifold import SpectralEmbedding
from tvb.adapters.datatypes.db.fcd import FcdIndex
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.adapters.datatypes.h5.fcd_h5 import FcdH5
from tvb.basic.neotraits.api import Float
from tvb.basic.neotraits.info import narray_describe
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.neocom import h5
from tvb.core.neotraits.forms import TraitDataTypeSelectField, FloatField
from tvb.core.neotraits.view_model import ViewModel, DataTypeGidAttr
from tvb.datatypes.fcd import Fcd
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.datatypes.time_series import TimeSeriesRegion


class FCDAdapterModel(ViewModel):
    time_series = DataTypeGidAttr(
        linked_datatype=TimeSeriesRegion,
        label="Time Series",
        required=True,
        doc="""The time-series for which the fcd matrices are calculated."""
    )

    sw = Float(
        label="Sliding window length (ms)",
        default=120000,
        doc="""Length of the time window used to divided the time series.
        FCD matrix is calculated in the following way: the time series is divided in time window of fixed length and
        with an overlapping of fixed length. The data-points within each window, centered at time ti, are used to
        calculate FC(ti) as Pearson correlation. The ij element of the FCD matrix is calculated as the Pearson
        Correlation between FC(ti) and FC(tj) arranged in a vector.""")

    sp = Float(
        label="Spanning between two consecutive sliding window (ms)",
        default=2000,
        doc="""Spanning= (time windows length)-(overlapping between two consecutive time window). FCD matrix is
        calculated in the following way: the time series is divided in time window of fixed length and with an
        overlapping of fixed length. The data-points within each window, centered at time ti, are used to calculate
        FC(ti) as Pearson Correlation. The ij element of the FCD matrix is calculated as the Pearson correlation
        between FC(ti) and FC(tj) arranged in a vector""")


class FCDAdapterForm(ABCAdapterForm):
    def __init__(self):
        super(FCDAdapterForm, self).__init__()
        self.time_series = TraitDataTypeSelectField(FCDAdapterModel.time_series, name=self.get_input_name(),
                                                    conditions=self.get_filters(), has_all_option=True)
        self.sw = FloatField(FCDAdapterModel.sw)
        self.sp = FloatField(FCDAdapterModel.sp)

    @staticmethod
    def get_view_model():
        return FCDAdapterModel

    @staticmethod
    def get_required_datatype():
        return TimeSeriesRegionIndex

    @staticmethod
    def get_filters():
        return FilterChain(fields=[FilterChain.datatype + '.data_ndim'], operations=["=="], values=[4])

    @staticmethod
    def get_input_name():
        return "time_series"


class FunctionalConnectivityDynamicsAdapter(ABCAdapter):
    """ TVB adapter for calling the Pearson CrossCorrelation algorithm.

        The present class will do the following actions:

        - Compute the the fcd of the timeseries; the fcd is calculated in the following way:
            the time series is divided in time window of fixed length and with an overlapping of fixed length.
            The data-points within each window, centered at time ti, are used to calculate FC(ti) as Pearson correlation
            The ij element of the FCD matrix is calculated as the Pearson correlation between FC(ti) and FC(tj)
            -in a vector
        - Apply to the fcd the spectral embedding algorithm in order to calculate epochs of stability of the fcd
            (length of time during which FC matrix are high correlated).

        The algorithm can produce 2 kind of results:

        - case 1: the algorithm is able to identify the epochs of stability
            -- fcs calculated over the epochs of stability (excluded the first one = artifact,
            due to initial conditions)
            -- 3 eigenvectors, associated to the 3 largest eigenvalues, of the fcs are extracted
        - case 2: the algorithm is not able to identify the epochs of stability
            -- fc over the all time series is calculated
            -- 3 first eigenvectors, associated to the 3 largest eigenvalues, of the fcs are extracted

        :return
            - fcd matrix whose values are between -1 and 1, inclusive.
            - in case 1: fcd matrix segmented i.e. fcd whose values are between -1 and 1.1, inclusive.
                (Value=1.1 for time not belonging to epochs of stability identified with spectral embedding algorithm)
                in case 2: fcd matrix segmented identical to the fcd matrix not segmented
            - dictionary containing the eigenvectors.
            - dictionary containing the eigenvalues
            - connectivity associated to the TimeSeriesRegions

    """
    _ui_name = "FCD matrix"
    _ui_description = "Functional Connectivity Dynamics metric"
    _ui_subsection = "fcd_calculator"

    def get_form_class(self):
        return FCDAdapterForm

    def get_output(self):
        return [FcdIndex, ConnectivityMeasureIndex]

    def configure(self, view_model):
        # type: (FCDAdapterModel) -> None
        """
        Store the input shape to be later used to estimate memory usage
        """

        self.input_time_series_index = self.load_entity_by_gid(view_model.time_series)
        self.input_shape = (self.input_time_series_index.data_length_1d,
                            self.input_time_series_index.data_length_2d,
                            self.input_time_series_index.data_length_3d,
                            self.input_time_series_index.data_length_4d)
        self.log.debug("time_series shape is %s" % str(self.input_shape))
        self.actual_sp = float(view_model.sp) / self.input_time_series_index.sample_period
        self.actual_sw = float(view_model.sw) / self.input_time_series_index.sample_period
        actual_ts_length = self.input_shape[0]

        if self.actual_sw >= actual_ts_length or self.actual_sp >= actual_ts_length or self.actual_sp >= self.actual_sw:
            raise LaunchException(
                "Spanning (Sp) and Sliding (Sw) window size parameters need to be less than the TS length, "
                "and Sp < Sw. After calibration with sampling period, current values are: Sp=%d, Sw=%d, Ts=%d). "
                "Please configure valid input parameters." % (self.actual_sp, self.actual_sw, actual_ts_length))

    def get_required_memory_size(self, view_model):
        # type: (FCDAdapterModel) -> int
        # We do not know how much memory is needed.
        return -1

    def get_required_disk_size(self, view_model):
        # type: (FCDAdapterModel) -> int
        return 0

    def _populate_fcd_index(self, fcd_index, source_gid, fcd_h5):
        fcd_index.fk_source_gid = source_gid
        fcd_index.labels_ordering = json.dumps(Fcd.labels_ordering.default)
        self.fill_index_from_h5(fcd_index, fcd_h5)

    @staticmethod
    def _populate_fcd_h5(fcd_h5, fcd_data, gid, source_gid, sw, sp):
        fcd_h5.array_data.store(fcd_data)
        fcd_h5.gid.store(uuid.UUID(gid))
        fcd_h5.source.store(uuid.UUID(source_gid))
        fcd_h5.sw.store(sw)
        fcd_h5.sp.store(sp)
        fcd_h5.labels_ordering.store(json.dumps(Fcd.labels_ordering.default))

    def launch(self, view_model):
        # type: (FCDAdapterModel) -> [FcdIndex, ConnectivityMeasureIndex]
        """
        Launch algorithm and build results.
        :param view_model: the ViewModel keeping the algorithm inputs
        :return: the fcd index for the computed fcd matrix on the given time-series, with that sw and that sp
        """
        with h5.h5_file_for_index(self.input_time_series_index) as ts_h5:
            [fcd, fcd_segmented, eigvect_dict, eigval_dict] = self._compute_fcd_matrix(ts_h5)
            connectivity_gid = ts_h5.connectivity.load()
            connectivity = self.load_traited_by_gid(connectivity_gid)

        result = []  # list to store: fcd index, fcd_segmented index (eventually), and connectivity measure indexes

        # Create an index for the computed fcd.
        fcd_index = FcdIndex()
        fcd_h5_path = self.path_for(FcdH5, fcd_index.gid)
        with FcdH5(fcd_h5_path) as fcd_h5:
            self._populate_fcd_h5(fcd_h5, fcd, fcd_index.gid, self.input_time_series_index.gid,
                                  view_model.sw, view_model.sp)
            self._populate_fcd_index(fcd_index, self.input_time_series_index.gid, fcd_h5)
        result.append(fcd_index)

        if np.amax(fcd_segmented) == 1.1:
            result_fcd_segmented_index = FcdIndex()
            result_fcd_segmented_h5_path = self.path_for(FcdH5, result_fcd_segmented_index.gid)
            with FcdH5(result_fcd_segmented_h5_path) as result_fcd_segmented_h5:
                self._populate_fcd_h5(result_fcd_segmented_h5, fcd_segmented,
                                      result_fcd_segmented_index.gid,
                                      self.input_time_series_index.gid, view_model.sw,
                                      view_model.sp)
                self._populate_fcd_index(result_fcd_segmented_index, self.input_time_series_index.gid,
                                         result_fcd_segmented_h5)
            result.append(result_fcd_segmented_index)

        for mode in eigvect_dict.keys():
            for var in eigvect_dict[mode].keys():
                for ep in eigvect_dict[mode][var].keys():
                    for eig in range(3):
                        cm_data = eigvect_dict[mode][var][ep][eig]
                        measure = ConnectivityMeasure()
                        measure.connectivity = connectivity
                        measure.array_data = cm_data
                        measure.title = "Epoch # %d, eigenvalue = %s, variable = %s, " \
                                        "mode = %s." % (ep, eigval_dict[mode][var][ep][eig], var, mode)
                        cm_index = self.store_complete(measure)
                        result.append(cm_index)
        return result

    def _compute_fcd_matrix(self, ts_h5):
        self.log.debug("timeseries_h5.data")
        self.log.debug(narray_describe(ts_h5.data[:]))

        input_shape = ts_h5.data.shape
        result_shape = self._result_shape(input_shape)

        fcd = np.zeros(result_shape)
        fc_stream = {}  # dict where the fc calculated over the sliding window will be stored
        for mode in range(result_shape[3]):
            for var in range(result_shape[2]):
                start = -self.actual_sp  # in order to well initialize the first starting point of the FC stream
                for nfcd in range(result_shape[0]):
                    start += self.actual_sp
                    current_slice = tuple([slice(int(start), int(start + self.actual_sw) + 1), slice(var, var + 1),
                                           slice(input_shape[2]), slice(mode, mode + 1)])
                    data = ts_h5.read_data_slice(current_slice).squeeze()
                    fc = np.corrcoef(data.T)
                    # the triangular part of the fc is organized as a vector, excluding the diagonal (always ones)
                    triangular = np.triu_indices(len(fc), 1)
                    fc_stream[nfcd] = fc[triangular]
                for i in range(result_shape[0]):
                    j = i
                    while j < result_shape[0]:
                        fci = fc_stream[i]
                        fcj = fc_stream[j]
                        fcd[i, j, var, mode] = np.corrcoef(fci, fcj)[0, 1]
                        fcd[j, i, var, mode] = fcd[i, j, var, mode]
                        j += 1

        self.log.debug("FCD")
        self.log.debug(narray_describe(fcd))

        num_eig = 3  # number of the eigenvector that will be extracted

        eigvect_dict = {}  # holds eigenvectors of the fcs calculated over the epochs, key1=mode, key2=var, key3=numb ep
        eigval_dict = {}  # holds eigenvalues of the fcs calculated over the epochs, key1=mode, key2=var, key3=numb ep
        fcd_segmented = None
        for mode in range(result_shape[3]):
            eigvect_dict[mode] = {}
            eigval_dict[mode] = {}
            for var in range(result_shape[2]):
                eigvect_dict[mode][var] = {}
                eigval_dict[mode][var] = {}
                fcd_matrix = fcd[:, :, var, mode]
                [xir, xir_cutoff] = self._spectral_embedding(fcd_matrix)
                epochs_extremes = self._epochs_interval(xir, xir_cutoff, self.actual_sp, self.actual_sw)
                fcd_segmented = fcd.copy()
                if epochs_extremes.shape[0] <= 1:
                    # means that there are no more than 1 epochs of stability, thus the eigenvectors of
                    # the FC calculated over the entire TimeSeries will be calculated
                    epochs_extremes = np.zeros((2, 2), dtype=float)
                    epochs_extremes[1, 1] = input_shape[0]  # [0,0] set in order to skip the first epoch
                else:
                    # means that more than 1 epochs of stability is identified thus fcd_segmented is calculated
                    fcd_segmented[xir > xir_cutoff, :, var, mode] = 1.1
                    fcd_segmented[:, xir > xir_cutoff, var, mode] = 1.1

                for ep in range(1, epochs_extremes.shape[0]):
                    eigvect_dict[mode][var][ep] = []
                    eigval_dict[mode][var][ep] = []
                    current_slice = tuple([slice(int(epochs_extremes[ep][0]), int(epochs_extremes[ep][1]) + 1),
                                           slice(var, var + 1), slice(input_shape[2]), slice(mode, mode + 1)])
                    data = ts_h5.read_data_slice(current_slice).squeeze()
                    fc = np.corrcoef(data.T)  # calculate fc over the epoch of stability
                    eigval_matrix, eigvect_matrix = linalg.eig(fc)
                    eigval_matrix = np.real(eigval_matrix)
                    eigvect_matrix = np.real(eigvect_matrix)
                    eigval_matrix = eigval_matrix / np.sum(
                        np.abs(eigval_matrix))  # normalize eigenvalues to [0 and 1)
                    for en in range(num_eig):
                        index = np.argmax(eigval_matrix)
                        eigvect_dict[mode][var][ep].append(abs(eigvect_matrix[:, index]))
                        eigval_dict[mode][var][ep].append(eigval_matrix[index])
                        eigval_matrix[index] = 0

        return [fcd, fcd_segmented, eigvect_dict, eigval_dict]

    def _result_shape(self, input_shape):
        """Returns the shape of the fcd"""
        fcd_points = int((input_shape[0] - self.actual_sw) / self.actual_sp)
        result_shape = (fcd_points, fcd_points, input_shape[1], input_shape[3])
        return result_shape

    def _result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the main result of .
        """
        result_size = np.sum(list(map(np.prod, self._result_shape(input_shape)))) * 8.0  # Bytes
        return result_size

    @staticmethod
    def _spectral_dbscan(fcd, n_dim=2, eps=0.3, min_samples=50):
        fcd = fcd - fcd.min()
        se = SpectralEmbedding(n_dim, affinity="precomputed")
        xi = se.fit_transform(fcd)
        pd = pdist(xi)
        eps = np.percentile(pd, int(100 * eps))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xi)
        return xi.T, db.labels_

    @staticmethod
    def _compute_radii(xi, centered=False):
        if centered:
            xi = xi.copy() - xi.mean(axis=1).reshape((len(xi), 1))
        radii = np.sqrt(np.sum(xi ** 2, axis=0))
        return radii

    def _spectral_embedding(self, fcd):
        xi, _ = self._spectral_dbscan(fcd, 2)
        xir = self._compute_radii(xi, True)
        xir_sorted = np.sort(xir)
        xir_cutoff = 0.5 * xir_sorted[-1]
        return xir, xir_cutoff

    @staticmethod
    def _epochs_interval(xir, xir_cutoff, sp, sw):
        # Calculate the starting point and the ending point of each epoch of stability
        # sp=spanning, sw=sliding window
        epochs_dict = {}  # here the starting and the ending point will be stored
        thresholds = np.where(xir < xir_cutoff)
        tt = 0
        ep = 0
        while (tt + 2) < len(thresholds[0]):
            epochs_dict[ep] = [thresholds[0][tt]]  # starting point of epoch ep
            while ((tt + 2) != len(thresholds[0])) & (thresholds[0][tt + 1] == thresholds[0][tt] + 1):
                # until the vector is not finish and until each entries +1 is equal to the next one
                tt += 1
            epochs_dict[ep].append(thresholds[0][tt])
            tt += 1
            ep += 1
        # The relation between the index of the fcd[T] and the time point [t(i)] of the BOLD is the following:
        # T=0 indicates the FC calculate over the length of time that starts at (t=0) and that
        #     ends at (t=0)+sw (sw=length of the sliding window, sp=spanning between sliding windows)
        # T=1 indicates the FC calculate over the length of time that starts at (t=0)+sp and that ends at (t=0)+sp+sw
        # T=2 indicates the FC calculate over the length of time that starts at (t=0)+2*sp and ends at (t=0)+s*sp+sw
        # Thus we can write:
        # [interval of T=0]=[t(0)] U [t(0)+sw]
        # [interval of T=1]=[t(0)+sp] U [t(0)+sp+sw]
        # [interval of T=2]=[t(0)+2*sp] U [t(0)+2*sp+sw]
        # ...
        # [interval of T=i]=[t(0)+i*sp] U [t(0)+i*sp+sw]
        # Once we have the interval of the Epoch of stability that starts at T=s and ends at T=f
        # we want to calculate the FC taking the BOLD that starts and ends respectively at:
        # t(0)+s*sp; t(0)+f*sp+sw
        # Thus (we save the BOLD time in the epochs_extremes matrix)
        epochs_extremes = np.zeros((len(epochs_dict), 2), dtype=float)
        for ep in range(len(epochs_dict)):
            epochs_extremes[ep, 0] = epochs_dict[ep][0] * sp
            epochs_extremes[ep, 1] = epochs_dict[ep][1] * sp + sw
        return epochs_extremes
