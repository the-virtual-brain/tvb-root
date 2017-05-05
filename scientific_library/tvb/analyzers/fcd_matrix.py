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
Adapter that uses the traits module to generate interfaces for ... Analyzer.

.. moduleauthor:: Francesca Melozzi <france.melozzi@gmail.com>
.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""
import numpy as np
import tvb.datatypes.time_series as time_series
from tvb.basic.traits import core, types_basic, util
from tvb.basic.logger.builder import get_logger
from scipy.spatial.distance import pdist
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import DBSCAN
from numpy import linalg


LOG = get_logger(__name__)


class FcdCalculator(core.Type):
    """
    The present class will do the following actions:

    - Compute the the fcd of the timeseries; the fcd is calculated in the following way:
        the time series is divided in time window of fixed length and with an overlapping of fixed length.
        The data-points within each window, centered at time ti, are used to calculate FC(ti) as Pearson correlation.
        The ij element of the FCD matrix is calculated as the Pearson correlation between FC(ti) and FC(tj) -in a vector
    - Apply to the fcd the spectral embedding algorithm in order to calculate epochs of stability of the fcd
        (length of time during which FC matrix are high correlated).

    The algorithm can produce 2 kind of results:

    - case 1: the algorithm is able to identify the epochs of stability
        -- fcs calculated over the epochs of stability (excluded the first one = artifact, due to initial conditions)
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

    time_series = time_series.TimeSeriesRegion(
        label="Time Series",
        required=True,
        doc="""The time-series for which the fcd matrices are calculated.""")

    sw = types_basic.Float(
        label="Sliding window length (ms)",
        default=120000,
        doc="""Length of the time window used to divided the time series.
        FCD matrix is calculated in the following way: the time series is divided in time window of fixed length and
        with an overlapping of fixed length. The datapoints within each window, centered at time ti, are used to
        calculate FC(ti) as Pearson correlation. The ij element of the FCD matrix is calculated as the Pearson
        Correlation between FC(ti) and FC(tj) arranged in a vector.""")

    sp = types_basic.Float(
        label="Spanning between two consecutive sliding window (ms)",
        default=2000,
        doc="""Spanning= (time windows length)-(overlapping between two consecutive time window). FCD matrix is
        calculated in the following way: the time series is divided in time window of fixed length and with an
        overlapping of fixed length. The datapoints within each window, centered at time ti, are used to calculate
        FC(ti) as Pearson Correlation. The ij element of the FCD matrix is calculated as the Pearson correlation
        between FC(ti) and FC(tj) arranged in a vector""")

    def evaluate(self):
        cls_attr_name = self.__class__.__name__ + ".time_series"
        self.time_series.trait["data"].log_debug(owner=cls_attr_name)

        # Pass sp and sw in the right time reference (means considering the sample period)
        sp = float(self.sp) / self.time_series.sample_period
        sw = float(self.sw) / self.time_series.sample_period

        input_shape = self.time_series.read_data_shape()
        result_shape = self.result_shape(input_shape)

        fcd = np.zeros(result_shape)
        fc_stream = {}  # dict where the fc calculated over the sliding window will be stored
        start = -sp  # in order to well initialize the first starting point of the FC stream
        for mode in range(result_shape[3]):
            for var in range(result_shape[2]):
                for nfcd in range(result_shape[0]):
                    start += sp
                    current_slice = tuple([slice(int(start), int(start+sw) + 1), slice(var, var + 1),
                                           slice(input_shape[2]), slice(mode, mode + 1)])
                    data = self.time_series.read_data_slice(current_slice).squeeze()
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

        util.log_debug_array(LOG, fcd, "FCD")

        num_eig = 3  # number of the eigenvector that will be extracted

        eigvect_dict = {}  # holds eigenvectors of the fcs calculated over the epochs, key1=mode, key2=var, key3=numb ep
        eigval_dict = {}  # holds eigenvalues of the fcs calculated over the epochs, key1=mode, key2=var, key3=numb ep
        for mode in range(result_shape[3]):
            eigvect_dict[mode] = {}
            eigval_dict[mode] = {}
            for var in range(result_shape[2]):
                eigvect_dict[mode][var] = {}
                eigval_dict[mode][var] = {}
                fcd_matrix = fcd[:, :, var, mode]
                [xir, xir_cutoff] = spectral_embedding(fcd_matrix)
                epochs_extremes = epochs_interval(xir, xir_cutoff, sp, sw)
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
                    data = self.time_series.read_data_slice(current_slice).squeeze()
                    fc = np.corrcoef(data.T)  # calculate fc over the epoch of stability
                    eigval_matrix, eigvect_matrix = linalg.eig(fc)
                    eigval_matrix = np.real(eigval_matrix)
                    eigvect_matrix = np.real(eigvect_matrix)
                    eigval_matrix = eigval_matrix / np.sum(np.abs(eigval_matrix))  # normalize eigenvalues to [0 and 1)
                    for en in range(num_eig):
                        index = np.argmax(eigval_matrix)
                        eigvect_dict[mode][var][ep].append(abs(eigvect_matrix[:, index]))
                        eigval_dict[mode][var][ep].append(eigval_matrix[index])
                        eigval_matrix[index] = 0

        return [fcd, fcd_segmented, eigvect_dict, eigval_dict, self.time_series.connectivity]


    def result_shape(self, input_shape):
        """Returns the shape of the fcd"""
        sp = float(self.sp) / self.time_series.sample_period
        sw = float(self.sw) / self.time_series.sample_period
        fcd_points = int((input_shape[0] - sw) / sp)
        result_shape = (fcd_points, fcd_points, input_shape[1], input_shape[3])
        return result_shape


    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the main result of .
        """
        result_size = np.sum(map(np.prod, self.result_shape(input_shape))) * 8.0  # Bytes
        return result_size


    def extended_result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the extended result of the ....
        That is, it includes storage of the evaluated ... attributes
        such as ..., etc.
        """
        extend_size = self.result_size(input_shape)  # Currently no derived attributes.
        return extend_size


# Methods:
def spectral_dbscan(fcd, n_dim=2, eps=0.3, min_samples=50):
    fcd = fcd - fcd.min()
    se = SpectralEmbedding(n_dim, affinity="precomputed")
    xi = se.fit_transform(fcd)
    pd = pdist(xi)
    eps = np.percentile(pd, 100 * eps)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xi)
    return xi.T, db.labels_


def compute_radii(xi, centered=False):
    if centered:
        xi = xi.copy() - xi.mean(axis=1).reshape((len(xi), 1))
    radii = np.sqrt(np.sum(xi ** 2, axis=0))
    return radii


def spectral_embedding(fcd):
    xi, _ = spectral_dbscan(fcd, 2)
    xir = compute_radii(xi, True)
    xir_sorted = np.sort(xir)
    xir_cutoff = 0.5 * xir_sorted[-1]
    return xir, xir_cutoff


def epochs_interval(xir, xir_cutoff, sp, sw):
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
    # T=2 indicates the FC calculate over the length of time that starts at (t=0)+2*sp and that ends at (t=0)+s*sp+sw
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
