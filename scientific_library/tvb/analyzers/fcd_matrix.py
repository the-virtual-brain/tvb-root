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
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
import tvb.basic.traits.util as util
from tvb.basic.logger.builder import get_logger
from scipy.spatial.distance import pdist
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import DBSCAN
from numpy import linalg as LA



LOG = get_logger(__name__)

class FcdCalculator(core.Type):
    """
    Compute the the fcd of the timeseries

    Return a Fcd datatype, whose values of are between -1
    and 1, inclusive.
    Return eigenvectors of the FC calculated over the epochs of stability identified with spectral embedding in the FCD
    (the larger components of the eigenvectors, associated with the larger values of the eigenvalues of the FC, identified the functional hubs of the corresponding epoch of stability)
    """

    time_series = time_series.TimeSeriesRegion(
        label = "Time Series",
        required = True,
        doc = """The time-series for which the fcd matrices are calculated.""")

    sw = basic.Float(
        label="Sliding window length (ms)",
        default=120000,
        doc="""Length of the time window used to divided the time series.
                FCD matrix is calculated in the following way: the time series is divided in time window of fixed length and with an overlapping of fixed length.
                The datapoints within each window, centered at time ti, are used to calculate FC(ti) as Pearson correlation.
                The ij element of the FCD matrix is calculated as the Pearson correlation between FC(ti) and FC(tj) arranged in a vector.""")

    sp = basic.Float(
        label="Spanning between two consecutive sliding window (ms)",
        default=2000,
        doc="""Spanning= (time windows length)-(overlapping between two consecutive time window).
                FCD matrix is calculated in the following way: the time series is divided in time window of fixed length and with an overlapping of fixed length.
                The datapoints within each window, centered at time ti, are used to calculate FC(ti) as Pearson correlation.
                The ij element of the FCD matrix is calculated as the Pearson correlation between FC(ti) and FC(tj) arranged in a vector""")

    def evaluate(self):
        cls_attr_name = self.__class__.__name__ + ".time_series"
        self.time_series.trait["data"].log_debug(owner=cls_attr_name)

        sp=float(self.sp)
        sw=float(self.sw)
        # Pass sp and sw in the right reference (means considering the sample period)

        sp = sp  / self.time_series.sample_period
        sw = sw / self.time_series.sample_period
        # (fcd_points, fcd_points, state-variables, modes)
        input_shape = self.time_series.read_data_shape()

        result_shape = self.result_shape(input_shape)

        FCD = np.zeros(result_shape)
        FCstream = {}  # Here I will put before the stram of the FC
        start = -sp  # in order to well initialize the first starting point of the FC stream
        # One fcd matrix, for each state-var & mode.
        for mode in range(result_shape[3]):
            for var in range(result_shape[2]):
                for nfcd in range(result_shape[0]):
                    start += sp
                    current_slice = tuple([slice(int(start), int(start+sw) + 1), slice(var, var + 1),
                                           slice(input_shape[2]), slice(mode, mode + 1)])
                    data = self.time_series.read_data_slice(current_slice).squeeze()
                    FC = np.corrcoef(data.T)
                    Triangular = np.triu_indices(len(FC),
                                                 1)  # I organize the triangular part of the FC as a vector excluding the diagonal (always ones)
                    FCstream[nfcd] = FC[Triangular]
                for i in range(result_shape[0]):
                    j = i
                    while j < result_shape[0]:
                        fci = FCstream[i]
                        fcj = FCstream[j]
                        FCD[i, j, var, mode] = np.corrcoef(fci, fcj)[0, 1]
                        FCD[j, i, var, mode] = FCD[i, j, var, mode]
                        j += 1

        util.log_debug_array(LOG, FCD, "FCD")

        num_eig = 3  # I fix the value of the eigenvector to extract = 3, but maybe this can be changed

        Eigenvectors = {}  # in this dictionary I will store the eigenvector of each epoch, key1=mode, key2=var, key3=numb ep
        Eigenvalues = {}  # in this dictionary I will store the eigenvalues of each epoch, key1=mode, key2=var, key3=numb ep
        for mode in range(result_shape[3]):
            Eigenvectors[mode]={}
            Eigenvalues[mode]={}
            for var in range(result_shape[2]):
                Eigenvectors[mode][var]={}
                Eigenvalues[mode][var]={}
                FCD_matrix = FCD[:, :, var, mode]
                [xir, xir_cutoff] = spectral_embedding(FCD_matrix)
                epochs_extremes = epochs_interval(xir, xir_cutoff, sp, sw)
                FCD_segmented = FCD.copy()
                if epochs_extremes.shape[0]<=1: #(means that there are not epochs of stability, thus I will calculate eigenvector of the FC calculated over the entire timeseries)
                    epochs_extremes = np.zeros((2, 2), dtype=float)
                    epochs_extremes[1, 1] = input_shape[0]  # [0,0] setted because I skip first epoch
                else: #there are epochs so you can calculate the starting and the ending point of each epoch
                    FCD_segmented[xir > xir_cutoff, :,var, mode] = 1.1
                    FCD_segmented[:, xir > xir_cutoff, var, mode] = 1.1
                for ep in range(1,epochs_extremes.shape[0]):
                    Eigenvectors[mode][var][ep]=[]
                    Eigenvalues[mode][var][ep]=[]
                    current_slice = tuple([slice(int(epochs_extremes[ep][0]), int(epochs_extremes[ep][1]) + 1), slice(var, var + 1),
                                           slice(input_shape[2]), slice(mode, mode + 1)])
                    data = self.time_series.read_data_slice(current_slice).squeeze()
                    FC = np.corrcoef(data.T)
                    D, V = LA.eig(FC)
                    D = np.real(D)
                    V = np.real(V)
                    D = D / np.sum(np.abs(D))  # normalize eigenvalues between 0 and 1
                    for en in range(num_eig):
                        index = np.argmax(D)
                        Eigenvectors[mode][var][ep].append(V[:, index])
                        Eigenvalues[mode][var][ep].append(D[index])
                        D[index] = 0

        Connectivity=self.time_series.connectivity

        return [FCD, FCD_segmented, Eigenvectors, Eigenvalues, Connectivity]


    def result_shape(self, input_shape):
        """Returns the shape of the main result of ...."""
        sw=float(self.sw)
        sp=float(self.sp)
        sp = (sp)  / (self.time_series.sample_period)
        sw = (sw) / (self.time_series.sample_period)
        fcd_points = int((input_shape[0] - sw) / sp)
        result_shape = (fcd_points, fcd_points,
                        input_shape[1], input_shape[3])
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



#Methods:
def spectral_dbscan(FCD, n_dim=2, eps=0.3, min_samples=50):
    FCD_matrix=FCD
    FCD = FCD - FCD.min()
    se = SpectralEmbedding(n_dim, affinity="precomputed")
    xi = se.fit_transform(FCD)
    pd = pdist(xi)
    eps = np.percentile(pd, 100 * eps)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xi)
    return xi.T, db.labels_

def compute_radii(xi, centered=False):
    if centered:
        xi = xi.copy() - xi.mean(axis=1).reshape((len(xi), 1))
    radii = np.sqrt(np.sum(xi ** 2, axis=0))
    return radii

def spectral_embedding(FCD):
    xi, _ = spectral_dbscan(FCD, 2)
    xir = compute_radii(xi, True)
    xir_sorted = np.sort(xir)
    xir_cutoff = 0.5 * xir_sorted[-1]
    return xir, xir_cutoff

def epochs_interval(xir, xir_cutoff, sp, sw):
    # Calculate the start and the end of each epoch of stability
    # sp=spanning, sw=sliding window
    Epochs = {}  # in Epochs I will put the starting and the final time point of each epoch
    Tresholds = np.where(xir < xir_cutoff)
    tt = 0
    ep = 0
    while ((tt + 2) < len(Tresholds[0])):
        Epochs[ep] = [Tresholds[0][tt]]  # starting point of epoch ep
        while (((tt + 2) != len(Tresholds[0])) & (Tresholds[0][tt + 1] == Tresholds[0][
            tt] + 1)):  # until the vector is not finish and until each entries +1 is equal to the next one
            tt += 1
        Epochs[ep].append(Tresholds[0][tt])
        tt += 1
        ep += 1

    # The relation between the index of the FCD [T] and the time point [t(i)] of the BOLD is the following:
    # T=0 indicates the FC calculate over the length of time that starts at (t=0) and that ends at (t=0)+sw (sw=length of the sliding window, sp=spanning between sliding windows)
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
    epochs_extremes = np.zeros((len(Epochs), 2), dtype=float)
    for ep in range(len(Epochs)):
	    epochs_extremes[ep, 0] = Epochs[ep][0] * sp
	    epochs_extremes[ep, 1] = Epochs[ep][1] * sp + sw
    return epochs_extremes

