# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

"""
.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
.. moduleauthor:: Gabriel Florea <gabriel.florea@codemart.ro>
"""

import numpy as np
from scipy.signal import welch, periodogram, spectrogram
from scipy.interpolate import interp1d, griddata
from tvb.simulator.plot.config import CONFIGURED
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)


# Analyzers utils_pack:

def spectral_analysis(x, fs, freq=None, method="periodogram", output="spectrum", nfft=None, window='hanning',
                      nperseg=256, detrend='constant', noverlap=None, f_low=10.0, log_scale=False):
    if freq is None:
        freq = np.linspace(f_low, nperseg, nperseg - f_low - 1)
        df = freq[1] - freq[0]
    psd = []
    for iS in range(x.shape[1]):
        if method is welch:
            f, temp_psd = welch(x[:, iS],
                                fs=fs,  # sample rate
                                nfft=nfft,
                                window=window,  # apply a Hanning window before taking the DFT
                                nperseg=nperseg,  # compute periodograms of 256-long segments of x
                                detrend=detrend,
                                scaling="spectrum",
                                noverlap=noverlap,
                                return_onesided=True,
                                axis=0)
        else:
            f, temp_psd = periodogram(x[:, iS],
                                      fs=fs,  # sample rate
                                      nfft=nfft,
                                      window=window,  # apply a Hanning window before taking the DFT
                                      detrend=detrend,
                                      scaling="spectrum",
                                      return_onesided=True,
                                      axis=0)
        f = interp1d(f, temp_psd)
        temp_psd = f(freq)
        if output == "density":
            temp_psd /= (np.sum(temp_psd) * df)
        psd.append(temp_psd)
    # Stack them to a ndarray
    psd = np.stack(psd, axis=1)
    if output == "energy":
        return np.sum(psd, axis=0)
    else:
        if log_scale:
            psd = np.log(psd)
        return psd, freq


def time_spectral_analysis(x, fs, freq=None, mode="psd", nfft=None, window='hanning', nperseg=256, detrend='constant',
                           noverlap=None, f_low=10.0, calculate_psd=True, log_scale=False):
    # TODO: add a Continuous Wavelet Transform implementation
    if freq is None:
        freq = np.linspace(f_low, nperseg, nperseg - f_low - 1)
    stf = []
    for iS in range(x.shape[1]):
        f, t, temp_s = spectrogram(x[:, iS], fs=fs, nperseg=nperseg, nfft=nfft, window=window, mode=mode,
                                   noverlap=noverlap, detrend=detrend, return_onesided=True, scaling='spectrum', axis=0)
        t_mesh, f_mesh = np.meshgrid(t, f, indexing="ij")
        temp_s = griddata((t_mesh.flatten(), f_mesh.flatten()), temp_s.T.flatten(),
                          tuple(np.meshgrid(t, freq, indexing="ij")), method='linear')
        stf.append(temp_s)
    # Stack them to a ndarray
    stf = np.stack(stf, axis=2)
    if log_scale:
        stf = np.log(stf)
    if calculate_psd:
        psd, _ = spectral_analysis(x, fs, freq=freq, method="periodogram", output="spectrum", nfft=nfft, window=window,
                                   nperseg=nperseg, detrend=detrend, noverlap=noverlap, log_scale=log_scale)
        return stf, t, freq, psd
    else:
        return stf, t, freq


# Data structures utils

def isequal_string(a, b, case_sensitive=False):
    if case_sensitive:
        return a == b
    else:
        try:
            return a.lower() == b.lower()
        except AttributeError:
            LOG.warning("Case sensitive comparison!")
            return a == b


def ensure_list(arg):
    if not (isinstance(arg, list)):
        try:  # if iterable
            if isinstance(arg, (str, dict)):
                arg = [arg]
            elif hasattr(arg, "__iter__"):
                arg = list(arg)
            else:  # if not iterable
                arg = [arg]
        except:  # if not iterable
            arg = [arg]
    return arg


def generate_region_labels(n_regions, labels=[], str=". ", numbering=True, numbers=[]):
    if len(numbers) != n_regions:
        numbers = list(range(n_regions))
    if len(labels) == n_regions:
        if numbering:
            return np.array([str.join(["%d", "%s"]) % tuple(l) for l in zip(numbers, labels)])
        else:
            return np.array(labels)
    else:
        return np.array(["%d" % l for l in numbers])


# Computational utils

def compute_in_degree(weights):
    return np.expand_dims(np.sum(weights, axis=1), 1).T


def normalize_weights(weights, percentile=CONFIGURED.WEIGHTS_NORM_PERCENT, remove_diagonal=True, ceil=1.0):
    # Create the normalized connectivity weights:
    if len(weights) > 0:
        normalized_w = np.array(weights)
        if remove_diagonal:
            # Remove diagonal elements
            n_regions = normalized_w.shape[0]
            normalized_w *= (1.0 - np.eye(n_regions))
        # Normalize with the 95th percentile
        normalized_w = np.array(normalized_w / np.percentile(normalized_w, percentile))
        if ceil:
            if ceil is True:
                ceil = 1.0
            normalized_w[normalized_w > ceil] = ceil
        return normalized_w
    else:
        return np.array([])
