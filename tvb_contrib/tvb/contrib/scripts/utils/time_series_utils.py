# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

from itertools import cycle
import numpy as np
from matplotlib.mlab import detrend_mean
from scipy.interpolate import interp1d, griddata
from scipy.signal import butter, filtfilt, welch, periodogram, spectrogram, decimate
from scipy.stats import zscore
from six import string_types
from tvb.basic.logger.builder import get_logger

logger = get_logger(__name__)
# Pointwise analyzers:

# x is assumed to be data (real numbers) arranged along the first dimension of an ndarray
from tvb.simulator.plot.utils import ensure_list, isequal_string, raise_value_error


def interval_scaling(x, min_targ=0.0, max_targ=1.0, min_orig=None, max_orig=None):
    if min_orig is None:
        min_orig = np.min(x, axis=0)
    if max_orig is None:
        max_orig = np.max(x, axis=0)
    scale_factor = (max_targ - min_targ) / (max_orig - min_orig)
    return min_targ + (x - min_orig) * scale_factor


def abs_envelope(x):
    x_mean = x.mean(axis=0) * np.ones(x.shape[1:])
    # Mean center each signal
    x -= x_mean
    # Compute the absolute value and add back the mean
    return np.abs(x) + x_mean


def spectrogram_envelope(x, fs, lpf=None, hpf=None, nperseg=None):
    envelope = []
    for xx in x.T:
        F, T, C = spectrogram(xx, fs, nperseg=nperseg)
        fmask = np.ones(F.shape, 'bool')
        if hpf:
            fmask *= F > hpf
        if lpf:
            fmask *= F < lpf
        envelope.append(C[fmask].sum(axis=0))
    return np.array(envelope).T, T


# Time domain:

def decimate_signals(signals, time, decim_ratio):
    if decim_ratio > 1:
        signals = decimate(signals, decim_ratio, axis=0, zero_phase=True, ftype="fir")
        time = decimate(time, decim_ratio, zero_phase=True, ftype="fir")
        dt = np.mean(np.diff(time))
        (n_times, n_signals) = signals.shape
        return signals, time, dt, n_times


def cut_signals_tails(signals, time, cut_tails):
    signals = signals[cut_tails[0]:-cut_tails[-1]]
    time = time[cut_tails[0]:-cut_tails[-1]]
    (n_times, n_signals) = signals.shape
    return signals, time, n_times


NORMALIZATION_METHODS = ["zscore", "mean", "min", "max", "baseline", "baseline-amplitude", "baseline-std", "minmax"]


def normalize_signals(signals, normalization=None, axis=None, percent=None):
    # Following matplotlibl.mlab detrend_mean:

    def matrix_subtract_along_axis(x, y, axis=0):
        "Return x minus y, where y corresponds to some statistic of x along the specified axis"
        if axis == 0 or axis is None or x.ndim <= 1:
            return x - y
        ind = [slice(None)] * x.ndim
        ind[axis] = np.newaxis
        return x - y[ind]

    def matrix_divide_along_axis(x, y, axis=0):
        "Return x divided by y, where y corresponds to some statistic of x along the specified axis"
        if axis == 0 or axis is None or x.ndim <= 1:
            return x / y
        ind = [slice(None)] * x.ndim
        ind[axis] = np.newaxis
        return x / y[ind]

    for norm, ax, prcnd in zip(ensure_list(normalization), cycle(ensure_list(axis)), cycle(ensure_list(percent))):
        if isinstance(norm, string_types):
            if isequal_string(norm, "zscore"):
                signals = zscore(signals, axis=ax)  # / 3.0
            elif isequal_string(norm, "baseline-std"):
                signals = normalize_signals(["baseline", "std"], axis=axis)
            elif norm.find("baseline") == 0 and norm.find("amplitude") >= 0:
                signals = normalize_signals(signals, ["baseline", norm.split("-")[1]], axis=axis, percent=percent)
            elif isequal_string(norm, "minmax"):
                signals = normalize_signals(signals, ["min", "max"], axis=axis)
            elif isequal_string(norm, "mean"):
                signals = detrend_mean(signals, axis=ax)
            elif isequal_string(norm, "baseline"):
                if prcnd is None:
                    prcnd = 1
                signals = matrix_subtract_along_axis(signals, np.percentile(signals, prcnd, axis=ax), axis=ax)
            elif isequal_string(norm, "min"):
                signals = matrix_subtract_along_axis(signals, np.min(signals, axis=ax), axis=ax)
            elif isequal_string(norm, "max"):
                signals = matrix_divide_along_axis(signals, np.max(signals, axis=ax), axis=ax)
            elif isequal_string(norm, "std"):
                signals = matrix_divide_along_axis(signals, signals.std(axis=ax), axis=ax)
            elif norm.find("amplitude") >= 0:
                if prcnd is None:
                    prcnd = [1, 99]
                amplitude = np.percentile(signals, prcnd[1], axis=ax) - np.percentile(signals, prcnd[0], axis=ax)
                this_ax = ax
                if isequal_string(norm.split("amplitude")[0], "max"):
                    amplitude = amplitude.max()
                    this_ax = None
                elif isequal_string(norm.split("amplitude")[0], "mean"):
                    amplitude = amplitude.mean()
                    this_ax = None
                signals = matrix_divide_along_axis(signals, amplitude, axis=this_ax)
            else:
                raise_value_error("Ignoring signals' normalization " + normalization +
                                  ",\nwhich is not one of the currently available " + str(NORMALIZATION_METHODS) + "!",
                                  logger)
    return signals


# Frequency domain:

def _butterworth_bandpass(fs, mode, lowcut, highcut, order=3):
    """
    Build a diggital Butterworth filter
    """
    nyq = 0.5 * fs
    freqs = []
    if lowcut is not None:
        freqs.append(lowcut / nyq)  # normalize frequency
    if highcut is not None:
        freqs.append(highcut / nyq)  # normalize frequency
    b, a = butter(order, freqs, btype=mode)  # btype : {'lowpass', 'highpass', 'bandpass', 'bandstop}, optional
    return b, a


def filter_data(data, fs, lowcut=None, highcut=None, mode='bandpass', order=3, axis=0):
    # get filter coefficients
    b, a = _butterworth_bandpass(fs, mode, lowcut, highcut, order)
    # filter data
    y = filtfilt(b, a, data, axis=axis)
    # y = lfilter(b, a, data, axis=axis)
    return y


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
