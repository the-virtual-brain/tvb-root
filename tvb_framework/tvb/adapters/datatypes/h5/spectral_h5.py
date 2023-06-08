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
import json

import numpy
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Reference, Enum
from tvb.datatypes.spectral import FourierSpectrum, WaveletCoefficients, CoherenceSpectrum, ComplexCoherenceSpectrum


class DataTypeMatrixH5(H5File):
    def get_min_max_values(self):
        """
        Retrieve the minimum and maximum values from the metadata.
        :returns: (minimum_value, maximum_value)
        """
        metadata = self.array_data.get_cached_metadata()
        return metadata.min, metadata.max


class FourierSpectrumH5(DataTypeMatrixH5):

    def __init__(self, path):
        super(FourierSpectrumH5, self).__init__(path)
        self.array_data = DataSet(FourierSpectrum.array_data, self, expand_dimension=2)
        self.source = Reference(FourierSpectrum.source, self)
        self.segment_length = Scalar(FourierSpectrum.segment_length, self)
        self.windowing_function = Enum(FourierSpectrum.windowing_function, self)
        self.amplitude = DataSet(FourierSpectrum.amplitude, self, expand_dimension=2)
        self.phase = DataSet(FourierSpectrum.phase, self, expand_dimension=2)
        self.power = DataSet(FourierSpectrum.power, self, expand_dimension=2)
        self.average_power = DataSet(FourierSpectrum.average_power, self, expand_dimension=2)
        self.normalised_average_power = DataSet(FourierSpectrum.normalised_average_power, self,
                                                expand_dimension=2)

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        # self.store_data_chunk('array_data', partial_result, grow_dimension=2, close_file=False)

        # mhtodo: these computations on the partial_result belong in the caller not here

        self.array_data.append(partial_result.array_data, close_file=False)

        partial_result.compute_amplitude()
        self.amplitude.append(partial_result.amplitude, close_file=False)

        partial_result.compute_phase()
        self.phase.append(partial_result.phase, close_file=False)

        partial_result.compute_power()
        self.power.append(partial_result.power, close_file=False)

        partial_result.compute_average_power()
        self.average_power.append(partial_result.average_power, close_file=False)

        partial_result.compute_normalised_average_power()
        self.normalised_average_power.append(partial_result.normalised_average_power, close_file=False)

    def get_fourier_data(self, selected_state, selected_mode, normalized):
        shape = self.array_data.shape

        slices = (slice(shape[0]),
                  slice(int(selected_state), min(int(selected_state) + 1, shape[1]), None),
                  slice(shape[2]),
                  slice(int(selected_mode), min(int(selected_mode) + 1, shape[3]), None))

        if normalized == "yes":
            data_matrix = self.normalised_average_power[slices]
        else:
            data_matrix = self.average_power[slices]

        data_matrix = data_matrix.reshape((shape[0], shape[2]))
        ymin = numpy.amin(data_matrix)
        ymax = numpy.amax(data_matrix)
        data_matrix = data_matrix.transpose()
        # mhtodo: this form with string inputs and json outputs belongs in some viewer not here
        return dict(data_matrix=json.dumps(data_matrix.tolist()),
                    ymin=ymin,
                    ymax=ymax)


class WaveletCoefficientsH5(DataTypeMatrixH5):

    def __init__(self, path):
        super(WaveletCoefficientsH5, self).__init__(path)
        self.array_data = DataSet(WaveletCoefficients.array_data, self, expand_dimension=2)
        self.source = Reference(WaveletCoefficients.source, self)
        self.mother = Scalar(WaveletCoefficients.mother, self)
        self.sample_period = Scalar(WaveletCoefficients.sample_period, self)
        self.frequencies = DataSet(WaveletCoefficients.frequencies, self)
        self.normalisation = Scalar(WaveletCoefficients.normalisation, self)
        self.q_ratio = Scalar(WaveletCoefficients.q_ratio, self)
        self.amplitude = DataSet(WaveletCoefficients.amplitude, self, expand_dimension=2)
        self.phase = DataSet(WaveletCoefficients.phase, self, expand_dimension=2)
        self.power = DataSet(WaveletCoefficients.power, self, expand_dimension=2)

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        # mhtodo: these computations on the partial_result belong in the caller not here

        self.array_data.append(partial_result.array_data, close_file=False)

        partial_result.compute_amplitude()
        self.amplitude.append(partial_result.amplitude, close_file=False)

        partial_result.compute_phase()
        self.phase.append(partial_result.phase, close_file=False)

        partial_result.compute_power()
        self.power.append(partial_result.power, close_file=False)


class CoherenceSpectrumH5(DataTypeMatrixH5):

    def __init__(self, path):
        super(CoherenceSpectrumH5, self).__init__(path)
        self.array_data = DataSet(CoherenceSpectrum.array_data, self, expand_dimension=3)
        self.source = Reference(CoherenceSpectrum.source, self)
        self.nfft = Scalar(CoherenceSpectrum.nfft, self)
        self.frequency = DataSet(CoherenceSpectrum.frequency, self)

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.array_data.append(partial_result.array_data, close_file=False)


class ComplexCoherenceSpectrumH5(DataTypeMatrixH5):
    spectrum_types = ["Imaginary", "Real", "Absolute"]

    def __init__(self, path):
        super(ComplexCoherenceSpectrumH5, self).__init__(path)
        self.cross_spectrum = DataSet(ComplexCoherenceSpectrum.cross_spectrum, self, expand_dimension=2)
        self.array_data = DataSet(ComplexCoherenceSpectrum.array_data, self, expand_dimension=2)
        self.source = Reference(ComplexCoherenceSpectrum.source, self)
        self.epoch_length = Scalar(ComplexCoherenceSpectrum.epoch_length, self)
        self.segment_length = Scalar(ComplexCoherenceSpectrum.segment_length, self)
        self.windowing_function = Scalar(ComplexCoherenceSpectrum.windowing_function, self)

    def write_data_slice(self, partial_result):
        """
        Append chunk.
        """
        self.cross_spectrum.append(partial_result.cross_spectrum, close_file=False)

        self.array_data.append(partial_result.array_data, close_file=False)

    def get_spectrum_data(self, selected_spectrum):
        shape = self.array_data.shape
        slices = (slice(shape[0]), slice(shape[1]), slice(shape[2]))

        if selected_spectrum == self.spectrum_types[0]:
            data_matrix = self.array_data[slices].imag
            indices = numpy.triu_indices(shape[0], 1)
            data_matrix = data_matrix[indices]

        elif selected_spectrum == self.spectrum_types[1]:
            data_matrix = self.array_data[slices].real
            data_matrix = data_matrix.reshape(shape[0] * shape[0], shape[2])

        else:
            data_matrix = self.array_data[slices]
            data_matrix = numpy.absolute(data_matrix)
            data_matrix = data_matrix.reshape(shape[0] * shape[0], shape[2])

        coh_spec_sd = numpy.std(data_matrix, axis=0)
        coh_spec_av = numpy.mean(data_matrix, axis=0)

        ymin = numpy.amin(coh_spec_av - coh_spec_sd)
        ymax = numpy.amax(coh_spec_av + coh_spec_sd)

        coh_spec_sd = json.dumps(coh_spec_sd.tolist())
        coh_spec_av = json.dumps(coh_spec_av.tolist())

        return dict(coh_spec_sd=coh_spec_sd,
                    coh_spec_av=coh_spec_av,
                    ymin=ymin,
                    ymax=ymax)
