# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
The Data component of Spectral datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>

"""

import tvb.basic.traits.types_basic as basic
import tvb.basic.traits.core as core
import tvb.datatypes.arrays as arrays
import tvb.datatypes.time_series as time_series



class FourierSpectrumData(arrays.MappedArray):
    """
    Result of a Fourier  Analysis.
    """
    #Overwrite attribute from superclass
    array_data = arrays.ComplexArray(file_storage=core.FILE_STORAGE_EXPAND)

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="Links to the time-series on which the FFT is applied.")

    segment_length = basic.Float(
        label="Segment length",
        doc="""The timeseries was segmented into equally sized blocks
            (overlapping if necessary), prior to the application of the FFT.
            The segement length determines the frequency resolution of the
            resulting spectra.""")

    windowing_function = basic.String(
        label="Windowing function",
        doc="""The windowing function applied to each time segment prior to
            application of the FFT.""")

    amplitude = arrays.FloatArray(
        label="Amplitude",
        file_storage=core.FILE_STORAGE_EXPAND)

    phase = arrays.FloatArray(
        label="Phase",
        file_storage=core.FILE_STORAGE_EXPAND)

    power = arrays.FloatArray(
        label="Power",
        file_storage=core.FILE_STORAGE_EXPAND)

    average_power = arrays.FloatArray(
        label="Average Power",
        file_storage=core.FILE_STORAGE_EXPAND)

    normalised_average_power = arrays.FloatArray(
        label="Normalised Power",
        file_storage=core.FILE_STORAGE_EXPAND)

    __generate_table__ = True



class WaveletCoefficientsData(arrays.MappedArray):
    """
    This class bundles all the elements of a Wavelet Analysis into a single 
    object, including the input TimeSeries datatype and the output results as 
    arrays (FloatArray)
    """
    #Overwrite attribute from superclass
    array_data = arrays.ComplexArray()

    source = time_series.TimeSeries(label="Source time-series")

    mother = basic.String(
        label="Mother wavelet",
        default="morlet",
        doc="""A string specifying the type of mother wavelet to use,
            default is 'morlet'.""") # default to 'morlet'

    sample_period = basic.Float(label="Sample period")
    #sample_rate = basic.Integer(label = "")  inversely related

    frequencies = arrays.FloatArray(
        label="Frequencies",
        doc="A vector that maps scales to frequencies.")

    normalisation = basic.String(label="Normalisation type")
    # 'unit energy' | 'gabor'

    q_ratio = basic.Float(label="Q-ratio", default=5.0)

    amplitude = arrays.FloatArray(
        label="Amplitude",
        file_storage=core.FILE_STORAGE_EXPAND)

    phase = arrays.FloatArray(
        label="Phase",
        file_storage=core.FILE_STORAGE_EXPAND)

    power = arrays.FloatArray(
        label="Power",
        file_storage=core.FILE_STORAGE_EXPAND)

    __generate_table__ = True



class CoherenceSpectrumData(arrays.MappedArray):
    """
    Result of a NodeCoherence Analysis.
    """
    #Overwrite attribute from superclass
    array_data = arrays.FloatArray(file_storage=core.FILE_STORAGE_EXPAND)

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="""Links to the time-series on which the node_coherence is
            applied.""")

    nfft = basic.Integer(
        label="Data-points per block",
        default=256,
        doc="""NOTE: must be a power of 2""")

    frequency = arrays.FloatArray(label="Frequency")

    __generate_table__ = True



class ComplexCoherenceSpectrumData(arrays.MappedArray):
    """
    Result of a NodeComplexCoherence Analysis.
    """

    cross_spectrum = arrays.ComplexArray(
        label="The cross spectrum",
        file_storage=core.FILE_STORAGE_EXPAND,
        doc=""" A complex ndarray that contains the nodes x nodes cross
            spectrum for every frequency frequency and for every segment.""")

    array_data = arrays.ComplexArray(
        label="Complex Coherence",
        file_storage=core.FILE_STORAGE_EXPAND,
        doc="""The complex coherence coefficients calculated from the cross
            spectrum. The imaginary values of this complex ndarray represent the 
            imaginary coherence.""")

    source = time_series.TimeSeries(
        label="Source time-series",
        doc="""Links to the time-series on which the node_coherence is
            applied.""")

    epoch_length = basic.Float(
        label="Epoch length",
        doc="""The timeseries was segmented into equally sized blocks
            (overlapping if necessary), prior to the application of the FFT.
            The segement length determines the frequency resolution of the
            resulting spectra.""")

    segment_length = basic.Float(
        label="Segment length",
        doc="""The timeseries was segmented into equally sized blocks
            (overlapping if necessary), prior to the application of the FFT.
            The segement length determines the frequency resolution of the
            resulting spectra.""")

#    frequency = arrays.FloatArray(
#        label = "Frequency",
#        doc = """DOC ME""")

    windowing_function = basic.String(
        label="Windowing function",
        doc="""The windowing function applied to each time segment prior to
            application of the FFT.""")

#    number_of_epochs = basic.Integer(
#        label = "Number of epochs",
#        doc = """DOC ME""")
#
#    number_of_segments = basic.Integer(
#        label = "Number of segments",
#        doc = """DOC ME""")

    __generate_table__ = True

