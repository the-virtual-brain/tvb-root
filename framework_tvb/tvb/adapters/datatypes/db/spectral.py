# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
from sqlalchemy import Column, Integer, ForeignKey, String, Float
from sqlalchemy.orm import relationship
from tvb.datatypes.spectral import FourierSpectrum, WaveletCoefficients, CoherenceSpectrum, ComplexCoherenceSpectrum
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex
from tvb.core.entities.model.model_datatype import DataTypeMatrix
from tvb.core.neotraits.db import from_ndarray


class FourierSpectrumIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    segment_length = Column(Float, nullable=False)
    windowing_function = Column(String, nullable=True)
    frequency_step = Column(Float, nullable=False)
    max_frequency = Column(Float, nullable=False)

    source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid), nullable=not FourierSpectrum.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_gid, primaryjoin=TimeSeriesIndex.gid == source_gid)

    def fill_from_has_traits(self, datatype):
        # type: (FourierSpectrum)  -> None
        super(FourierSpectrumIndex, self).fill_from_has_traits(datatype)
        self.segment_length = datatype.segment_length
        self.windowing_function = datatype.windowing_function
        self.frequency_step = datatype.frequency_step
        self.max_frequency = datatype.max_frequency
        self.source_gid = datatype.source.gid.hex


class WaveletCoefficientsIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid), nullable=not WaveletCoefficients.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_gid, primaryjoin=TimeSeriesIndex.gid == source_gid)

    mother = Column(String, nullable=False)
    normalisation = Column(String, nullable=False)
    q_ratio = Column(Float, nullable=False)
    sample_period = Column(Float, nullable=False)
    number_of_scales = Column(Integer, nullable=False)

    frequencies_min = Column(Float)
    frequencies_max = Column(Float)

    def fill_from_has_traits(self, datatype):
        # type: (WaveletCoefficients)  -> None
        super(WaveletCoefficientsIndex, self).fill_from_has_traits(datatype)
        self.mother = datatype.mother
        self.normalisation = datatype.normalisation
        self.q_ratio = datatype.q_ratio
        self.sample_period = datatype.sample_period
        self.number_of_scales = datatype.frequencies.shape[0]
        self.frequencies_min, self.frequencies_max, _ = from_ndarray(datatype.frequency)
        self.source_gid = datatype.source.gid.hex


class CoherenceSpectrumIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid), nullable=not CoherenceSpectrum.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_gid, primaryjoin=TimeSeriesIndex.gid == source_gid)

    nfft = Column(Integer, nullable=False)
    frequencies_min = Column(Float)
    frequencies_max = Column(Float)

    def fill_from_has_traits(self, datatype):
        # type: (CoherenceSpectrum)  -> None
        super(CoherenceSpectrumIndex, self).fill_from_has_traits(datatype)
        self.nfft = datatype.nfft
        self.frequencies_min, self.frequencies_max, _ = from_ndarray(datatype.frequency)
        self.source_gid = datatype.source.gid.hex


class ComplexCoherenceSpectrumIndex(DataTypeMatrix):
    id = Column(Integer, ForeignKey(DataTypeMatrix.id), primary_key=True)

    source_gid = Column(String(32), ForeignKey(TimeSeriesIndex.gid),
                        nullable=not ComplexCoherenceSpectrum.source.required)
    source = relationship(TimeSeriesIndex, foreign_keys=source_gid, primaryjoin=TimeSeriesIndex.gid == source_gid)

    epoch_length = Column(Float, nullable=False)
    segment_length = Column(Float, nullable=False)
    windowing_function = Column(String, nullable=False)
    frequency_step = Column(Float, nullable=False)
    max_frequency = Column(Float, nullable=False)

    def fill_from_has_traits(self, datatype):
        # type: (ComplexCoherenceSpectrum)  -> None
        super(ComplexCoherenceSpectrumIndex, self).fill_from_has_traits(datatype)
        self.epoch_length = datatype.epoch_length
        self.segment_length = datatype.segment_length
        self.windowing_function = datatype.windowing_function
        self.frequency_step = datatype.freq_step
        self.max_frequency = datatype.max_freq
        self.source_gid = datatype.source.gid.hex
