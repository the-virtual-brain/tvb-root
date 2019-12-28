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

from tvb.simulator.monitors import Monitor, Raw, SpatialAverage, Projection, EEG, MEG, iEEG, Bold
from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.neotraits.h5 import Scalar, DataSet, Reference, EquationScalar


class MonitorH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(MonitorH5, self).__init__(path)
        self.period = Scalar(Monitor.period, self)
        self.variables_of_interest = DataSet(Monitor.variables_of_interest, self)


class RawH5(MonitorH5):

    def __init__(self, path):
        super(RawH5, self).__init__(path)
        self.period = Scalar(Raw.period, self)
        self.variables_of_interest = DataSet(Raw.variables_of_interest, self)


class SubSampleH5(MonitorH5):
    """"""


class SpatialAverageH5(MonitorH5):

    def __init__(self, path):
        super(SpatialAverageH5, self).__init__(path)
        self.spatial_mask = DataSet(SpatialAverage.spatial_mask, self)
        self.default_mask = Scalar(SpatialAverage.default_mask, self)


class GlobalAverageH5(MonitorH5):
    """"""


class TemporalAverageH5(MonitorH5):
    """"""


class ProjectionH5(MonitorH5):

    def __init__(self, path):
        super(ProjectionH5, self).__init__(path)
        self.region_mapping = Reference(Projection.region_mapping, self)
        self.obnoise = Reference(Projection.obsnoise, self)

    def store(self, datatype, scalars_only=False, store_references=False):
        # type: (Projection, bool, bool) -> None
        super(ProjectionH5, self).store(datatype, scalars_only, store_references)
        self.region_mapping.store(datatype.region_mapping.gid)


class EEGH5(ProjectionH5):

    def __init__(self, path):
        super(EEGH5, self).__init__(path)
        self.projection = Reference(EEG.projection, self)
        self.sensors = Reference(EEG.sensors, self)
        self.reference = Scalar(EEG.reference, self)
        self.sigma = Scalar(EEG.sigma, self)

    def store(self, datatype, scalars_only=False, store_references=False):
        # type: (Projection, bool, bool) -> None
        super(EEGH5, self).store(datatype, scalars_only, store_references)
        self.projection.store(datatype.projection.gid)
        self.sensors.store(datatype.sensors.gid)


class MEGH5(ProjectionH5):

    def __init__(self, path):
        super(MEGH5, self).__init__(path)
        self.projection = Reference(MEG.projection, self)
        self.sensors = Reference(MEG.sensors, self)

    def store(self, datatype, scalars_only=False, store_references=False):
        # type: (Projection, bool, bool) -> None
        super(MEGH5, self).store(datatype, scalars_only, store_references)
        self.projection.store(datatype.projection.gid)
        self.sensors.store(datatype.sensors.gid)


class iEEGH5(ProjectionH5):

    def __init__(self, path):
        super(iEEGH5, self).__init__(path)
        self.projection = Reference(iEEG.projection, self)
        self.sensors = Reference(iEEG.sensors, self)
        self.sigma = Scalar(iEEG.sigma, self)

    def store(self, datatype, scalars_only=False, store_references=False):
        # type: (Projection, bool, bool) -> None
        super(iEEGH5, self).store(datatype, scalars_only, store_references)
        self.projection.store(datatype.projection.gid)
        self.sensors.store(datatype.sensors.gid)


class BoldH5(MonitorH5):

    def __init__(self, path):
        super(BoldH5, self).__init__(path)
        self.period = Scalar(Bold.period, self)
        self.hrf_kernel = EquationScalar(Bold.hrf_kernel, self)
        self.hrf_length = Scalar(Bold.hrf_length, self)


class BoldRegionROIH5(BoldH5):
    """"""
