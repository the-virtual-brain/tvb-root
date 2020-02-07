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
from tvb.simulator.coupling import *
from tvb.core.entities.file.simulator.configurations_h5 import SimulatorConfigurationH5
from tvb.core.neotraits.h5 import DataSet, Scalar


class LinearH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(LinearH5, self).__init__(path)
        self.a = DataSet(Linear.a, self)
        self.b = DataSet(Linear.b, self)


class ScalingH5(SimulatorConfigurationH5):
    def __init__(self, path):
        super(ScalingH5, self).__init__(path)
        self.a = DataSet(Scaling.a, self)


class HyperbolicTangentH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(HyperbolicTangentH5, self).__init__(path)
        self.a = DataSet(HyperbolicTangent.a, self)
        self.b = DataSet(HyperbolicTangent.b, self)
        self.midpoint = DataSet(HyperbolicTangent.midpoint, self)
        self.sigma = DataSet(HyperbolicTangent.sigma, self)


class SigmoidalH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(SigmoidalH5, self).__init__(path)
        self.cmin = DataSet(Sigmoidal.cmin, self)
        self.cmax = DataSet(Sigmoidal.cmax, self)
        self.midpoint = DataSet(Sigmoidal.midpoint, self)
        self.a = DataSet(Sigmoidal.a, self)
        self.sigma = DataSet(Sigmoidal.sigma, self)


class SigmoidalJansenRitH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(SigmoidalJansenRitH5, self).__init__(path)
        self.cmin = DataSet(SigmoidalJansenRit.cmin, self)
        self.cmax = DataSet(SigmoidalJansenRit.cmax, self)
        self.midpoint = DataSet(SigmoidalJansenRit.midpoint, self)
        self.r = DataSet(SigmoidalJansenRit.r, self)
        self.a = DataSet(SigmoidalJansenRit.a, self)


class PreSigmoidalH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(PreSigmoidalH5, self).__init__(path)
        self.H = DataSet(PreSigmoidal.H, self)
        self.Q = DataSet(PreSigmoidal.Q, self)
        self.G = DataSet(PreSigmoidal.G, self)
        self.P = DataSet(PreSigmoidal.P, self)
        self.theta = DataSet(PreSigmoidal.theta, self)
        self.dynamic = Scalar(PreSigmoidal.dynamic, self)
        self.globalT = Scalar(PreSigmoidal.globalT, self)


class DifferenceH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(DifferenceH5, self).__init__(path)
        self.a = DataSet(Difference.a, self)


class KuramotoH5(SimulatorConfigurationH5):

    def __init__(self, path):
        super(KuramotoH5, self).__init__(path)
        self.a = DataSet(Kuramoto.a, self)
