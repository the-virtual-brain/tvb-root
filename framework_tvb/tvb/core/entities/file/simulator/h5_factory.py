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
from tvb.datatypes.cortex import Cortex
from tvb.simulator.coupling import Coupling
from tvb.simulator.integrators import *
from tvb.simulator.models import ModelsEnum
from tvb.simulator.monitors import Monitor, SubSample, GlobalAverage, TemporalAverage
from tvb.simulator.noise import Additive, Multiplicative, Noise


# TODO: rethink this solution
def config_h5_factory(config_class):
    from tvb.core.entities.file.simulator.cortex_h5 import CortexH5

    if issubclass(config_class, Noise):
        return noise_h5_factory(config_class)
    if issubclass(config_class, Integrator):
        return integrator_h5_factory(config_class)
    if issubclass(config_class, Coupling):
        return coupling_h5_factory(config_class)
    if issubclass(config_class, ModelsEnum.BASE_MODEL.get_class()):
        return model_h5_factory(config_class)
    if issubclass(config_class, Monitor):
        return monitor_h5_factory(config_class)
    if issubclass(config_class, Cortex):
        return CortexH5
    return None


def noise_h5_factory(noise_class):
    from tvb.core.entities.file.simulator.noise_h5 import AdditiveH5, MultiplicativeH5

    noise_class_to_h5 = {
        Additive: AdditiveH5,
        Multiplicative: MultiplicativeH5
    }

    return noise_class_to_h5.get(noise_class)


def integrator_h5_factory(integrator_class):
    from tvb.core.entities.file.simulator.integrator_h5 import IntegratorH5, IntegratorStochasticH5

    integrator_class_to_h5 = {
        HeunDeterministic: IntegratorH5,
        HeunStochastic: IntegratorStochasticH5,
        EulerDeterministic: IntegratorH5,
        EulerStochastic: IntegratorStochasticH5,
        RungeKutta4thOrderDeterministic: IntegratorH5,
        Identity: IntegratorH5,
        VODE: IntegratorH5,
        VODEStochastic: IntegratorStochasticH5,
        Dopri5: IntegratorH5,
        Dopri5Stochastic: IntegratorStochasticH5,
        Dop853: IntegratorH5,
        Dop853Stochastic: IntegratorStochasticH5
    }

    return integrator_class_to_h5.get(integrator_class)


def coupling_h5_factory(coupling_class):
    from tvb.simulator.coupling import Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, \
        Difference, Kuramoto
    from tvb.core.entities.file.simulator.coupling_h5 import LinearH5, ScalingH5, HyperbolicTangentH5, SigmoidalH5, \
        SigmoidalJansenRitH5, PreSigmoidalH5, DifferenceH5, KuramotoH5

    coupling_class_to_h5 = {
        Linear: LinearH5,
        Scaling: ScalingH5,
        HyperbolicTangent: HyperbolicTangentH5,
        Sigmoidal: SigmoidalH5,
        SigmoidalJansenRit: SigmoidalJansenRitH5,
        PreSigmoidal: PreSigmoidalH5,
        Difference: DifferenceH5,
        Kuramoto: KuramotoH5
    }

    return coupling_class_to_h5.get(coupling_class)


def model_h5_factory(model_class):
    from tvb.core.entities.file.simulator.model_h5 import EpileptorH5, Epileptor2DH5, EpileptorCodim3H5, \
        EpileptorCodim3SlowModH5, HopfieldH5, JansenRitH5, ZetterbergJansenH5, EpileptorRestingStateH5, \
        LarterBreakspearH5, LinearH5, Generic2dOscillatorH5, KuramotoH5, ReducedSetFitzHughNagumoH5, \
        ReducedSetHindmarshRoseH5, WilsonCowanH5, ReducedWongWangH5, ReducedWongWangExcInhH5, \
        ZerlautAdaptationFirstOrderH5, \
        ZerlautAdaptationSecondOrderH5, SupHopfH5
    model_class_to_h5 = {
        ModelsEnum.EPILEPTOR.get_class(): EpileptorH5,
        ModelsEnum.EPILEPTOR_2D.get_class(): Epileptor2DH5,
        ModelsEnum.EPILEPTOR_CODIM_3.get_class(): EpileptorCodim3H5,
        ModelsEnum.EPILEPTOR_CODIM_3_SLOW.get_class(): EpileptorCodim3SlowModH5,
        ModelsEnum.HOPFIELD.get_class(): HopfieldH5,
        ModelsEnum.JANSEN_RIT.get_class(): JansenRitH5,
        ModelsEnum.ZETTERBERG_JANSEN.get_class(): ZetterbergJansenH5,
        ModelsEnum.EPILEPTOR_RS.get_class(): EpileptorRestingStateH5,
        ModelsEnum.LARTER_BREAKSPEAR.get_class(): LarterBreakspearH5,
        ModelsEnum.LINEAR.get_class(): LinearH5,
        ModelsEnum.GENERIC_2D_OSCILLATOR.get_class(): Generic2dOscillatorH5,
        ModelsEnum.KURAMOTO.get_class(): KuramotoH5,
        ModelsEnum.REDUCED_SET_FITZ_HUGH_NAGUMO.get_class(): ReducedSetFitzHughNagumoH5,
        ModelsEnum.REDUCED_SET_HINDMARSH_ROSE.get_class(): ReducedSetHindmarshRoseH5,
        ModelsEnum.WILSON_COWAN.get_class(): WilsonCowanH5,
        ModelsEnum.REDUCED_WONG_WANG.get_class(): ReducedWongWangH5,
        ModelsEnum.REDUCED_WONG_WANG_EXCH_INH.get_class(): ReducedWongWangExcInhH5,
        ModelsEnum.ZERLAUT_FIRST_ORDER.get_class(): ZerlautAdaptationFirstOrderH5,
        ModelsEnum.ZERLAUT_SECOND_ORDER.get_class(): ZerlautAdaptationSecondOrderH5,
        ModelsEnum.SUP_HOPF.get_class(): SupHopfH5
    }

    return model_class_to_h5.get(model_class)


def monitor_h5_factory(model_class):
    from tvb.simulator.monitors import Raw, SpatialAverage, EEG, MEG, iEEG, Bold, BoldRegionROI
    from tvb.core.entities.file.simulator.monitor_h5 import RawH5, SpatialAverageH5, EEGH5, MEGH5, iEEGH5, BoldH5, \
        BoldRegionROIH5, SubSampleH5, GlobalAverageH5, TemporalAverageH5

    monitor_class_to_h5 = {
        Raw: RawH5,
        SubSample: SubSampleH5,
        SpatialAverage: SpatialAverageH5,
        GlobalAverage: GlobalAverageH5,
        TemporalAverage: TemporalAverageH5,
        EEG: EEGH5,
        MEG: MEGH5,
        iEEG: iEEGH5,
        Bold: BoldH5,
        BoldRegionROI: BoldRegionROIH5
    }

    return monitor_class_to_h5.get(model_class)
