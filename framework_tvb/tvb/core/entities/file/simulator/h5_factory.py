from tvb.datatypes.cortex import Cortex
from tvb.simulator.coupling import Coupling
from tvb.simulator.integrators import Integrator, HeunDeterministic, HeunStochastic, \
    EulerDeterministic, EulerStochastic, RungeKutta4thOrderDeterministic, Identity, VODE, VODEStochastic, Dopri5, \
    Dopri5Stochastic, Dop853, Dop853Stochastic
from tvb.simulator.models import Model
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
    if issubclass(config_class, Model):
        return model_h5_factory(config_class)
    if issubclass(config_class, Monitor):
        return monitor_h5_factory(config_class)
    if config_class == Cortex:
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
    from tvb.simulator.models import Epileptor, Epileptor2D, EpileptorCodim3, EpileptorCodim3SlowMod, Hopfield, \
        JansenRit, ZetterbergJansen, EpileptorRestingState, LarterBreakspear, Generic2dOscillator, \
        ReducedSetFitzHughNagumo, ReducedSetHindmarshRose, WilsonCowan, ReducedWongWang, ReducedWongWangExcInh, \
        ZerlautFirstOrder, ZerlautSecondOrder, SupHopf, Linear, Kuramoto
    from tvb.core.entities.file.simulator.model_h5 import EpileptorH5, Epileptor2DH5, EpileptorCodim3H5, \
        EpileptorCodim3SlowModH5, HopfieldH5, JansenRitH5, ZetterbergJansenH5, EpileptorRestingStateH5, \
        LarterBreakspearH5, LinearH5, Generic2dOscillatorH5, KuramotoH5, ReducedSetFitzHughNagumoH5, \
        ReducedSetHindmarshRoseH5, WilsonCowanH5, ReducedWongWangH5, ReducedWongWangExcInhH5, ZerlautFirstOrderH5, \
        ZerlautSecondOrderH5, SupHopfH5
    model_class_to_h5 = {
        Epileptor: EpileptorH5,
        Epileptor2D: Epileptor2DH5,
        EpileptorCodim3: EpileptorCodim3H5,
        EpileptorCodim3SlowMod: EpileptorCodim3SlowModH5,
        Hopfield: HopfieldH5,
        JansenRit: JansenRitH5,
        ZetterbergJansen: ZetterbergJansenH5,
        EpileptorRestingState: EpileptorRestingStateH5,
        LarterBreakspear: LarterBreakspearH5,
        Linear: LinearH5,
        Generic2dOscillator: Generic2dOscillatorH5,
        Kuramoto: KuramotoH5,
        ReducedSetFitzHughNagumo: ReducedSetFitzHughNagumoH5,
        ReducedSetHindmarshRose: ReducedSetHindmarshRoseH5,
        WilsonCowan: WilsonCowanH5,
        ReducedWongWang: ReducedWongWangH5,
        ReducedWongWangExcInh: ReducedWongWangExcInhH5,
        ZerlautFirstOrder: ZerlautFirstOrderH5,
        ZerlautSecondOrder: ZerlautSecondOrderH5,
        SupHopf: SupHopfH5
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
