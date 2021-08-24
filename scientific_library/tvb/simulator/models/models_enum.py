from tvb.basic.neotraits.api import HasTraitsEnum
from tvb.simulator.models.epileptor import Epileptor, Epileptor2D
from tvb.simulator.models.epileptor_rs import EpileptorRestingState
from tvb.simulator.models.epileptorcodim3 import EpileptorCodim3, EpileptorCodim3SlowMod
from tvb.simulator.models.hopfield import Hopfield
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin, CoombesByrne, CoombesByrne2D, GastSchmidtKnosche_SD, \
    GastSchmidtKnosche_SF, DumontGutkin
from tvb.simulator.models.jansen_rit import JansenRit, ZetterbergJansen
from tvb.simulator.models.larter_breakspear import LarterBreakspear
from tvb.simulator.models.linear import Linear
from tvb.simulator.models.oscillator import Generic2dOscillator, Kuramoto, SupHopf
from tvb.simulator.models.stefanescu_jirsa import ReducedSetFitzHughNagumo, ReducedSetHindmarshRose
from tvb.simulator.models.wilson_cowan import WilsonCowan
from tvb.simulator.models.wong_wang import ReducedWongWang
from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh
from tvb.simulator.models.zerlaut import ZerlautAdaptationFirstOrder, ZerlautAdaptationSecondOrder


class ModelsEnum(HasTraitsEnum):
    GENERIC_2D_OSCILLATOR = (Generic2dOscillator, "Generic 2D Oscillator")
    KURAMOTO = (Kuramoto, "Kuramoto Oscillator")
    SUP_HOPF = (SupHopf, "Suphopf")
    HOPFIELD = (Hopfield, "Hopfield")
    EPILEPTOR = (Epileptor, "Epileptor")
    EPILEPTOR_2D = (Epileptor2D, "Epileptor2D")
    EPILEPTOR_CODIM_3 = (EpileptorCodim3, "Epileptor Codim 3")
    EPILEPTOR_CODIM_3_SLOW = (EpileptorCodim3SlowMod, "Epileptor Codim 3 Ultra-Slow Modulations")
    EPILEPTOR_RS = (EpileptorRestingState, "Epileptor Resting State")
    JANSEN_RIT = (JansenRit, "Jansen-Rit")
    ZETTERBERG_JANSEN = (ZetterbergJansen, "Zetterberg-Jansen")
    REDUCED_WONG_WANG = (ReducedWongWang, "Reduced Wong-Wang")
    REDUCED_WONG_WANG_EXCH_INH = (ReducedWongWangExcInh, "Reduced Wong-Wang With Excitatory And Inhibitory Coupled Populations")
    REDUCED_SET_FITZ_HUGH_NAGUMO = (ReducedSetFitzHughNagumo, "Stefanescu-Jirsa 2D")
    REDUCED_SET_HINDMARSH_ROSE = (ReducedSetHindmarshRose, "Stefanescu-Jirsa 3D")
    ZERLAUT_FIRST_ORDER = (ZerlautAdaptationFirstOrder, "Zerlaut Adaptation First Order")
    ZERLAUT_SECOND_ORDER = (ZerlautAdaptationSecondOrder, "Zerlaut Adaptation Second Order")
    MONTBRIO_PAZO_ROXIN = (MontbrioPazoRoxin, "Montbrio Pazo Roxin")
    COOMBES_BYRNE = (CoombesByrne, "Coombes Byrne")
    COOMBES_BYRNE_2D = (CoombesByrne2D, "Coombes Byrne 2D")
    GAST_SCHMIDT_KNOSCHE_SD = (GastSchmidtKnosche_SD, "Gast Schmidt Knosche_Sd")
    GAST_SCHMIDT_KNOSCHE_SF = (GastSchmidtKnosche_SF, "Gast Schmidt Knosche_Sf")
    DUMONT_GUTKIN = (DumontGutkin, "Dumont Gutkin")
    LINEAR = (Linear, "Linear Model")
    WILSON_COWAN = (WilsonCowan, "Wilson-Cowan")
    LARTER_BREAKSPEAR = (LarterBreakspear, "Larter-Breakspear")

    @staticmethod
    def get_base_model_subclasses():
        return [model.value for model in list(ModelsEnum)]
