# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
from tvb.basic.neotraits.api import HasTraitsEnum, EnumAttr
from tvb.simulator.models.models_enum import ModelsEnum
from tvb.simulator.simulator import Simulator
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
from tvb.contrib.simulator import models


class ContribModelsEnum(HasTraitsEnum):
    # It includes the same models as the ones in tvb-library
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

    # But it also includes the models from tvb-contrib
    BRUNEL_WANG = (models.BrunelWang, "Brunel-Wang")
    HMJ_EPILEPTOR = (models.HMJEpileptor, "HMJ Epileptor")
    GENERIC_2D_OSCILLATOR_CONTRIB = (models.Generic2dOscillator, "Generic 2D Oscillator")
    HINDMARSCH_ROSE = (models.HindmarshRose, "Hindmarsch Rose")
    JANSEN_RIT_DAVID = (models.JansenRitDavid, "Jansen-Rit David")
    LARTER = (models.Larter, "Larter")
    LARTER_BREAKSPEAR_CONTRIB = (models.LarterBreakspear, "Larter-Breakspear")
    LILEY_STEYN_ROSS = (models.LileySteynRoss, "Liley Steyn Ross")
    MORRIS_LECAR = (models.MorrisLecar, "Morris Lecar")
    WONG_WANG = (models.WongWang, "Wong Wang")


class ContribSimulator(Simulator):
    model = EnumAttr(
        field_type=ModelsEnum,
        label="Local dynamic model",
        default=ModelsEnum.GENERIC_2D_OSCILLATOR.instance,
        required=True,
        doc="""A tvb.simulator.Model object which describe the local dynamic
        equations, their parameters, and, to some extent, where connectivity
        (local and long-range) enters and which state-variables the Monitors
        monitor. By default the 'Generic2dOscillator' model is used. Read the
        Scientific documentation to learn more about this model.""")
