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

The Equation datatypes. This brings together the scientific and framework 
methods that are associated with the Equation datatypes.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import tvb.datatypes.equations_scientific as equations_scientific
import tvb.datatypes.equations_framework as equations_framework
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)



class Equation(equations_scientific.EquationScientific, equations_framework.EquationFramework):
    """
    This class brings together the scientific and framework methods that are
    associated with the Equation dataTypes.
    
    ::
        
                           EquationData
                                 |
                                / \\
               EquationFramework   EquationScientific
                                \ /
                                 |
                              Equation
        
    
    """
    pass



class TemporalApplicableEquation(Equation):
    """
    Abstract class introduced just for filtering what equations to be displayed in UI,
    for setting the temporal component in Stimulus on region and surface.
    """
    pass



class FiniteSupportEquation(TemporalApplicableEquation):
    """
    Equations that decay to zero as the variable moves away from zero. It is
    necessary to restrict spatial equation evaluated on a surface to this
    class, are . The main purpose of this class is to facilitate filtering in the UI,
    for patters on surface (stimuli surface and localConnectivity).
    """
    pass



class SpatialApplicableEquation(Equation):
    """
    Abstract class introduced just for filtering what equations to be displayed in UI,
    for setting model parameters on the Surface level.
    """
    pass



class DiscreteEquation(equations_scientific.DiscreteEquationScientific,
                       equations_framework.DiscreteEquationFramework, FiniteSupportEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Discrete datatypes.
    
    ::
        
                           DiscreteData
                                 |
                                / \\
               DiscreteFramework   DiscreteScientific
                                \ /
                                 |
                              Discrete
        
    
    """
    pass



class Linear(equations_scientific.LinearScientific, equations_framework.LinearFramework, TemporalApplicableEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Linear datatypes.
    
    ::
        
                             LinearData
                                 |
                                / \\
                 LinearFramework   LinearScientific
                                \ /
                                 |
                               Linear
        
    
    """
    pass



class Gaussian(equations_scientific.GaussianScientific,
               equations_framework.GaussianFramework, SpatialApplicableEquation, FiniteSupportEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Gaussian datatypes.
    
    ::
        
                           GaussianData
                                 |
                                / \\
               GaussianFramework   GaussianScientific
                                \ /
                                 |
                              Gaussian
        
    
    """
    pass



class DoubleGaussian(equations_scientific.DoubleGaussianScientific,
                     equations_framework.DoubleGaussianFramework, FiniteSupportEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the DoubleGaussian datatypes.
    
    ::
        
                         DoubleGaussianData
                                 |
                                / \\
         DoubleGaussianFramework   DoubleGaussianScientific
                                \ /
                                 |
                            DoubleGaussian
        
    
    """
    pass



class Sigmoid(equations_scientific.SigmoidScientific,
              equations_framework.SigmoidFramework, SpatialApplicableEquation, FiniteSupportEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Sigmoid datatypes.
    
    ::
        
                            SigmoidData
                                 |
                                / \\
                SigmoidFramework   SigmoidScientific
                                \ /
                                 |
                              Sigmoid
        
    
    """
    pass



class GeneralizedSigmoid(equations_scientific.GeneralizedSigmoidScientific,
                         equations_framework.GeneralizedSigmoidFramework, TemporalApplicableEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Generalized Sigmoid datatypes.
    
    ::
        
                            GeneralizedSigmoidData
                                     |
                                    / \\
         GeneralizedSigmoidFramework   GeneralizedSigmoidScientific
                                    \ /
                                     |
                            GeneralizedSigmoid
        
    
    """
    pass



class Sinusoid(equations_scientific.SinusoidScientific,
               equations_framework.SinusoidFramework, TemporalApplicableEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Sinusoid datatypes.
    
    ::
        
                           SinusoidData
                                 |
                                / \\
               SinusoidFramework   SinusoidScientific
                                \ /
                                 |
                              Sinusoid
        
    
    """
    pass



class Cosine(equations_scientific.CosineScientific,
             equations_framework.CosineFramework, TemporalApplicableEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Sinusoid datatypes.
    
    ::
        
                           CosineData
                                 |
                                / \\
                 CosineFramework   CosineScientific
                                \ /
                                 |
                              Cosine
        
    
    """
    pass



class Alpha(equations_scientific.AlphaScientific,
            equations_framework.AlphaFramework, TemporalApplicableEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Alpha datatypes.
    
    ::
        
                             AlphaData
                                 |
                                / \\
                  AlphaFramework   AlphaScientific
                                \ /
                                 |
                               Alpha
        
    
    """
    pass



class PulseTrain(equations_scientific.PulseTrainScientific,
                 equations_framework.PulseTrainFramework, TemporalApplicableEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the PulseTrain datatypes.
    
    ::
        
                            PulseTrainData
                                 |
                                / \\
            PulseTrainFramework    PulseScientific
                                \ /
                                 |
                               Pulsetrain
        
    
    """
    pass




class HRFKernelEquation(Equation):
    """
    Abstract class introduced just for filtering what equations to be displayed in UI, for BOLD Monitor.
    """
    pass



class Gamma(equations_scientific.GammaScientific, equations_framework.GammaFramework, HRFKernelEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Gamma datatypes.
    
    ::
        
                             GammaData
                                 |
                                / \\
                  GammaFramework   GammaScientific
                                \ /
                                 |
                               Gamma
        
    
    """
    pass



class DoubleExponential(equations_scientific.DoubleExponentialScientific,
                        equations_framework.DoubleExponentialFramework, HRFKernelEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the DoubleExponential datatypes.
    
    ::
        
                             DoubleExponentialData
                                 |
                                / \\
      DoubleExponentialFramework   DoubleExponentialScientific
                                \ /
                                 |
                             DoubleExponential
        
    
    """
    pass



class FirstOrderVolterra(equations_scientific.FirstOrderVolterraScientific,
                         equations_framework.FirstOrderVolterraFramework, HRFKernelEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the DoubleExponential datatypes.
    
    ::
        
                             FirstOrderVolterraData
                                 |
                                / \\
      FirstOrderVolterraFramework   FirstOrderVolterraScientific
                                \ /
                                 |
                             FirstOrderVolterra
        
    
    """
    pass



class MixtureOfGammas(equations_scientific.MixtureOfGammasScientific, equations_framework.MixtureOfGammasFramework, HRFKernelEquation):
    """
    This class brings together the scientific and framework methods that are
    associated with the Gamma datatypes.
    
    ::
        
                        MixtureOfGammasData
                                 |
                                / \\
        MixtureOfGammasFramework   MixtureOfGammasScientific
                                \ /
                                 |
                          MixtureOfGammas
        
    
    """
    pass

