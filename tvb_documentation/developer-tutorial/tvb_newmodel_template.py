# -*- coding: utf-8 -*-

"""
My novel model.

.. moduleauthor:: Calvin Hobbs <calvinhobbs@invalid>

"""

# Third party python libraries
import numpy

#The Virtual Brain
try:
    from tvb.basic.logger.builder import get_logger
    LOG = get_logger(__name__)
except ImportError:
    import logging
    LOG = logging.getLogger(__name__)
    

#import datatypes and traited datatypes
from tvb.datatypes.arrays import FloatArray 
from tvb.basic.traits.types_basic import  Range, Integer, Dict
import tvb.simulator.models as models


class MyNewModelNameHere(models.Model):
    """
    .. [REF_2012] Name A and Then B. * Main Reference Title*. 
        Journal of Wonderful Research Vol, start.page-end.page, YYYY.   
     
    .. [REF2_2012] Idem 
   
    See also, http://www.scholarpedia.org/My_Model_is_in_scholarpedia
    
    .. automethod:: __init__
    
    """
    
    _ui_name = "My model's name that will be displayed in the web user interface(My Model Name)"
    #Define traited attributes for this model
    
    par_1 = FloatArray(
        label = ":math:`\\par_{1}`",
        default = numpy.array(["Insert default value as float number. Ex: 0.5",]),
        range = Range(lo = "Insert minimum_value here. Ex: 0.0", hi = "Insert maximum_value here. Ex: 1.0)"), 
        doc = """ Description of the physical magnitude or the effect of this parameter [units] """)
        
    par_2 = Integer(
        label = ":math:`\\par_{2}`",
        default = numpy.array(["default value as integer number. Ex: 5",]),
        range = Range(lo = "Insert minimum_value. Ex. 0", hi = "Insert maximum_value. Ex: 10"), 
        doc = """ Description of the physical magnitude or the effect of this parameter [units] """)
        
    # Define the range of possible values that the state variables can have. 
    # This will be used to set the initial conditions
    state_variable_range = Dict(
        label = "State Variable ranges [lo, hi]",
        default = {"my_first_state_variable_name": numpy.array(["low_value_first_variable", "high_value_first_variable"]),
                   "my_second_state_variable_name": numpy.array(["low_value_second_variable", "high_value_second_variable"])},
        doc = """The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current 
            parameters, it is used as a mechanism for bounding random inital 
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")
    
    # If the MyNewModel does not have more than 1 mode, then leave this piece of code as it is    
    number_of_modes = Integer(
        label = "Number of modes",
        default = 1,
        doc = """ Number of modes of the model """)
    
        
        
    def __init__(self, **kwargs):
        """
        Initialise parameters
        
        """
        
        super(MyNewModelNameHere, self).__init__(**kwargs)
        
        #state variables names        
        self._state_variables = ["my_first_state_variable_name", "my_second_state_variable_name"]
        
        # number of state variables
        self._nvar = 2
        self.cvar = numpy.array([0], dtype=numpy.int32)
        
        # the variable of interest
        self.voi = numpy.array([0], dtype=numpy.int32)
        
        #If there are derived parameters from the predefined parameters, then initialize them to None
        self.A = None
        self.B = None
    
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """
        The equations were taken from [REF_2012]
        cf. Eqns. (00) and (01), page 2
        
        .. math::
            \\dot{my_first_state_variable} &= 
            \\dot{my_second_state_variable} &=
               
        """
        
        my_first_state_variable_name = state_variables[0, :]
        my_second_state_variable_name = state_variables[1, :]
    
        # global coupling
        my_first_global_coupling_coefficient = coupling[0, :]
                
        dmy_first_state_variable_name = "Write the first state variable differential equation"
        dmy_second_state_variable_name = "Write the second state variable differential equation"

        
        derivative = numpy.array([dmy_first_state_variable_name, dmy_second_state_variable_name])
        return derivative
        
        
        def update_derived_parameters(self):
            """
            Calculate coefficients for the neural field model based on [REF_2012].
            cf. Eqns (X.X) and (Y.Y), page aaa
           
            Include equations here
            
            .. math::
                A &= equation
                B &= equation
                
            """
        
            self.A = "Insert equation here"
            self.B = "Insert equation here"
   
        
if __name__ == "__main__":
    # Do some stuff that tests or makes use of this module...
    LOG.info("Testing %s module..." % __file__)
    
    # Check that the docstring examples, if there are any, are accurate.
    import doctest
    doctest.testmod()
    
    #Initialise Model in their default state:
    model = MyNewModelNameHere()
    LOG.info("MyNewModelNameHere initialised in its default state without error...")


    