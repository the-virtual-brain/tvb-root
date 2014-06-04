#-*- coding: utf-8 -*-

"""
Make use of the parameter package: 
https://parameters.readthedocs.org to keep
a collection of interesting simulation parameters.

The default parameters of a model won't change very often. However there are
different combinations of model and simulation parameters that we wish to keep
track off. For instance, when trying to reproduce previous work.  These
collections of parameters should be apart of a model or the demo itself.  At
the least, parameters should be defined in a separate section at the start of
a file.

It will also (hopefully) provide a means to easily retrieve sets of parameters from the UI. 

TODO: Consider creating a Parameter factory. 
Add more levels, since sometimes the parameters 
should change defaults such as noise dispersion or conduction speed. 


NOTE: Kind of convention: the set of model parameters should be labeled with
the dynamics they describe for an isolated node; they are further identified
by their *key* in the next hierarchical set, linking the work they were taken
from. In this way people can decide whether to identifiy a set of prameters by
published work or by another name like "My_Chaotic_Mess_test_1", however the
lowest set should have a meaningful scientific description. This parameter
collection / factory should replace the tables in the models docstrings.


In a demo this will be used like this:

    import tvb.simulator.parameter_factory as dynamics
    these_dynamics = dynamics.generic_2d_oscillator_pars(these_dynamics="Reproduce_Knock_2009")
    oscillator = models.Generic2dOscillator(**these_dynamics.model_params.as_dict())

.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>
"""



from tvb.basic.logger.builder import get_logger
LOG = get_logger(__name__)

try:
	from parameters import ParameterSet
except ImportError:
    IMPORTED_PARAMETERS = False
    LOG.error("You need the parameters module")



def generic_2d_oscillator_pars(these_dynamics="Reproduce_Knock_2009"):
	""" 
	input:
	------

	      these_dynamics      : the key of the parameter configuration to retrieve.
	output:
	------ 

	       these_parameters   : a parameter set with a hierarchical structure

	"""


	# Set some defaults and overwrite if required
	sim_params    = ParameterSet({'dt': 2**-6, 'simulation_length': 2**10})

	knock_pars    = ParameterSet({'a': 1.05, 'b': -1., 'c':0.0, 'd':0.1, 'e':0.0, 'f':1/3., 
	                              'g':1.0, 'alpha':1.0, 'beta':0.2, 'tau':1.25, 'gamma':-1.0}, 
	                               label='fixed_point')


	sanzleon_pars = ParameterSet({'a': -0.5, 'b': -10., 'c':0.0, 'd':0.02, 'e':3.0, 'f':1., 
		                          'g':0.0, 'alpha':1.0, 'beta':1.0, 'tau':1., 'gamma':1.0}, 
		                           label='fixed_point')


	dynamics_collection = ParameterSet({'Reproduce_Knock_2009': knock_pars, 
		                                'Reproduce_SanzLeon_2013': sanzleon_pars})

	
	these_parameters    = ParameterSet({'sim_params': sim_params, 
		                                'model_params': dynamics_collection[these_dynamics]}, label=these_dynamics)

	return these_parameters






