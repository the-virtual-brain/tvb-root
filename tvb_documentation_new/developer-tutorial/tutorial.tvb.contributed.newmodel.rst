tutorial.tvb.contributed.newmodel
---------------------------------

General guidelines for implementing a new Model in the |TVB| framework

External dependencies
---------------------
::

    tvb 
      \-core 
      | \-logger 
      | \-traits 
      |   \-basic 
      |     \-Integer 
      |     \-Range 
      \-datatypes 
      | \-arrays 
      |   \-FloatArray 
      \-simulator 
        \-models
::

You can use third party libraries namely Numpy and Scipy. These should cover all
the mathematical needs.  
 
The Model Class:
    
    In order to implement a model in the |TVB| framework we strongly encourage 
    modularity. The model should be easily separated into functional elements 
    that we shall discuss next. These elements will fit into the Model class 
	methods.    
    The new model class DocString *must contain the main reference* where the 
	model was taken from (published article(s) / books). In addition to that a 
	brief description of the what the model represents and what its main 
	application is will be appreciated.

Datatyped and Traited parameters:
	All constant parameters representing physical magnitudes or scaling factors 
	should have:

    - label:  the mathematical symbol used in the equations
    - default value: the most significative value used in simulations
    - range: minimum and maximum values for which the model is valid and/or functional
    - doc: more detailed information about what the parameter represents, 
            what it affects, and possible effects of changing its values. Include
            units if it applies.
 
	As a way to ease parameter identification, consistency between parameter
	names and mathematical symbols used in the equations is required.
    	
	All the aforementioned elements should be taken from the main reference. This
    is useful to create a list of symbols if needed.
	
Init:
    The traited and datatyped parameters as well as the state variables are 
    initialized. Within this method you should define the number and the 
    corresponding names of the state variables.

Initial Conditions:


Dynamical Equations(dfun):
	The dynamical equations are one of the pillars of the large-scale network 
	dynamics. 
	The equations describing the behaviour of the state variables must be clear, 
	i.e., the state variables should be unambigously identified, documented with 
	the corresponding equations and referenced. 

	In addition to that the internal, local and global coupling coefficients 
	should be clearly identified in the set of equations describing the model.
    For example, given a two state variable model, and its dynamical equations: 
    .. math::
        \dot(x) = - a * x  + y * c + g_c + l_c + \lambda
        \dot(y) = - b * y  + x * d + g_c + l_c + \lambda
    
    where :math:`g_{c}` is the global coupling term representing how the local network is
    coupled with the external world, :math:`l_{c}` is the local coupling coefficient used in
    region based simulations; and finally :math:`\lambda` represents any input stimulation.

    Note that this is a toy example. The external coupling coefficient are 
    represented as additive terms only for simplicity reasons. 

    
 

Illustration
-------------

Include a graphical representation of the model whenever it is possible, where all the main parameters such as state variables, couplings and external stimulation are represented.

.. image:: images/network.pdf
   :width: 90%
   :align: center

Use the file template
-----------------------
As a first example see, tvb_newmodel_template.py
For more information on coding conventions read tvb/simulator/doc/tvb.simulator.devguide.rst

   
