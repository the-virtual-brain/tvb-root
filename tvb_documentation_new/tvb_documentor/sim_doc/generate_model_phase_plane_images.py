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
Generate a full set of phase-plane images for each of the local dynamic models
in the module tvb.simulator.models, for their default parameters.

Usage::
    
    #From an ipython prompt in the docs directory, with TVB in your python path
    run generate_model_phase_plane_images.py

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

#TODO: Clean this up for its new purpose, it's just hacked from an old tester...

# Third party python libraries
import numpy
import matplotlib.pyplot as pyplot

# From "The Virtual Brain"
try:
    from tvb.basic.logger.builder import get_logger
    LOG = get_logger(__name__)
except ImportError:
    import logging
    LOG = logging.getLogger(__name__)
    LOG.warning("Failed to import tvb.basic.logger.builder, falling back to logging")


from tvb.datatypes.arrays import FloatArray, IntegerArray
from tvb.basic.traits.core import Type
from tvb.basic.traits.types_basic import Integer

import tvb.simulator.models as models
import tvb.simulator.integrators as integrators

#NOTE: png files are ~5-10 times smaller than svg, but lower quality on zoom. 
IMG_SUFFIX = ".svg" #".png"

class TestModel(Type):
    """
    A class to enable phase-plane based checing of local model dynamics...
    
    Basic usage is::
        import tvb.simulator.model_tester as model_tester
        import tvb.simulator.models as models
        
        tester = model_tester.test_factory(models.Generic2dOscillator)
        
        tester.pplane(xlo=-4.0, xhi=4.0, ylo=-8.0, yhi=8.0)
        
    """
    
    model = models.Model
    
    phase_plane = IntegerArray(label = "state variables for the phase plane",
                              default = None)
                              
    xlo = FloatArray(label = "left edge the phase plane",
                     default = None) #-60.0
    xhi = FloatArray(label = "right edge of the phase plane",
                     default = None) #50.0
    ylo = FloatArray(label = "bottom edge of the phase plane",
                     default = None) #-0.2
    yhi = FloatArray(label = "top edge of the phase plane",
                     default = None) #0.8
    npts = Integer(label = "discretisation of the phase plane",
                   default = 42)
    int_steps = Integer(label = "integration steps for sample trajectory",
                        default = 2048)
    
    def __init__(self, **kwargs):
        """Use BaseModel passed in through test_factory()"""
        super(TestModel, self).__init__(**kwargs)
        
    def configure(self):
        super(TestModel, self).configure()
        
        self.model.configure()
        
        if self.phase_plane is None:
            self.phase_plane = self.all_state_variable_pairs()
    
    
    def __str__(self):
        """Use BaseModel passed in through test_factory()"""
        return self.model.__str__()

        
    def __repr__(self):
        """Use BaseModel passed in through test_factory()"""
        return self.model.__repr__()
        
        
    def initial(self, history_shape):
        """Use BaseModel passed in through test_factory()"""
        return self.model.initial(1.0, history_shape)

        
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        """Use BaseModel passed in through test_factory()"""
        return self.model.dfun(state_variables, coupling, local_coupling=local_coupling)


    def pplane(self, **kwargs):
        """
        Generate and plot a phase-plane diagram. Initial state of all the state
        variables, other than the two specified for the phase-plane, are 
        randomly set once for the entire phase-plane based on the initial() 
        method of the chosen model. In other words, running multiple times will
        give slightly different results, but they will all lie within the range
        of default initial conditions...
        
        **kwargs**:
            ``xlo``
            
            ``xhi``
            
            ``ylo``
            
            ``yhi``
            
            ``npts``
            
            ``phase-plane``
        
        """
        
        for key in kwargs:
            setattr(self, key, kwargs[key])
        #import pdb; pdb.set_trace()
        if self.xlo is None:  
            self.xlo = numpy.array([self.model.state_variable_range[self.model.state_variables[indx]][0] for indx in self.phase_plane[0]]) #self.model.state_variable_range[0][self.phase_plane[0]]
        
        if self.xhi is None:
            self.xhi = numpy.array([self.model.state_variable_range[self.model.state_variables[indx]][1] for indx in self.phase_plane[0]]) #self.model.state_variable_range[1][self.phase_plane[0]]
        
        if self.ylo is None:
            self.ylo = numpy.array([self.model.state_variable_range[self.model.state_variables[indx]][0] for indx in self.phase_plane[1]]) #self.model.state_variable_range[0][self.phase_plane[1]]
        
        if self.yhi is None:
            self.yhi = numpy.array([self.model.state_variable_range[self.model.state_variables[indx]][1] for indx in self.phase_plane[1]]) #self.model.state_variable_range[1][self.phase_plane[1]]
        
        init_cond = self.initial(history_shape=(1, self.model.nvar, 1, self.model.number_of_modes))
        init_cond = init_cond.reshape((self.model.nvar, 1, self.model.number_of_modes))
        no_coupling = numpy.zeros(init_cond.shape)
        #import pdb; pdb.set_trace()
        
        #Calculate an example trajectory
        state = init_cond.copy() 
        rk4 = integrators.RungeKutta4thOrderDeterministic(dt=2**-5)
        traj = numpy.zeros((self.int_steps, self.model.nvar, 1, self.model.number_of_modes))
        for step in range(self.int_steps):
            state = rk4.scheme(state, self.dfun)
            traj[step, :] = state

        npp = len(self.phase_plane[0])
        for hh in range(npp):
            #Calculate the vector field discretely sampled at a grid of points
            grid_point = init_cond.copy()
            X = numpy.mgrid[self.xlo[hh]:self.xhi[hh]:(self.npts*1j)]
            Y = numpy.mgrid[self.ylo[hh]:self.yhi[hh]:(self.npts*1j)]
            U = numpy.zeros((self.npts, self.npts, self.model.number_of_modes, npp))
            V = numpy.zeros((self.npts, self.npts, self.model.number_of_modes, npp))
            for ii in xrange(self.npts):
                grid_point[self.phase_plane[1, hh]] = Y[ii]
                for jj in xrange(self.npts):
                    #import pdb; pdb.set_trace()
                    grid_point[self.phase_plane[0, hh]] = X[jj]
                    
                    d = self.dfun(grid_point, no_coupling)
                    
                    for kk in range(self.model.number_of_modes):
                        U[ii, jj, kk, hh] = d[self.phase_plane[0, hh], 0, kk]
                        V[ii, jj, kk, hh] = d[self.phase_plane[1, hh], 0, kk]
        
        
            # Plot it, and save the figures to files...
            for kk in range(self.model.number_of_modes):
                
                #Create a plot window with a title
                pyplot.figure(10 * hh + kk)
                model_class_name = self.model.__class__.__name__
                pyplot.title(model_class_name + " mode " + str(kk+1))
                pyplot.xlabel("State Variable " + str(self.model.state_variables[self.phase_plane[0, hh]]))
                pyplot.ylabel("State Variable " + str(self.model.state_variables[self.phase_plane[1, hh]]))
                
                #Plot a discrete representation of the vector field
                pyplot.quiver(X, Y, U[:, :, kk, hh], V[:, :, kk, hh], 
                              width=0.0005, headwidth=8)
                
                #Plot the nullclines
                pyplot.contour(X, Y, U[:, :, kk, hh], [0], colors="r")
                pyplot.contour(X, Y, V[:, :, kk, hh], [0], colors="g")
                
                #Add an example trajectory to the on screen version of the phase-plane
                pyplot.plot(traj[:, self.phase_plane[0, hh], 0, kk], 
                            traj[:, self.phase_plane[1, hh], 0, kk])
                
                #Save this phase-plane plot to as an .png file (change to .svg when it's working...)
                pyplot.savefig("img/" + model_class_name + "_" + 
                str(self.phase_plane[0, hh]) + str(self.phase_plane[1, hh]) + 
                "_mode_" + str(kk) + "_pplane" + IMG_SUFFIX)
                
                pyplot.close("all")
                        
                        
    def all_state_variable_pairs(self):
        """
        Returns the state-variable index pairs for a complete set of phase-plane 
        diagrams... (N^2 - N) / 2 phase-planes, where N is the number of state 
        variables.
        """
        a = ()
        b = []
        for n in range(self.model.nvar):
            a = a + (self.model.nvar-n-1) * (n, )
            b = b + range(n+1, self.model.nvar)
        return numpy.array((a, b))


def test_factory(BaseModel, **kwargs):
    """
    Dynamically generate an instance of a class derived from the desired model.
    
    Args:
        BaseModel: The specific model class from which to derive the TestModel 
            class. For example:
                
                tvb.simulator.models.Generic2dOscillator
                

    Returns: An instance of TestModel, derived from the base class BaseModel

    
    """
    
    #SpecificTestModel = TestModel
    #SpecificTestModel.__bases__ = (BaseModel,)
    #tester = SpecificTestModel(**kwargs)
    tester = TestModel(model = BaseModel(**kwargs))
    tester.configure()
    tester.model.configure_initial(1.0, (1, tester.model.nvar, 1, tester.model.number_of_modes))
    return tester




if __name__ == '__main__':
    # Do some stuff that tests or makes use of this module... 
    LOG.info("Generating phase-plane images for tvb.simulator.models...")
    
    from tvb.basic.traits.parameters_factory import get_traited_subclasses
    AVAILABLE_MODELS = get_traited_subclasses(models.Model)
    ##AVAILABLE_MODELS = {models.LarterBreakspear.__name__: models.LarterBreakspear}
    MODEL_NAMES = AVAILABLE_MODELS.keys()
    
    for model_name in MODEL_NAMES:
        LOG.info("Generating phase-planes for %s" % model_name)
        tester = test_factory(AVAILABLE_MODELS[model_name])
        tester.pplane()
        

        

### EoF ###
