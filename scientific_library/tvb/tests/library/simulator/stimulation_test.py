"""
Tests stimulation when sub-cortical regions are present (in the connectivity matrix).

.. moduleauthor:: Borana Dollomaja <borana.dollomaja@univ-amu.fr>
"""
import pytest
import numpy
import time
# import TVB modules
from tvb.simulator.lab import *
from tvb.basic.neotraits.api import List
from tvb.tests.library.base_testcase import BaseTestCase

class PulseStimulation(object):
    """
    Stimulus test class
    """
    def __init__(self, con):
        # Configure stimulus spatial pattern.
        stim_node = [0] # target zone where we want to stimulate
        stim_weights = numpy.zeros((con.number_of_regions, ))
        stim_weights[stim_node] = numpy.array([2.0])

        # Configure stimulus temporal profile.
        eqn_t = equations.PulseTrain()
        eqn_t.parameters['onset'] = 2e3       # onset time [ms]
        eqn_t.parameters['T'] = 3000.0        # pulse repetition period [ms]
        eqn_t.parameters['tau'] = 100.0       # pulse duration [ms]

        # Configure Stimuli object.
        self.stimulus = patterns.StimuliRegion(temporal = eqn_t,
                                        connectivity = con, 
                                        weight =stim_weights)

class SetUpSimulator(object):
    """
    Stimulation test class
    """
    def __init__(self):
        """
        Initialize the structural information, coupling function, integrator, monitors, surface and stimulation.
        """
        # Connectome
        con = connectivity.Connectivity.from_file("connectivity_192.zip")
        con.configure()

        # Surface and local connectivity kernel
        surf = cortex.Cortex.from_file() #Initialise a surface
        surf.local_connectivity = local_connectivity.LocalConnectivity.from_file()
        surf.configure()

        # Model
        oscilator = models.Generic2dOscillator()

        self.sim = simulator.Simulator(
            conduction_speed=1.0,
            coupling= coupling.Difference(a=numpy.array([0.01])),
            surface=surf,
            stimulus=PulseStimulation(con).stimulus,
            integrator=integrators.Identity(dt=1.0),
            simulation_length=10.0,
            connectivity=con,
            model=oscilator,
            monitors=(monitors.Raw(),)
        )
        self.sim.configure()

    def run_simulation(self):
        return self.sim.run(simulation_length = 10)

class TestStimulation(BaseTestCase):
    def test_stimulation(self):
        model = SetUpSimulator()
        result = model.run_simulation()
        assert result is not None