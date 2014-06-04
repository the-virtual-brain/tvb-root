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
Demonstrate using the simulator at the region level, stochastic
integration and the BOLD monitor:   

- how to perform a simulation continuation, that is, to use data from one
simulation as initial conditions for a second simulation (ie, simulation
continuation) in order to avoid the temporal transient due to imperfect
initial conditions.

- how to save the random number generator state. 

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

from tvb.simulator.lab import *

######## build & run the initial simulation, if cryo file doesn't exist
try:
    os.stat('sim_cryo.npz')
    print 'old simulation found! moving along...'

except OSError:     # os.stat will give OSError if it doesn't find the file
    
    print 'old simulation not found, performing initial'

    oscillator = models.Generic2dOscillator()
    white_matter = connectivity.Connectivity(load_default=True)
    white_matter.speed = numpy.array([4.0])
    white_matter_coupling = coupling.Linear(a=0.0126)

    heunint = integrators.HeunStochastic(dt=2 ** -4,
                                         noise=noise.Additive(nsig=array([0.001])))

    #Initialise some Monitors with period in physical time
    tavg = monitors.TemporalAverage(period=1.0)     # 1000Hz
    bold = monitors.Bold(period=500)    # defaults to one data point every 500ms

    #Bundle them
    what_to_watch = (tavg, bold)

    #Define the stimulus
    #Specify a weighting for regions to receive stimuli... 
    white_matter.configure()    # Because we want access to number_of_regions
    nodes = [0, 7, 13, 33, 42]
    weighting = numpy.zeros((white_matter.number_of_regions, ))
    weighting[nodes] = numpy.array([2.0 ** -2, 2.0 ** -3, 2.0 ** -4, 2.0 ** -5, 2.0 ** -6])

    #Initialise Simulator -- Model, Connectivity, Integrator, Monitors, and stimulus.
    sim = simulator.Simulator(model=oscillator, connectivity=white_matter,
                              coupling=white_matter_coupling,
                              integrator=heunint, monitors=what_to_watch)
    sim.configure()

    #Perform the simulation
    tavg_time = []
    tavg_data = []
    bold_time = []
    bold_data = []

    for tavg_out, bold_out in sim(10000):    # 10 s simulation
        
        if not tavg_out is None:
            tavg_time.append(tavg_out[0])
            tavg_data.append(tavg_out[1])
        
        if not bold_out is None:
            bold_time.append(bold_out[0])
            bold_data.append(bold_out[1])


    # plot data to show the transient
    figure(1)
    plot(array(bold_time) / 1000.0, array(bold_data)[:, 0, :, 0])
    title('initial simulation')
    xlabel('time (s)')

    # now, save the state: sim history, bold stock & noise generator state
    history = sim.history.copy()
    bold1 = sim.monitors[1]._interim_stock.copy()
    bold2 = sim.monitors[1]._stock.copy()
    rng = sim.integrator.noise.random_stream.get_state()

    # save to file (sorry this is cryptic for the moment)
    savez('sim_cryo.npz', history=history, bold1=bold1, bold2=bold2)
    save('rng_%s_%d_%d_%.30g.npy' % ((rng[0],) + rng[2:]), rng[1])


############################### 
############################### 
############################### 


# build new simulator, possibly in a different script

oscillator = models.Generic2dOscillator()
white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = numpy.array([4.0])
white_matter_coupling = coupling.Linear(a=0.0126)

heunint = integrators.HeunStochastic(dt=2 ** -4,
                                     noise=noise.Additive(nsig=array([0.001])))

#Initialise some Monitors with period in physical time
tavg = monitors.TemporalAverage(period=1.0)     # 1000Hz
bold = monitors.Bold(period=500)        # defaults to one data point every 500ms

#Bundle them
what_to_watch = (tavg, bold)

#Define the stimulus
#Specify a weighting for regions to receive stimuli... 
white_matter.configure()    # Because we want access to number_of_regions
nodes = [0, 7, 13, 33, 42]
weighting = numpy.zeros((white_matter.number_of_regions, ))
weighting[nodes] = numpy.array([2.0 ** -2, 2.0 ** -3, 2.0 ** -4, 2.0 ** -5, 2.0 ** -6])

#Initialise Simulator -- Model, Connectivity, Integrator, Monitors, and stimulus.
sim = simulator.Simulator(model=oscillator, connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint, monitors=what_to_watch)
sim.configure()


### NOW, we load from files the brain state
cryo = load('sim_cryo.npz')

# find the first rng file (i.e. delete the old ones!), and rebuild random state
rng_name = glob.glob("rng_*.npy")[0]
rng = [f(a) for a, f in zip(rng_name[:-4].split('_')[1:], [str, int, int, float])]
rng_state = [rng[0], load(rng_name)] + rng[1:]

# set the new simulator with the old brain state

sim.history[:] = cryo['history']
sim.monitors[1]._stock[:] = cryo['bold2']
sim.monitors[1]._interim_stock[:] = cryo['bold1']

sim.integrator.noise.random_stream.set_state(rng_state)

# now we can "continue" the new simulation
tavg_time = []
tavg_data = []
bold_time = []
bold_data = []

for tavg_out, bold_out in sim(10000, rng_state):

    if not tavg_out is None:
        tavg_time.append(tavg_out[0])
        tavg_data.append(tavg_out[1])
    
    if not bold_out is None:
        bold_time.append(bold_out[0])
        bold_data.append(bold_out[1])


# then, plot to see if there is transient or not
figure(2)
plot(array(bold_time) / 1000.0, array(bold_data)[:, 0, :, 0])
title('continued simulation')
xlabel('time (s)')
###EoF###