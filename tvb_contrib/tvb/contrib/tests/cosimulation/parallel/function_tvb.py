# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications

"""
.. moduleauthor:: Lionel Kusch <lkusch@thevirtualbrain.org>
.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
"""

import numpy as np

import tvb.simulator.lab as lab
from tvb.contrib.tests.cosimulation.parallel.ReducedWongWang import \
    ReducedWongWangProxy, _numba_dfun_proxy
from tvb.contrib.cosimulation.cosim_monitors import RawCosim
from tvb.contrib.cosimulation.cosimulator import CoSimulator


def tvb_model(dt, weight, delay, id_proxy):
    """
        Initialise TVB with Wong-Wang models and default connectivity

        WARNING : in this first example most of the parameter for the simulation is define. In the future, this function
        will be disappear and replace only by the tvb_init. This function is only here in order to constraint the usage
         of proxy
    :param dt: the resolution of the raw monitor (ms)
    :param weight: weight on the connexion
    :param delay: delay on the connexion
    :param id_proxy: id of the proxy
    :return:
        populations: model in TVB
        white_matter: Connectivity in TVB
        white_matter_coupling: Coupling in TVB
        heunint: Integrator in TVB
        what_to_watch: Monitor in TVB
    """
    region_label = np.repeat(['real'], len(weight))  # name of region fixed can be modify for parameter of function
    region_label[id_proxy] = 'proxy'
    populations = ReducedWongWangProxy()
    white_matter = lab.connectivity.Connectivity(region_labels=region_label,
                                                 weights=weight,
                                                 speed=np.array(1.0),
                                                 tract_lengths=delay,
                                                 # TVB don't take care about the delay only the track length and speed
                                                 #  delai = tract_lengths/speed
                                                 centres=np.ones((weight.shape[0], 3))
                                                 )
    white_matter_coupling = lab.coupling.Linear(a=np.array(0.0154))
    heunint = lab.integrators.EulerDeterministic(dt=dt, bounded_state_variable_indices=np.array([0]),
                                                state_variable_boundaries=np.array([[0.0, 1.0]]))
    return populations, white_matter, white_matter_coupling, heunint, id_proxy


def tvb_init(parameters, time_synchronize, initial_condition):
    """
        To initialise Nest and to create the connectome model
    :param parameters : (model,connectivity,coupling,integrator) : parameter for the simulation without monitor
    :param time_synchronize : the time of synchronization for the proxy
    :param initial_condition: the initial condition of the model
    :return:
        sim : the TVB simulator,
        (weights_in,delay_in): the connectivity of disconnect region input
        (weights_out,delay_out): the connectivity of disconnect region ouput
    """
    model, connectivity, coupling, integrator, id_proxy = parameters
    # Initialise some Monitors with period in physical time
    monitors = (lab.monitors.Raw(variables_of_interest=np.array(0)),)

    # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
    if len(id_proxy) == 0:
        sim = lab.simulator.Simulator(
            model=model,
            connectivity=connectivity,
            coupling=coupling,
            integrator=integrator,
            monitors=monitors,
            initial_conditions=initial_condition
        )
    else:
        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim = CoSimulator(
                          voi=np.array([0]),
                          synchronization_time=time_synchronize,
                          cosim_monitors=(RawCosim(),),
                          proxy_inds=np.asarray(id_proxy, dtype=np.int_),
                          model=model,
                          connectivity=connectivity,
                          coupling=coupling,
                          integrator=integrator,
                          monitors=monitors,
                          initial_conditions=initial_condition
                          )
    sim.configure()
    return sim


def tvb_simulation(time, sim, data_proxy):
    """
    Simulate t->t+dt:
    :param time: the time of simulation
    :param sim: the simulator
    :param data_proxy : the firing rate of the next steps
    :return:
        the time, the firing rate and the state of the network
    """
    if isinstance(sim, CoSimulator):
        if data_proxy is not None:
            # We assume only 1 voi, 1 or 2 proxy nodes, and only 1 mode,
            # therefore data_proxy.shape -> (synchronization_n_step, n_voi=1, n_proxy, n_mode=1)
            data_proxy[1] = np.reshape(data_proxy[1],
                                       (data_proxy[1].shape[0], sim.voi.shape[0],
                                        sim.proxy_inds.shape[0], sim.model.number_of_modes))
        result_delayed = sim.run(cosim_updates=data_proxy)
        result = sim.loop_cosim_monitor_output()
        time = result[0][0]
        s = [result[0][1][:,0], result_delayed[0][1][:,0]]
        rate = [result[0][1][:,1], result_delayed[0][1][:,1]]
    elif isinstance(sim, lab.simulator.Simulator):
        result = sim.run(simulation_length=time)
        time = result[0][0]
        s = result[0][1][:, 0]
        rate = result[0][1][:, 1]
    else:
        raise ValueError('The class type is not supported.')
    return time, s, rate


class TvbSim:

    def __init__(self, weight, delay, id_proxy, resolution_simulation, synchronization_time, initial_condition=None):
        """
        initialise the simulator
        :param weight: weight on the connexion
        :param delay: delay of the connexions
        :param id_proxy: the id of the proxy
        :param resolution_simulation: the resolution of the simulation
        :param initial_condition: initial condition for S and H
        """
        self.nb_node = weight.shape[0] - len(id_proxy)
        model = tvb_model(resolution_simulation, weight, delay, id_proxy)
        self.sim = tvb_init(model, synchronization_time, initial_condition)
        self.dt = self.sim.integrator.dt
        if initial_condition is not None:
            self.current_state = np.expand_dims(initial_condition[-1 ,0, id_proxy, 0],[0, 2]) # only one mode, only S

    def __call__(self, time, proxy_data=None, rate_data=None, rate=False):
        """
        run simulation for t biological
        :param time: the time of the simulation
        :param proxy_data: the firing rate fo the next steps for the proxy
        :return:
            the result of time, the firing rate and the state of the network
        """
        if rate_data is None:
            time, s_out, rates = tvb_simulation(time, self.sim, proxy_data)
        else:
            proxy_data = [rate_data[0], self.transform_rate_to_s(rate_data[1])]
            time, s_out, rates = tvb_simulation(time, self.sim, proxy_data)

        if rate:
            return time, s_out, rates
        else:
            return time, s_out

    def transform_rate_to_s(self, rate_data):
        """
        Transform the rate in percentage of open synapse
        :param rate_data: rate data
        :return: the percentage of open synapse
        """
        S = []
        X = self.current_state
        for h in rate_data:
            def dfun_proxy(x,c, local_coupling=0.0, stimulus=0.0):
                g = self.sim.model.gamma
                t = self.sim.model.tau_s
                return _numba_dfun_proxy(x,np.expand_dims(h,[0,2]),g,t)
            X = self.sim.integrator.scheme(X, dfun_proxy, 0.0, 0.0, 0.0)
            S.append(X)
        self.current_state = X
        return np.concatenate(S)
