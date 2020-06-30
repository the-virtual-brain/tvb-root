#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "


"""
Defines a set Interface input and output of TVB.

.. moduleauthor:: Lionel Kusch <lkusch@thevirtualbrain.org>

"""
from tvb.simulator.monitors import Raw, NArray, Float
from tvb.simulator.history import NDArray,Dim
import numpy

class Interface_co_simulation(Raw):
    id_proxy = NArray(
        dtype=numpy.int,
        label="Identifier of proxies",
    )
    time_synchronize = Float(
        label="simulated time between receiving the value of the proxy",
    )

    def __init__(self, **kwargs):
        super(Interface_co_simulation, self).__init__(**kwargs)

    def config_for_sim(self, simulator):
        # configuration of all monitor
        super(Interface_co_simulation, self).config_for_sim(simulator)
        # add some internal variable
        self._id_node = \
        numpy.where(numpy.logical_not(numpy.isin(numpy.arange(0, simulator.number_of_nodes, 1), self.id_proxy)))[0]
        self._nb_step_time = numpy.int(self.time_synchronize / simulator.integrator.dt)
        self.period = simulator.integrator.dt

        # ####### WARNING:Create a new instance of history for taking in count the proxy (replace the old history) #########
        id_proxy = self.id_proxy
        id_node = self._id_node
        dt = simulator.integrator.dt
        # find the minimum of delay supported for the simulation
        delay_proxy = simulator.history.delays[id_proxy, :]
        delay_proxy = delay_proxy[:, id_node]
        min_delay =  -numpy.min(delay_proxy, initial=numpy.Inf, where=delay_proxy != 0.0)
        if min_delay == -numpy.Inf:
            min_delay = numpy.iinfo(numpy.int32).min
        else:
            min_delay = int(-numpy.min(delay_proxy, initial=numpy.Inf, where=delay_proxy != 0.0))
        class History_proxy(simulator.history.__class__):
            n_proxy = Dim()
            # WARNING same dimension than the buffer in history. (the dimension can be reduce to the minimum of delay)
            # The precision are different for take in count of the input precision
            # The creation of buffer for proxy because it's impossible to replace the data in the buffer of state variable
            buffer_proxy = NDArray(('n_time', 'n_cvar', 'n_proxy', 'n_mode'), numpy.float64, read_only=False)

            def __init__(self, weights, delays, cvars, n_mode):
                super(History_proxy, self).__init__(weights, delays, cvars, n_mode)
                self.n_proxy = id_proxy.shape[0]

            # Update the proxy buffer with the new data
            def update_proxy(self, step, data):
                """
                update the history with the new value
                :param step: the current step
                """
                if id_proxy.size != 0:
                    step_n = data[0] / dt - step # the index of the buffer
                    # TODO need to check if not missing values ( for moment is not robust)
                    # the following works because the simulation length have the same dimension then the data
                    if numpy.rint(numpy.max(step_n)).astype(int)>-min_delay:
                        raise Exception('ERROR missing value for the run')
                    if any(step_n > self.n_time):  # check if there are not too much data
                        raise Exception('ERROR too early')
                    indice = numpy.rint(step_n + step).astype(int) % self.n_time
                    if indice.size != numpy.unique(indice).size:  # check if the index is correct
                        raise Exception('ERROR two times are the same')
                    self.buffer_proxy[indice] = numpy.reshape(data[1],(indice.shape[0],self.n_cvar,self.n_proxy,self.n_mode))

            # query of the value of the proxy
            def query_proxy(self,step):
                return self.buffer_proxy[(step - 1) % self.n_time]

            # WARNING should be change if the function update of the history change  (the actual update is the same all history)
            def update(self, step, new_state):
                self.buffer[step % self.n_time][:,id_node] = new_state[:,id_node][self.cvars]
                self.buffer[step % self.n_time][:,id_proxy] = self.buffer_proxy[step % self.n_time]

        # ####### WARNING:Change the instance simulator for taking in count the proxy ########
        # overwrite of the simulator for update the proxy value
        index_cvar_make = numpy.array(
            [[i, j, k] for i in simulator.model.cvar for j in id_proxy for k in range(simulator.model.number_of_modes)])

        mask_no_cvar = numpy.ones(len(simulator.model.state_variables), numpy.bool)
        mask_no_cvar[simulator.model.cvar] = False
        index_no_cvar_make = numpy.array(
            [[i, j, k] for i in numpy.where(mask_no_cvar)[0] for j in id_proxy for k in range(simulator.model.number_of_modes)])

        class Simulator_proxy(type(simulator)):
            index_cvar =  NArray(label="Index of coupling variable",
                                         default=index_cvar_make)
            index_no_cvar =  NArray(label="Index of notcoupling variable",
                                 default=index_no_cvar_make)
            # Modify the call method of the simulator in order to update the proxy value
            # for the integration in TVB, it's just to modify the state variable after the after the integration scheme
            def __call__(self, simulation_length=None, random_state=None, proxy_data=None):
                if hasattr(self.history, 'update_proxy') and proxy_data is not None:
                    if int(numpy.around(simulation_length / dt)) != proxy_data[0].shape[0]:
                        raise Exception('mission value in the proxy data')
                    self.history.update_proxy(self.current_step, proxy_data)
                yield from super(type(simulator), self).__call__(simulation_length=simulation_length,
                                                         random_state=random_state)
                # update the current state
                if self.index_cvar.shape[0] != 0:
                        self.current_state[(self.index_cvar[:,0],self.index_cvar[:,1],self.index_cvar[:,2])] = numpy.squeeze(self.history.query_proxy(self.current_step+1))
                if self.index_no_cvar.shape[0] != 0:
                    self.current_state[(self.index_no_cvar[:,0],self.index_no_cvar[:,1],self.index_no_cvar[:,2])] = numpy.NAN

            def _loop_monitor_output(self, step, state):
                # modify the state variable before the record of the monitor
                if self.index_cvar.shape[0] != 0:
                    state[(self.index_cvar[:,0],self.index_cvar[:,1],self.index_cvar[:,2])] = numpy.squeeze(self.history.query_proxy(step+1))
                if self.index_no_cvar.shape[0] != 0:
                        state[(self.index_no_cvar[:,0],self.index_no_cvar[:,1],self.index_no_cvar[:,2])] = numpy.NAN
                return super(type(simulator), self)._loop_monitor_output(step, state)

        # change the class of the simulator
        simulator.__class__ = Simulator_proxy

        # WARNING this is the simplification of the initialisation of the history
        # if the the function __loop_history change, the initialisation can change
        # create the new history and initialise it
        new_history = History_proxy(
            simulator.connectivity.weights,
            simulator.connectivity.idelays,
            simulator.model.cvar,
            simulator.model.number_of_modes)
        new_history.initialize(simulator.history.buffer)

        # replace the history
        del simulator.history  # remove old history
        simulator.history = new_history

        # Save the function coupling for the return the coupling of proxy node in the sample
        def coupling(step):
            return simulator._loop_compute_node_coupling(step)
        self.coupling = coupling

    def sample(self, step, state):
        """
        record of the monitor in order to send result of not proxy node
        :param step: current step
        :param state: the state of all the node and also the value of the proxy
        :return:
        """
        self.step = step
        time = (step + self._nb_step_time) * self.period
        result= self.coupling(step + self._nb_step_time)[:, self.id_proxy, :]
        return [time, result]
