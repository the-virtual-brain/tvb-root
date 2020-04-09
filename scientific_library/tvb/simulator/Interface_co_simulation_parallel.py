# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
Defines a set Interface input and output of TVB.

.. moduleauthor:: Lionel Kusch <lkusch@thevirtualbrain.org>

"""
from tvb.simulator.monitors import Raw, NArray, Float
import numpy


class HistoryProxy:
    def copy_inst(self, history):
        """
        Copy the value of an instance without proxy
        :param history: History proxy
        """
        # copy the import element for define all the object
        for key in dir(history):
            try:
                if key[0] != '_':
                    setattr(self, key, getattr(history, key))
            except AttributeError:
                pass
            except TypeError:
                pass
        # copy all elements
        for key in dir(history):
            try:
                if key[0] != '_':
                    setattr(self, key, getattr(history, key))
            except AttributeError:
                pass


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
        self._id_node = \
        numpy.where(numpy.logical_not(numpy.isin(numpy.arange(0, simulator.number_of_nodes, 1), self.id_proxy)))[0]
        self._nb_step_time = numpy.int(self.time_synchronize / simulator.integrator.dt)

        # ####### WARNING:Change the instance simulator for taking in count the proxy ########
        # overwrite of the simulator for update the proxy value
        class Simulator_proxy(type(simulator)):
            # Modify the call method of the simulator in order to update the proxy value
            def __call__(self, simulation_length=None, random_state=None, proxy_data=None):
                if hasattr(self.history, 'update_proxy') and proxy_data is not None:
                    self.history.update_proxy(self.current_step, proxy_data)
                return super(type(simulator), self).__call__(simulation_length=simulation_length,
                                                             random_state=random_state)

        # change the class of the simulator
        simulator.__class__ = Simulator_proxy

        def coupling(step):
            return simulator._loop_compute_node_coupling(step)

        self.coupling = coupling

        # ####### WARNING:Change the instance history for taking in count the proxy #########
        id_proxy = self.id_proxy
        dt = simulator.integrator.dt
        delay_proxy = simulator.history.delays[id_proxy, :]
        delay_proxy = delay_proxy[:, id_proxy]
        min_delay =  -numpy.min(delay_proxy, initial=numpy.Inf, where=delay_proxy != 0.0)
        if min_delay == -numpy.Inf:
            min_delay = numpy.iinfo(numpy.int32).min
        else:
            min_delay = int(-numpy.min(delay_proxy, initial=numpy.Inf, where=delay_proxy != 0.0))

        class History_proxy(HistoryProxy, simulator.history.__class__):
            def __init__(self):
                pass

            # WARNING should be change if the histiry are different (the actal update is the same all history
            def update_proxy(self, step, data):
                """
                update the history with the new value
                :param step: the current step
                """
                if id_proxy.size != 0:
                    step_n = data[0] / dt - step  # the index of the buffer
                    if any(step_n > self.n_time):  # check if there are not too much data
                        raise Exception('ERROR too early')
                    if any(numpy.rint(step_n).astype(int) < min_delay):  # check if it's not missing value
                        # WARNING doesn't in count the modification of delay
                        raise Exception('ERROR too late')
                    indice = numpy.expand_dims(numpy.rint(step_n + step).astype(int) % self.n_time, 1)
                    if indice.size != numpy.unique(indice).size:  # check if the index is correct
                        raise Exception('ERRROR two times are the same')
                    self.buffer[indice, :, id_proxy, :] = data[1]

        new_history = History_proxy()
        new_history.copy_inst(simulator.history)
        del simulator.history  # remove old history
        # add a method for update the history
        simulator.history = new_history

    def sample(self, step, state):
        """
        record of the monitor in order to send result of not proxy node
        :param step: current step
        :param state: the state of all the node and also the value of the proxy
        :return:
        """
        self.step = step
        time = numpy.empty((2,), dtype=object)
        time[:] = [step * self.dt, (step + self._nb_step_time) * self.dt]
        result = numpy.empty((2,), dtype=object)
        result[:] = [state[:, self._id_node, :], self.coupling(step + self._nb_step_time)[:, self.id_proxy, :]]
        return [time, result]
