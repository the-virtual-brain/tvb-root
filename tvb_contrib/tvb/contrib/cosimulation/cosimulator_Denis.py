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
This is the module responsible for co-simulation of TVB with spiking simulators.
It inherits the Simulator class.

.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>


"""
import time
import math
import numpy

from tvb.basic.neotraits.api import Attr
from tvb.contrib.cosimulation.tvb_to_cosim_interfaces import TVBtoCosimInterfaces
from tvb.contrib.cosimulation.cosim_to_tvb_interfaces import CosimToTVBInterfaces
from tvb.contrib.cosimulation.coSimulator_1 import CoSimulator


class CoSimulator_Denis(CoSimulator):

    tvb_to_cosim_interfaces = Attr(
        field_type=TVBtoCosimInterfaces,
        label="TVB to cosimulation outlet interfaces",
        default=None,
        required=False,
        doc="""Interfaces to couple from TVB to a 
               cosimulation outlet (i.e., translator level or another (co-)simulator""")

    cosim_to_tvb_interfaces = Attr(
        field_type=CosimToTVBInterfaces,
        label="Cosimulation to TVB interfaces",
        default=None,
        required=False,
        doc="""Interfaces for updating from a cosimulation outlet 
               (i.e., translator level or another (co-)simulator to TVB.""")

    PRINT_PROGRESSION_MESSAGE = True

    def _configure_cosimulation(self):
        """This method will
           - set the synchronization time and number of steps,
           - create CosimHistory,
           - run all the configuration methods of all TVB <-> Cosimulator interfaces,
           If there are any Cosimulator -> TVB update interfaces:
            - remove connectivity among region nodes modelled exclusively in the other co-simulator.
           If there is any current state cosim_to_tvb interface update:
            - generate a CosimModel class from the original Model class,
            - set the cosim_vars and cosim_vars_proxy_inds properties of the CosimModel class,
              based on the respective vois and proxy_inds of all cosim_to_tvb state interfaces.
           If there is any history cosim_to_tvb interface update:
            - throw a warning if we are not using a coupling function that requires the current state.
           """
        self.proxy_inds = numpy.array([])
        self.voi = numpy.array([])
        if self.tvb_to_cosim_interfaces:
            # Configure any TVB to Cosim interfaces:
            self.tvb_to_cosim_interfaces.configure()
        if self.cosim_to_tvb_interfaces:
            # Configure any Cosim to TVB interfaces:
            self.cosim_to_tvb_interfaces.configure(self)
            # A flag to know if the connectivity needs to be reconfigured:
            reconfigure_connectivity = False
            for variable in self.cosim_to_tvb_interfaces.vio:
                if variable not in self.vio:
                    self.vio = numpy.concatenate((self.voi,[variable]))
            for interface in self.cosim_to_tvb_interfaces.interfaces:
                for index_proxy in interface.proxy_inds:
                    self.proxy_inds = numpy.concatenate((self.proxy_inds,[index_proxy]))
                # Interfaces marked as "exclusive" by the user
                # should eliminate the connectivity weights among the proxy nodes,
                # since those nodes are mutually coupled within the other (co-)simulator network model.
                if interface.exclusive:
                    reconfigure_connectivity = True
                    self.connectivity.weights[interface.proxy_inds][:, interface.proxy_inds] = 0.0
            if reconfigure_connectivity:
                self.connectivity.configure()

    def configure(self, full_configure=True):
        """Configure simulator and its components.

        The first step of configuration is to run the configure methods of all
        the Simulator's components, ie its traited attributes.

        Configuration of a Simulator primarily consists of calculating the
        attributes, etc, which depend on the combinations of the Simulator's
        traited attributes (keyword args).

        Converts delays from physical time units into integration steps
        and updates attributes that depend on combinations of the 6 inputs.

        Returns
        -------
        sim: Simulator
            The configured Simulator instance.

        """
        self._configure_cosimulation()
        super(CoSimulator_Denis, self).configure(full_configure=full_configure)
        return self

    def _print_progression_message(self, step, n_steps):
        """
        #TODO do yu it for the moment
        :param step:
        :param n_steps:
        :return:
        """
        if step - self.current_step >= self._tic_point:
            toc = time.time() - self._tic
            if toc > 600:
                if toc > 7200:
                    time_string = "%0.1f hours" % (toc / 3600)
                else:
                    time_string = "%0.1f min" % (toc / 60)
            else:
                time_string = "%0.1f sec" % toc
            print_this = "\r...%0.1f%% done in %s" % \
                         (100.0 * (step - self.current_step) / n_steps, time_string)
            self.log.info(print_this)
            self._tic_point += self._tic_ratio * n_steps

    def run(self, **kwds):
        # TODO need to be test it
        """Convenience method to call the simulator with **kwds and collect output data."""
        ts, xs = [], []
        for _ in self.monitors:
            ts.append([])
            xs.append([])
        wall_time_start = time.time()
        # loop over all the synchronization step
        for step_synch in range( 0,int(math.ceil(self.simulation_length / self.integrator.dt)),self.synchronization_n_step):
            # get the data to update the values of the proxies
            data_proxy = [None for i in self.vio]
            for interface in self.cosim_to_tvb_interfaces.interfaces:
                tmp = interface.get_value() # structure : [0] : time  [1] : values [ nb_time,nb_variable,nb_id_proxy,nb_mod]
                for index,i in enumerate(interface.vio):
                    data_proxy[numpy.where(self.vio == i)[0][0]] = tmp[1][:,:,index,:]
            # loop of TVB simulator
            for data in self(cosim_updates=[tmp[0],data_proxy],**kwds):
                for tl, xl, t_x in zip(ts, xs, data):
                    if t_x is not None:
                        t, x = t_x
                        tl.append(t)
                        xl.append(x)
            data_co_sim = self.output_co_sim_monitor(self,step_synch*self.synchronization_n_step,self.synchronization_n_step)
            # loop to push the data to the other simulator
            for interface in self.tvb_to_cosim_interfaces.interfaces:
                tmp = [data[1][:,:,interface.vio,:] for data in data_co_sim]
                interface.record([data_co_sim[0][0],tmp])
            elapsed_wall_time = time.time() - wall_time_start
            self.log.info("%.3f s elapsed, %.3fx real time", elapsed_wall_time,
                          elapsed_wall_time * 1e3 / self.simulation_length)
        for i in range(len(ts)):
            ts[i] = numpy.array(ts[i])
            xs[i] = numpy.array(xs[i])
        return list(zip(ts, xs))
