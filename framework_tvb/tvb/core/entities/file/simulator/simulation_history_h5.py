# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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

from tvb.basic.neotraits.api import HasTraits, NArray, Int, List, Attr, Float
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar, Json
from tvb.simulator.integrators import IntegratorStochastic


class SimulationHistory(HasTraits):
    """
    Simulation State, prepared for H5 file storage.
    """
    history = NArray(required=False)
    # State array, all state variables
    current_state = NArray(required=False)
    # Simulator's current step number (in time)
    current_step = Int()

    # Array with _stock array for every monitor configured in current simulation.
    # As the monitors are dynamic, we prepare a bunch of arrays for storage in H5 file.
    monitor_stock_1 = NArray(required=False)
    monitor_stock_2 = NArray(required=False)
    monitor_stock_3 = NArray(required=False)
    monitor_stock_4 = NArray(required=False)
    monitor_stock_5 = NArray(required=False)
    monitor_stock_6 = NArray(required=False)
    monitor_stock_7 = NArray(required=False)
    monitor_stock_8 = NArray(required=False)
    monitor_stock_9 = NArray(required=False)
    monitor_stock_10 = NArray(required=False)
    monitor_stock_11 = NArray(required=False)
    monitor_stock_12 = NArray(required=False)
    monitor_stock_13 = NArray(required=False)
    monitor_stock_14 = NArray(required=False)
    monitor_stock_15 = NArray(required=False)
    # For matching the stocks above on reload
    monitor_names = List(of=str)

    # In case of noisy integrator, remember the Random State generator status
    integrator_noise_rng_state_algo = Attr(field_type=str, required=False)
    integrator_noise_rng_state_keys = NArray(dtype='uint32', required=False)
    integrator_noise_rng_state_pos = Int(required=False)
    integrator_noise_rng_state_has_gauss = Int(required=False)
    integrator_noise_rng_state_cached_gauss = Float(required=False)

    def __init__(self, **kwargs):
        """
        Constructor for Simulator State
        """
        super(SimulationHistory, self).__init__(**kwargs)
        self.visible = False

    def populate_from(self, simulator_algorithm):
        """
        Prepare a state for storage from a Simulator object.
        """
        self.history = simulator_algorithm.history.buffer.copy()
        self.current_step = simulator_algorithm.current_step
        self.current_state = simulator_algorithm.current_state

        monitor_names = []
        for i, monitor in enumerate(simulator_algorithm.monitors):
            field_name = "monitor_stock_" + str(i + 1)
            setattr(self, field_name, monitor._stock)
            monitor_names.append(type(monitor).__name__)

        if isinstance(simulator_algorithm.integrator, IntegratorStochastic):
            rng_state = simulator_algorithm.integrator.noise.random_stream.get_state()
            self.integrator_noise_rng_state_algo = rng_state[0]
            self.integrator_noise_rng_state_keys = rng_state[1]
            self.integrator_noise_rng_state_pos = rng_state[2]
            self.integrator_noise_rng_state_has_gauss = rng_state[3]
            self.integrator_noise_rng_state_cached_gauss = rng_state[4]

    def fill_into(self, simulator_algorithm):
        """
        Populate a Simulator object from current stored-state.
        """
        simulator_algorithm.history.initialize(self.history)
        simulator_algorithm.current_step = self.current_step
        simulator_algorithm.current_state = self.current_state

        for i, monitor in enumerate(simulator_algorithm.monitors):
            monitor._stock = getattr(self, "monitor_stock_" + str(i + 1))

        if self.integrator_noise_rng_state_algo is not None:
            rng_state = (
                self.integrator_noise_rng_state_algo,
                self.integrator_noise_rng_state_keys,
                self.integrator_noise_rng_state_pos,
                self.integrator_noise_rng_state_has_gauss,
                self.integrator_noise_rng_state_cached_gauss
            )
            simulator_algorithm.integrator.noise.random_stream.set_state(rng_state)


class SimulationHistoryH5(H5File):

    def __init__(self, path):
        super(SimulationHistoryH5, self).__init__(path)
        self.history = DataSet(SimulationHistory.history, self)
        self.current_state = DataSet(SimulationHistory.current_state, self)
        self.current_step = Scalar(SimulationHistory.current_step, self)

        self.monitor_names = Json(SimulationHistory.monitor_names, self)
        for i in range(1, 16):
            stock_name = 'monitor_stock_%i' % i
            setattr(self, stock_name, DataSet(getattr(SimulationHistory, stock_name), self))

        self.integrator_noise_rng_state_algo = Scalar(SimulationHistory.integrator_noise_rng_state_algo, self)
        self.integrator_noise_rng_state_keys = DataSet(SimulationHistory.integrator_noise_rng_state_keys, self)
        self.integrator_noise_rng_state_pos = Scalar(SimulationHistory.integrator_noise_rng_state_pos, self)
        self.integrator_noise_rng_state_has_gauss = Scalar(SimulationHistory.integrator_noise_rng_state_has_gauss, self)
        self.integrator_noise_rng_state_cached_gauss = Scalar(SimulationHistory.integrator_noise_rng_state_cached_gauss,
                                                              self)
