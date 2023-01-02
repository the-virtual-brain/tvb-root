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

import numpy

from tvb.basic.neotraits.api import HasTraits, Attr, NArray
from tvb.simulator.coupling import Coupling, Linear
from tvb.simulator.monitors import Raw, RawVoi, AfferentCoupling


class CosimMonitor(HasTraits):
    """
    Abstract base class for cosimulation monitors implementations.
    """

    def _sample_with_tvb_monitor(self, step, state):
        """
        This method provides output from TVB Monitor classes, and should be set to a TVB Monitor parent class.
        Use the original signature.
        """
        raise NotImplemented

    def _get_sample(self, current_step, start_step, n_steps, history, cosim):
        end_step = start_step + n_steps
        if end_step - 1 > current_step:
            raise ValueError("Values of state variables are missing for %d time steps "
                             "from start_step (=%d) to start_step + n_steps - 1 (=%d).\n"
                             "The simulator contains only the state until step %d."
                             % (n_steps, start_step, end_step - 1, current_step))
        # cosim_history has n_time = n_synchronization_step past values, including the current_step
        # so, the last available step in the past is current_step - n_synchronization_step + 1
        # whereas TVB history has n_time = max_delay + 1 past values, i.e., max_delay steps + the current_step,
        # and the last TVB available step in the past is current_step - max_delay
        last_available_step_in_the_past = current_step - history.n_time + numpy.where(cosim, 1, 0)
        if start_step < current_step - history.n_time:
            raise ValueError("Values of state variables are missing for %d time steps "
                             "from start_step (=%d) to start_step + n_steps - 1 (=%d).\n"
                             "The simulator contains only the state from start_step = %d."
                             % (n_steps, start_step, end_step - 1, last_available_step_in_the_past))
        times = []
        values = []
        for step in range(start_step, end_step):
            if cosim:
                state = history.query(step)
            else:
                state = history.query(step)[0]
            tmp = self._sample_with_tvb_monitor(step, state)
            if tmp is not None:
                times.append(tmp[0])
                values.append(tmp[1])
        return [numpy.array(times), numpy.array(values)]

    def sample(self, current_step, start_step, n_steps, cosim_history, history):
        """
        This method provides monitor output, and should be overridden by subclasses.
        Use the original signature.
        """
        raise NotImplemented


class CosimMonitorFromCoupling(CosimMonitor):
    """
       Abstract base class for a monitor that records the future coupling values.
    """

    coupling = Attr(
        field_type=Coupling,
        label="Long-range coupling function",
        default=Linear(),
        required=True,
        doc="""The coupling function is applied to the activity propagated
               between regions by the ``Long-range connectivity`` before it enters the local
               dynamic equations of the Model. Its primary purpose is to 'rescale' the
               incoming activity to a level appropriate to Model.""")

    synchronization_n_step = None

    def _get_sample(self, current_step, start_step, n_steps, history):
        end_step = start_step + n_steps
        last_available_step_in_the_future = current_step + self.synchronization_n_step
        if end_step - 1 > last_available_step_in_the_future:
            raise ValueError("Values of coupling are missing for %d time steps "
                             "from start_step (=%d) to start_step + n_steps -1 (=%d).\n"
                             "The coupling can be computed until step %d."
                             % (n_steps, start_step, end_step - 1, last_available_step_in_the_future))
        first_available_step = current_step + 1
        if start_step < first_available_step:
            raise ValueError("Values of coupling are missing for %d time steps "
                             "from start_step (=%d) to start_step + n_steps -1 (=%d).\n"
                             "The coupling can be computed from current_step + 1 = %d."
                             % (n_steps, start_step, end_step - 1, first_available_step))
        times = []
        values = []
        for step in range(start_step, end_step):
            tmp = self._sample_with_tvb_monitor(step, self.coupling(step, history))
            if tmp is not None:
                times.append(tmp[0])
                values.append(tmp[1])
        return [numpy.array(times), numpy.array(values)]

    def _config_time(self, simulator):
        self.synchronization_n_step = simulator.synchronization_n_step
        # For less constraint, the previous value can be replaced by the minimum of delay.
        # i.e. : numpy.min(simulator.connectivity.idelays[numpy.nonzero(simulator.connectivity.idelays)


class RawCosim(Raw, CosimMonitor):
    """
    A monitor that records the output raw data
    from the partial (up to synchronization time) cosimulation history of TVB simulation.
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """
    _ui_name = "Cosimulation Raw recording"

    def _sample_with_tvb_monitor(self, step, state):
        return Raw.sample(self, step, state)

    def sample(self, current_step, start_step, n_steps, cosim_history, history):
        "Return all the states of the partial (up to synchronization time) cosimulation history"
        return self._get_sample(current_step, start_step, n_steps, cosim_history, cosim=True)


class RawVoiCosim(RawVoi, CosimMonitor):
    """
    A monitor that records the output raw data of selected variables
    from the partial (up to synchronization time) history of TVB simulation.
    It collects:

        - voi state variables and all modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """
    _ui_name = "Cosimulation RawVoi recording"

    def _sample_with_tvb_monitor(self, step, state):
        return RawVoi.sample(self, step, state)

    def sample(self, current_step, start_step, n_steps, cosim_history, history):
        "Return all the states of the partial (up to synchronization time) cosimulation history"
        return self._get_sample(current_step, start_step, n_steps, cosim_history, cosim=True)


class RawDelayed(Raw, CosimMonitor):
    """
    A monitor that records the output raw data of all coupling variables
    from the full history of a TVB simulation.
    It collects:

        - all coupling state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Cosimulation Raw Delayed recording"

    def _sample_with_tvb_monitor(self, step, state):
        return Raw.sample(self, step, state)

    def sample(self, current_step, start_step, n_steps, cosim_history, history):
        "Return all the states of the delayed (by synchronization time) TVB history"
        return self._get_sample(current_step, start_step, n_steps, history, cosim=False)


class RawVoiDelayed(RawVoi, CosimMonitor):
    """
    A monitor that records the output raw data of selected coupling variables
    from the full history of a TVB simulation.
    It collects:

        - selected coupling state variables and all modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Cosimulation RawVoi Delayed recording"

    def _sample_with_tvb_monitor(self, step, state):
        return RawVoi.sample(self, step, state)

    def sample(self, current_step, start_step, n_steps, cosim_history, history):
        "Return selected states of the delayed (by synchronization time) TVB history"
        return self._get_sample(current_step, start_step, n_steps, history, cosim=False)


class CosimCoupling(AfferentCoupling, CosimMonitorFromCoupling):
    """
    A monitor that records the future coupling of selected variables:
    It collects:

        - selected coupling values and all modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps
    """

    _ui_name = "Cosimulation Coupling recording"

    def _config_time(self, simulator):
        " Define time variables for the monitors and the cosimonitor."
        AfferentCoupling._config_time(self,simulator)
        CosimMonitorFromCoupling._config_time(self,simulator)

    def _sample_with_tvb_monitor(self, step, state):
        return AfferentCoupling.sample(self, step, state)

    def sample(self, current_step, start_step, n_steps, cosim_history, history):
        "Return selected values of future coupling from (up to synchronization time) cosimulation history"
        return self._get_sample(current_step, start_step, n_steps, history)
