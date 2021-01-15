# -*- coding: utf-8 -*-

import abc

import numpy

from tvb.basic.neotraits.api import Attr, NArray
from tvb.simulator.coupling import Coupling, Linear
from tvb.simulator.monitors import Raw


class CosimMonitor(Raw):
    """
    Abstract base class for monitor implementations.
    """

    @abc.abstractmethod
    def sample(self, start_step, n_steps,history_incomplete,history_delayed):
        """
        This method provides monitor output, and should be overridden by subclasses.
        Use the initial signature
        """
        pass


class RawDelayed(CosimMonitor):
    """
    A monitor that records the output raw data from the full history of a tvb simulation:
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Cosimulation Raw Delayed recording"

    def sample(self, start_step, n_steps, cosim_history, history):
        " Return the states of the delayed (by synchronization time) TVB history "
        times = []
        values = []
        for step in range(start_step, start_step + n_steps):
            times.append(step*self.dt)
            values.append(history.query_sparse(step))
        return [numpy.array(times),numpy.array(values)]


class RawCosim(CosimMonitor):
    """
    A monitor that records the output raw data from the incomplete history of tvb simulation:
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Cosimulation Raw recording"

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return the states of the incomplete (up to synchronization time) cosimulation history "
        times = []
        values = []
        for step in range(start_step, start_step + n_steps):
            times.append(step*self.dt)
            values.append(cosim_history.query(step))
        return [numpy.array(times),numpy.array(values)]


class CosimCoupling(CosimMonitor):
    """
    WARNING don't use this monitor for a time smaller han the synchronization variable
    A monitor that records the future coupling of the variable:
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Cosimulation Coupling recording"

    variables_of_interest = NArray(
        dtype=int,
        label="Set only the coupling variable!!! Resistance is futile...",
        required=False)

    coupling = Attr(
        field_type=Coupling,
        label="Long-range coupling function",
        default=Linear(),
        required=True,
        doc="""The coupling function is applied to the activity propagated
        between regions by the ``Long-range connectivity`` before it enters the local
        dynamic equations of the Model. Its primary purpose is to 'rescale' the
        incoming activity to a level appropriate to Model.""")

    def _config_vois(self, simulator):
        self.voi = simulator.model.cvar

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return the coupling values of the nodes from the TVB history"
        times = []
        couplings = []
        for step in range(start_step, start_step + n_steps):
            times.append(step*self.dt)
            couplings.append(self.coupling(step, history))
        return [numpy.array(times),numpy.array(couplings)]
