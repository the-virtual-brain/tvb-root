# -*- coding: utf-8 -*-

import abc

import numpy

from tvb.basic.neotraits.api import HasTraits, Attr, NArray
from tvb.simulator.coupling import Coupling, Linear
from tvb.simulator.monitors import AfferentCoupling, Raw, RawVoi


class CosimMonitor(HasTraits):
    """
    Abstract base class for cosimnulaiton monitors implementations.
    """

    @abc.abstractmethod
    def cosim_sample(self, voi, start_step, n_steps, cosim_history, history):
        """
        This method provides monitor output, and should be overridden by subclasses.
        Use the original signature.
        """
        pass


class CosimMonitorFromHistory(CosimMonitor):
    """
        Abstract base class for a monitor that records output raw data of coupling variables
        from the full history of a tvb simulation
    """

    def cosim_sample(self, voi, dt, start_step, n_steps, cosim_history, history):
        "Return voi's states of the delayed (by synchronization time) TVB history"
        times = []
        values = []
        for step in range(start_step, start_step + n_steps):
            times.append(step * dt)
            values.append(history.query_sparse(step)[voi])
        return [numpy.array(times), numpy.array(values)]


class CosimMonitorFromCoupling(CosimMonitor):
    """
       Abstract base class for a monitor that records the future coupling values.
       !!!WARNING don't use this monitor for a time smaller than the synchronization variable!!!
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

    def cosim_sample(self, voi, dt, start_step, n_steps, cosim_history, history):
        "Return voi's coupling values of the nodes from the TVB history"
        times = []
        couplings = []
        for step in range(start_step, start_step + n_steps):
            times.append(step * dt)
            couplings.append(self.coupling(step, history)[voi])
        return [numpy.array(times), numpy.array(couplings)]


class CosimMonitorFromCosimHistory(CosimMonitor):
    """
        Abstract base class for a monitor that records output raw data from the incomplete history of TVB simulation.
    """

    def cosim_sample(self, voi, dt, start_step, n_steps, cosim_history, history):
        "Return voi's states of the incomplete (up to synchronization time) cosimulation history"
        times = []
        values = []
        for step in range(start_step, start_step + n_steps):
            times.append(step * dt)
            values.append(cosim_history.query(step)[voi])
        return [numpy.array(times), numpy.array(values)]


class RawDelayed(CosimMonitorFromHistory, AfferentCoupling):
    """
    A monitor that records the output raw data of all coupling variables from the full history of a tvb simulation.
    It collects:

        - all coupling state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Cosimulation Raw Delayed recording"

    variables_of_interest = NArray(
        dtype=int,
        label="Set all the couplings variables exclusively!!! Resistance is futile...",
        required=False)

    def _config_vois(self, simulator):
        self.voi = numpy.r_[:len(simulator.model.cvar)]

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return all the states of the delayed (by synchronization time) TVB history"
        return self.cosim_sample(self.voi, self.dt, start_step, n_steps, cosim_history, history)


class RawVoiDelayed(CosimMonitorFromHistory, AfferentCoupling):
    """
    A monitor that records the output raw data of selected coupling variables from the full history of a tvb simulation.
    It collects:

        - selected coupling state variables and all modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Cosimulation RawVoi Delayed recording"

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return selected states of the delayed (by synchronization time) TVB history"
        return self.cosim_sample(self.voi, self.dt, start_step, n_steps, cosim_history, history)


class CosimCoupling(CosimMonitorFromCoupling, AfferentCoupling):
    """
    A monitor that records the future coupling of all variables:
    It collects:

        - all coupling values and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    !!!WARNING don't use this monitor for a time smaller than the synchronization variable!!!
    """

    _ui_name = "Cosimulation Coupling recording"

    variables_of_interest = NArray(
        dtype=int,
        label="Set all the couplings variables exclusively!!! Resistance is futile...",
        required=False)

    def _config_vois(self, simulator):
        self.voi = numpy.r_[:len(simulator.model.cvar)]

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return the all coupling values of the nodes from the TVB history"
        return self.cosim_sample(self.voi, self.dt, start_step, n_steps, cosim_history, history)


class CosimCouplingVoi(CosimMonitorFromCoupling, AfferentCoupling):
    """

    A monitor that records the future coupling of selected variables:
    It collects:

        - selected coupling values and all modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    !!!WARNING don't use this monitor for a time smaller than the synchronization variable!!!
    """

    _ui_name = "Cosimulation CouplingVoi recording"

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return selected coupling values of the nodes from the TVB history"
        return self.cosim_sample(self.voi, self.dt, start_step, n_steps, cosim_history, history)


class RawCosim(CosimMonitorFromCosimHistory, Raw):
    """
    A monitor that records the output raw data from the incomplete history of TVB simulation.
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """
    _ui_name = "Cosimulation Raw recording"

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return all the states of the incomplete (up to synchronization time) cosimulation history"
        return self.cosim_sample(self.voi, self.dt, start_step, n_steps, cosim_history, history)


class RawVoiCosim(CosimMonitorFromCosimHistory, RawVoi):
    """
    A monitor that records the output raw data of selected variables from the incomplete history of TVB simulation.
    It collects:

        - voi state variables and all modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """
    _ui_name = "Cosimulation RawVoi recording"

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return all the states of the incomplete (up to synchronization time) cosimulation history"
        return self.cosim_sample(self.voi, self.dt, start_step, n_steps, cosim_history, history)
