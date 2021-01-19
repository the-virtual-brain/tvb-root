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

    def get_sample(self, start_step, n_steps, history,cosim):
        times = []
        values = []
        for step in range(start_step, start_step + n_steps):
            if cosim:
                state = history.query(step)
            else:
                state = history.query(step)[0]
            tmp = super(self.__class__,self).sample(step,state)
            if tmp is not None:
                times.append(tmp[0])
                values.append(tmp[1])
        return [numpy.array(times), numpy.array(values)]

    @abc.abstractmethod
    def sample(self, voi, start_step, n_steps, cosim_history, history):
        """
        This method provides monitor output, and should be overridden by subclasses.
        Use the original signature.
        """
        pass

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

    def get_sample(self,start_step,n_steps,history):
        times = []
        values = []
        for step in range(start_step, start_step + n_steps):
            tmp = super(self.__class__,self).sample(step,self.coupling(step,history))
            if tmp is not None:
                times.append(tmp[0])
                values.append(tmp[1])
        return [numpy.array(times), numpy.array(values)]


class RawCosim(Raw, CosimMonitor):
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
        return self.get_sample(start_step, n_steps, cosim_history, cosim=True)


class RawVoiCosim(RawVoi, CosimMonitor):
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
        return self.get_sample(start_step, n_steps, cosim_history, cosim=True)


class RawDelayed(Raw, CosimMonitor):
    """
    A monitor that records the output raw data of all coupling variables from the full history of a tvb simulation.
    It collects:

        - all coupling state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Cosimulation Raw Delayed recording"

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return all the states of the incomplete (up to synchronization time) cosimulation history"
        return self.get_sample(start_step, n_steps,history, cosim=False)


class RawVoiDelayed(RawVoi,CosimMonitor):
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
        return self.get_sample(start_step, n_steps,history, cosim=False)


class CosimCoupling(AfferentCoupling, CosimMonitorFromCoupling):
    """
    A monitor that records the future coupling of all variables:
    It collects:

        - all coupling values and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    !!!WARNING don't use this monitor for a time smaller than the synchronization variable!!!
    """

    _ui_name = "Cosimulation Coupling recording"

    def sample(self, start_step, n_steps, cosim_history, history):
        "Return all the states of the incomplete (up to synchronization time) cosimulation history"
        return self.get_sample(start_step, n_steps, history)