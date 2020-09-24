# -*- coding: utf-8 -*-
import numpy
from tvb.basic.neotraits.api import Attr, Float
from tvb.basic.neotraits.ex import TraitAttributeError
from tvb.simulator.common import iround
from tvb.simulator.history import BaseHistory
from tvb.simulator.coupling import Coupling, Linear
from tvb.simulator.monitors import Monitor
from tvb.contrib.cosimulation.history import CosimHistory


class CosimMonitor(Monitor):

    """Class that wraps around any TVB Monitor class"""

    def from_tvb_monitor(self, monitor, simulator):
        """
        Create a CosimMonitor by copying the attributes of an instance of a TVB Monitor class
        :param
            model: TVB monitor
            simulator: TVB co-simulator
        """
        self._tvb_monitor_class = monitor.__class__
        for key, value in vars(monitor).items():
            try:
                setattr(self, key, value)
            except TraitAttributeError:
                # variable final don't need to copy
                pass
        self.configure()

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator.
            Grab the Simulator's integration step size.
            Set the monitor's variables
            of interest based on the Monitor's 'variables_of_interest' attribute, if
            it was specified, otherwise use the ones specified
            for the Model. Calculate the number of integration steps (isteps)
            between returns by the record method. This method is called from within
            the the Simulator's configure() method.
        """
        self._tvb_monitor_class.config_for_sim(self, simulator)

    def sample(self, step, state):
        """Sample using the TVB Monitor class sampling method"""
        return self._tvb_monitor_class.sample(step, state)


class CosimHistoryMonitor(CosimMonitor):

    """Class that wraps around any TVB Monitor class in order to impose recording only every period = sync_time,
       when the TVB state is assumed to be updated by the other co-simulator,
       for the time [current_time - sync_time, current_time]."""

    sync_time = Float(
        label="Cosimulation monitor synchronization time (ms)",  # order = 10
        default=0.9765625,  # ms. 0.9765625 => 1024Hz #ms, 0.5 => 2000Hz
        required=True,
        doc="""Cosimulation monitor synchronization time in milliseconds, must be an integral multiple
                of integration-step size. As a guide: 2048 Hz => 0.48828125 ms ;  
                1024 Hz => 0.9765625 ms ; 512 Hz => 1.953125 ms.""")

    history = Attr(
        field_type=CosimHistory,
        label="Cosimulation history",
        default=None,
        required=True,
        doc="""A history of the whole TVB state for the cosimulation synchronization time.""")

    def from_tvb_monitor(self, monitor, simulator, sync_time):
        """
        Create a CosimMonitor by copying the attributes of an instance of a TVB Monitor class
        :param
            model: TVB monitor
            sync_time: cosimulation synchronization time
            simulator: TVB co-simulator
        """
        self.sync_time = sync_time
        self.history = simulator.history
        super(CosimHistoryMonitor, self).from_tvb_monitor(monitor, simulator)

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator.
                    Grab the Simulator's integration step size.
                    Set the monitor's variables
                    of interest based on the Monitor's 'variables_of_interest' attribute, if
                    it was specified, otherwise use the cvar specified
                    for the Model. Calculate the number of integration steps (isteps)
                    between returns by the record method. This method is called from within
                    the the Simulator's configure() method.
        """
        self._tvb_monitor_class.config_for_sim(self, simulator)
        self.history = simulator.history
        if self.istep > self.history.n_time:
            raise ValueError("Monitor period %g cannot be longer than "
                             "the history buffer time length %g!"
                             % (self.period, self.dt * self.history.n_time))
        self.sync_istep = iround(self.sync_time / simulator.integrator.dt)
        if self.sync_istep > self.history.n_time:
            raise ValueError("Synchronization time %g for cosimulation update cannot "
                             "be longer than the history buffer time length %g!"
                             % (self.sync_time, self.dt * self.history.n_time))

    def sample(self, step, state):
        """Force sampling every sync_time and for all output (for vois only) generated during sync_time."""
        # If step % sync_istep is 0...
        if step % self.sync_istep == 0:
            # ...compute the last ...
            end_step = step + 1
            # ...and first step of the sampling period...
            start_step = end_step - self.sync_istep
            # ...and loop through the sampling period...
            time = []
            data = []
            for _step in range(start_step, end_step):
                # ...to provide states to the sample method of the TVB monitor...
                _output = self._tvb_monitor_class.sample(self,
                                                        _step,
                                                        self.history.query_state(_step))
                if _output is not None:
                    time.append(_output[0])
                    data.append(_output[1])
            if len(time) > 0:
                # ...and form the final output of record, if any...
                return [time, data]


class CouplingMonitor(Monitor):
    """
        Monitor that at every period time computes and returns node coupling
        for all past and present time steps equal to period time.

    """

    history = Attr(
        field_type=BaseHistory,
        label="Simulator history",
        default=None,
        required=False,
        doc="""Simulator history.""")

    coupling = Attr(
        field_type=Coupling,
        label="Simulator's Long-range coupling function",
        default=Linear(),
        required=False,
        doc="""The coupling function is applied to the activity propagated
            between regions by the ``Long-range connectivity`` before it enters the local
            dynamic equations of the Model. Its primary purpose is to 'rescale' the
            incoming activity to a level appropriate to Model.""")

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator.

                Grab the Simulator's integration step size, history and coupling instances.
                Set the monitor's variables
                of interest based on the Monitor's 'variables_of_interest' attribute, if
                it was specified, otherwise use the cvar specified
                for the Model. Calculate the number of integration steps (isteps)
                between returns by the record method. This method is called from within
                the the Simulator's configure() method.
        """
        self.dt = simulator.integrator.dt
        self.istep = iround(self.period / self.dt)
        self.voi = self.variables_of_interest
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.cvar)]
        self.history = simulator.history
        if self.istep > self.history.n_time:
            raise ValueError("Synchronization time %g for cosimulation update cannot "
                             "be longer than the history buffer time length %g!"
                             % (self.period, self.dt * self.history.n_time))
        self.coupling = simulator.coupling

    def sample(self, step, state):
        """
        Computes and samples node coupling from history
        for the time length of the sampling period
        and for state variables of indices voi,
        if integration step corresponds to sampling period.
        """
        # If step % istep is 0...
        if step % self.istep == 0:
            # ...compute the last ...
            end_step = step + 1
            # ...and first step of the sampling period...
            start_step = end_step - self.sync_istep
            # ...and loop through the sampling period by computing node coupling from history:
            output = []
            for _step in range(start_step, end_step):
                output.append(self.coupling(_step, self.history)[self.voi])
            # Finally, form the time vector and return the sample:
            return [numpy.arange(start_step, end_step) * self.dt,
                    numpy.array(output)]


class CouplingCosimMonitor(CouplingMonitor, CosimHistoryMonitor):

    @property
    def sync_time(self):
        return self.period

    def from_tvb_monitor(self, monitor, simulator, sync_time):
        """
        Create a CosimTVBMonitor by copying the attributes of an instance of a TVB Monitor class
        :param
            model: TVB monitor
            sync_time: cosimulation synchronization time
            simulator: TVB co-simulator
            Force period to be equal to sync_time.
        """
        self.period = sync_time
        self.history = simulator.history
        self.coupling = simulator.coupling
        super(CosimHistoryMonitor, self).from_tvb_monitor(monitor, simulator)

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator.

                Grab the Simulator's integration step size, history and coupling instances.
                Set the monitor's variables
                of interest based on the Monitor's 'variables_of_interest' attribute, if
                it was specified, otherwise use the cvar specified
                for the Model. Calculate the number of integration steps (isteps)
                between returns by the record method. This method is called from within
                the the Simulator's configure() method.
        """
        CouplingMonitor.config_for_sim(self, simulator)

    def sample(self, step, state):
        """
                Computes and samples node coupling from history
                for the time length of sync_time
                and for state variables of indices voi,
                if integration step corresponds to sync_time period.
        """
        CouplingMonitor.sample(self, step, state)
