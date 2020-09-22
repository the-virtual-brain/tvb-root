# -*- coding: utf-8 -*-
import os
import numpy
from tvb.basic.neotraits.api import Attr, NArray
from tvb.simulator.common import iround
from tvb.simulator.monitors import Monitor
from tvb.simulator.history import BaseHistory, SparseHistory
from tvb.simulator.coupling import Coupling, Linear


class CosimMonitor(Monitor):

    """Cosimulation Monitor base class."""

    proxy_inds = NArray(
        dtype=numpy.int,
        label="Indices of TVB proxy nodes",
        required=True,
    )

    number_of_proxy_nodes = Attr(field_type=int, required=True,
                                 default=0)

    def __init__(self, **kwargs):
        super(CosimMonitor, self).__init__(**kwargs)

    def config_for_sim(self, simulator):
        """Method to configure the Monitor
           and compute the number_of_proxy_nodes, from user defined proxy_inds"""
        self.number_of_proxy_nodes = len(self.proxy_inds)
        super(CosimMonitor, self).config_for_sim(simulator)


class CosimStateMonitor(CosimMonitor):
    """ CosimStateMonitor class:
        Monitors the value for the model's variable/s of interest over all
        the nodes at each sampling period. Time steps that are not modulo ``istep``
        are stored temporarily in the ``_stock`` attribute and then that temporary
        store is returned when time step is modulo ``istep``.

    """

    def __init__(self, **kwargs):
        super(CosimStateMonitor, self).__init__(**kwargs)

    def config_for_sim(self, simulator):
        """Method to
           - configure the Monitor
           - compute the number_of_proxy_nodes, from user defined proxy_inds,
           - and determine initialize _stock buffer."""
        super(CosimStateMonitor, self).config_for_sim(simulator)
        stock_size = (self.istep, self.voi.shape[0],
                      self.number_of_proxy_nodes,
                      simulator.model.number_of_modes)
        self.log.debug("CosimMonitor stock_size is %s" % (str(stock_size),))
        self._stock = numpy.zeros(stock_size)

    def sample(self, step, state):
        """
        Samples from current state for
        - state variables of indices voi,
        - and region nodes of indices proxy_inds,
        if integration step corresponds to sampling period, Otherwise
        just update the monitor's stock. When the step corresponds to the sample/synchronization
        period, the ``_stock`` is returned.
        """
        # Store the current state into stock
        self._stock[((step % self.istep) - 1), :, :] = state[self.voi][:, self.proxy_inds]
        if step % self.istep == 0:
            # Create the time vector and sample, if step % istep == 0
            time = numpy.arange(step - self.istep + 1, step + 1) * self.dt
            return [time, self._stock]


class CosimStateMonitorToFile(CosimStateMonitor):
    """ CosimStateMonitorToFile class:
            Monitors the value for the model's variable/s of interest over all
            the nodes at each sampling period. Time steps that are not modulo ``istep``
            are stored temporarily in the ``_stock`` attribute and then that temporary
            store is returned when time step is modulo ``istep``.

            Records to a file by overloading the record method.

        """

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state):
        """Method to record monitor output to a file."""
        raise NotImplementedError


class CosimStateMonitorToMPI(CosimStateMonitor):
    """ CosimStateMonitorToMPI class:
            Monitors the value for the model's variable/s of interest over all
            the nodes at each sampling period. Time steps that are not modulo ``istep``
            are stored temporarily in the ``_stock`` attribute and then that temporary
            store is returned when time step is modulo ``istep``.

            Records to a MPI port by overloading the record method.
    """

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state):
        """Method to record monitor output to a MPI port."""
        raise NotImplementedError


class CosimHistoryMonitor(Monitor):
    """ CosimHistoryMonitor class:
        Monitors the history for the model's coupling variable/s of interest
        over TVB proxy nodes at each sampling period. Time steps that are not modulo ``istep``
        are stored temporarily in the ``_stock`` attribute and then that temporary
        store is returned when time step is modulo ``istep``.
    """

    history = Attr(
        field_type=BaseHistory,
        label="Simulator history",
        default=SparseHistory(),
        required=True,
        doc="""A tvb.simulator.history""")

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator and compute the number_of_proxy_nodes, from user defined proxy_inds.

                Grab the Simulator's integration step size and history instance.
                Set the monitor's variables
                of interest based on the Monitor's 'variables_of_interest' attribute, if
                it was specified, otherwise use the cvar specified
                for the Model. Calculate the number of integration steps (isteps)
                between returns by the record method. This method is called from within
                the Simulator's configure() method.
        """
        self.voi = self.variables_of_interest
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.cvar)]
        self.history = simulator.history
        self.dt = simulator.integrator.dt
        self.istep = iround(self.period / self.dt)
        if self.istep > self.history.n_time:
            raise ValueError("Synchronization time %g for cosimulation update cannot "
                             "be longer than the history buffer time length %g!"
                             % (self.period, self.dt * self.n_time))

    def sample(self, step, state):
        """
        Samples from history's current state for the time length of the sampling period and for
        - state variables of indices voi,
        - and region nodes of indices proxy_inds,
        if integration step corresponds to sampling period.
        """
        # If step % istep is 0...
        if step % self.istep == 0:
            # ...compute the first ...
            start_step = step - self.istep + 1
            # ...and last step of the sampling period...
            end_step = step + 1
            # ...and loop through the sampling period by quering the history's current_state:
            output = []
            for _step in range(start_step, end_step):
                output.append(self.history.query(_step)[0][self.voi][:, self.proxy_inds])
            # Finally, form the time vector and return the sample:
            return [numpy.arange(start_step, end_step) * self.dt,
                    numpy.array(output)]


class CosimHistoryMonitorToFile(CosimHistoryMonitor):

    """ CosimHistoryMonitorToFile class:
            Monitors the history for the model's coupling variable/s of interest
            over TVB proxy nodes at each sampling period. Time steps that are not modulo ``istep``
            are stored temporarily in the ``_stock`` attribute and then that temporary
            store is returned when time step is modulo ``istep``.
            Records to a file by overloading the record method.
    """

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state):
        """Method to record monitor output to a file."""
        raise NotImplementedError


class CosimHistoryMonitorToMPI(CosimHistoryMonitor):

    """ CosimHistoryMonitorToFile class:
            Monitors the history for the model's coupling variable/s of interest
            over TVB proxy nodes at each sampling period. Time steps that are not modulo ``istep``
            are stored temporarily in the ``_stock`` attribute and then that temporary
            store is returned when time step is modulo ``istep``.
            Records to a MPI port by overloading the record method.
    """

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state):
        """Method to record monitor output to a MPI port."""
        raise NotImplementedError


class CosimCouplingMonitor(CosimHistoryMonitor):
    """
        Monitors the current node coupling for the model's coupling variable/s of interest
        over TVB proxy nodes at each sampling period.

    """

    coupling = Attr(
        field_type=Coupling,
        label="Simulator's Long-range coupling function",
        default=Linear(),
        required=True,
        doc="""The coupling function is applied to the activity propagated
            between regions by the ``Long-range connectivity`` before it enters the local
            dynamic equations of the Model. Its primary purpose is to 'rescale' the
            incoming activity to a level appropriate to Model.""")

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator and compute the number_of_proxy_nodes, from user defined proxy_inds.

                Grab the Simulator's integration step size, history and coupling instances.
                Set the monitor's variables
                of interest based on the Monitor's 'variables_of_interest' attribute, if
                it was specified, otherwise use the cvar specified
                for the Model. Calculate the number of integration steps (isteps)
                between returns by the record method. This method is called from within
                the the Simulator's configure() method.
        """
        super(CosimCouplingMonitor, self).config_for_sim(simulator)
        self.coupling = simulator.coupling

    def sample(self, step, state):
        """
        Samples after computing node coupling from history for the time length of the sampling period and for
        - state variables of indices voi,
        - and region nodes of indices proxy_inds,
        if integration step corresponds to sampling period.
        """
        # If step % istep is 0...
        if step % self.istep == 0:
            # ...compute the first ...
            start_step = step - self.istep + 1
            # ...and last step of the sampling period...
            end_step = step + 1
            # ...and loop through the sampling period by computing node coupling from history:
            output = []
            for _step in range(start_step, end_step):
                output.append(self.coupling(_step, self.history)[self.voi][:, self.proxy_inds])
            # Finally, form the time vector and return the sample:
            return [numpy.arange(start_step, end_step) * self.dt,
                    numpy.array(output)]


class CosimCouplingMonitorToFile(CosimCouplingMonitor):

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state):
        raise NotImplementedError


class CosimCouplingMonitorToMPI(CosimCouplingMonitor):

    path = Attr(field_type=os.PathLike, required=False, default="")

    def record(self, step, state):
        raise NotImplementedError
