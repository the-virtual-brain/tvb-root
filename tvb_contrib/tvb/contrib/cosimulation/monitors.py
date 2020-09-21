# -*- coding: utf-8 -*-
import os
import abc
import numpy
from tvb.basic.neotraits.api import Attr, NArray
from tvb.simulator.common import iround
from tvb.simulator.monitors import Monitor
from tvb.simulator.history import BaseHistory, SparseHistory
from tvb.simulator.coupling import Coupling, Linear


class CosimMonitor(Monitor):
    """
        Monitors the value for the model's (potentially coupling) variable/s of interest
        over the TVB proxy nodes at each sampling period. Time steps that are not modulo ``istep``
        are stored temporarily in the ``_stock`` attribute and then that temporary
        store is returned when time step is modulo ``istep``.
        The output can be returned like any other TVB monitor or written to file or to some MPI channel,
        depending on user input to record_to and record_path attributes.

    """

    record_to = Attr(field_type=str, required=True,
                     default="memory")  # other options are "file", "mpi"

    record_path = Attr(field_type=os.PathLike, required=False, default="")

    proxy_inds = NArray(
        dtype=numpy.int,
        label="Indices of TVB proxy nodes",
        required=True,
    )

    def __init__(self, **kwargs):
        super(CosimMonitor, self).__init__(**kwargs)

    @abc.abstractmethod
    def record_to_file(self, step, state):
        """Write to some file of a given format
           return 0 for success or None"""
        pass

    @abc.abstractmethod
    def record_to_mpi(self, step, state):
        """Write to some MPI channel of a given type
           return 0 for success or None"""
        pass

    def configure_record(self):
        # TODO: add a check for the record_path if record_to != "memory", and possibly for creating a default path/file
        self.record_to = self.record_to.lower()
        if self.record_to == "file":
            self.record = self.record_to_file
        elif self.record_to == "mpi":
            self.record = self.record_to_mpi()

    def configure_stock_size(self, simulator):
        self.number_of_proxy_nodes = len(self.proxy_inds)
        stock_size = (self.istep, self.voi.shape[0],
                      self.number_of_proxy_nodes,
                      simulator.model.number_of_modes)
        self.log.debug("CosimMonitor stock_size is %s" % (str(stock_size), ))
        self._stock = numpy.zeros(stock_size)

    def config_for_sim(self, simulator):
        super(CosimMonitor, self).config_for_sim(simulator)


class CosimStateMonitor(CosimMonitor):
    """
        Monitors the value for the model's variable/s of interest over all
        the nodes at each sampling period. Time steps that are not modulo ``istep``
        are stored temporarily in the ``_stock`` attribute and then that temporary
        store is returned when time step is modulo ``istep``.
        The output can be returned like any other TVB monitor or written to file or to some MPI channel,
        depending on user input to record_to and record_path attributes.

    """

    def __init__(self, **kwargs):
        super(CosimStateMonitor, self).__init__(**kwargs)

    def config_for_sim(self, simulator):
        super(CosimStateMonitor, self).config_for_sim(simulator)
        self.configure_stock_size()
        self.configure_record()

    def sample(self, step, state):
        """
        Records if integration step corresponds to sampling period, Otherwise
        just update the monitor's stock. When the step corresponds to the sample/synchronization
        period, the ``_stock`` is returned.
        """
        self._stock[((step % self.istep) - 1), :, :] = state[self.voi][:, self.proxy_inds]
        if step % self.istep == 0:
            time = numpy.arange(step - self.istep + 1, step + 1) * self.dt
            return [time, self._stock]


class CosimHistoryMonitor(Monitor):
    """
        Monitors the history for the model's coupling variable/s of interest
        over TVB proxy nodes at each sampling period. Time steps that are not modulo ``istep``
        are stored temporarily in the ``_stock`` attribute and then that temporary
        store is returned when time step is modulo ``istep``.
        The output can be returned like any other TVB monitor or written to file or to some MPI channel,
        depending on user input to record_to and record_path attributes.

    """

    history = Attr(
        field_type=BaseHistory,
        label="Simulator history",
        default=SparseHistory(),
        required=True,
        doc="""A tvb.simulator.history""")

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator.

                Grab the Simulator's integration step size. Set the monitor's variables
                of interest based on the Monitor's 'variables_of_interest' attribute, if
                it was specified, otherwise use the 'variables_of_interest' and cvar specified
                for the Model. Calculate the number of integration steps (isteps)
                between returns by the record method. This method is called from within
                the the Simulator's configure() method.
        """
        self.history = simulator.history
        self.dt = simulator.integrator.dt
        self.istep = iround(self.period / self.dt)
        self.voi = self.variables_of_interest
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.cvar)]
        self.configure_stock_size(simulator)
        self.configure_output_recorder()

    def sample(self, step, state):
        """
        Records from history current state,
        when the step corresponds to the sample/synchronization period.
        """
        if step % self.istep == 0:
            start_step = step - self.istep + 1
            end_step = step + 1
            output = []
            for _step in range(start_step, end_step):
                output.append(self.history.query(_step)[0][self.voi][:, self.proxy_inds])
            return [numpy.arange(start_step, end_step) * self.dt, numpy.array(output)]


class CosimCouplingMonitor(Monitor):
    """
        Monitors the current node coupling for the model's coupling variable/s of interest
        over TVB proxy nodes at each sampling period.
        The output can be returned like any other TVB monitor or written to file or to some MPI channel,
        depending on user input to record_to and record_path attributes.

    """

    history = Attr(
        field_type=BaseHistory,
        label="Simulator's history",
        default=SparseHistory(),
        required=True,
        doc="""A tvb.simulator.history""")

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
        """Configure monitor for given simulator.

                Grab the Simulator's integration step size. Set the monitor's variables
                of interest based on the Monitor's 'variables_of_interest' attribute, if
                it was specified, otherwise use the 'variables_of_interest' and cvar specified
                for the Model. Calculate the number of integration steps (isteps)
                between returns by the record method. This method is called from within
                the the Simulator's configure() method.
        """
        self.history = simulator.history
        self.coupling = simulator.coupling
        self.dt = simulator.integrator.dt
        self.period = self.dt
        self.istep = 1.0
        self.voi = self.variables_of_interest
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.cvar)]
        self.configure_stock_size(simulator)
        self.configure_output_recorder()

    def sample(self, step, state):
        """
        Records from current node coupling.
        """
        return [step * self.dt, self.coupling(step, self.history)[self.voi][:, self.proxy_inds]]
