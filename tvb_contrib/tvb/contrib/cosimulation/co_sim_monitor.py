
import numpy
import abc
from tvb.simulator import coupling, monitors
from tvb.basic.neotraits.api import Attr, NArray

class Co_sim_Monitor(monitors.Monitor):
    """
    Abstract base class for monitor implementations.
    """
    variables_of_interest = NArray(
        dtype=int,
        label="Model variables to watch",  # order=11,
        doc=("Indices of model's variables of interest (VOI) that this monitor should record. "
             "Note that the indices should start at zero, so that if a model offers VOIs V, W and "
             "V+W, and W is selected, and this monitor should record W, then the correct index is 0."),
        required=False)

    dt = None
    voi = None

    def __str__(self):
        clsname = self.__class__.__name__
        return '%s(voi=%s)' % (clsname, self.variables_of_interest.tolist())

    def _config_vois(self, simulator):
        self.voi = self.variables_of_interest
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.variables_of_interest)]

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator.

        Grab the Simulator's integration step size. Set the monitor's variables
        of interest based on the Monitor's 'variables_of_interest' attribute, if
        it was specified, otherwise use the 'variables_of_interest' specified
        for the Model. Calculate the number of integration steps (isteps)
        between returns by the record method. This method is called from within
        the the Simulator's configure() method.

        """
        self._config_vois(simulator)
        self.dt = simulator.integrator.dt

    @abc.abstractmethod
    def sample(self, start_step, n_steps,history_incomplete,history_delayed):
        """
        This method provides monitor output, and should be overridden by subclasses.

        """

class Raw_delayed(Co_sim_Monitor):
    """
    A monitor that records the output raw data from a tvb simulation:
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Raw recording co_sim"

    variables_of_interest = NArray(
        dtype=int,
        label="Raw Monitor sees all!!! Resistance is futile...",
        required=False)

    def _config_vois(self, simulator):
        self.voi = numpy.arange(len(simulator.model.variables_of_interest))

    def sample(self, start_step, n_steps,history_incomplete,history_delayed):
        times = []
        values = []
        for step in range(start_step, start_step + n_steps):
            times.append(step*self.dt)
            values.append(history_delayed.query_sparse(step))
        return [numpy.array(times),numpy.array(values)]

class Raw_incomplete(Co_sim_Monitor):
    """
    A monitor that records the output raw data from a tvb simulation:
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Raw recording co_sim"

    variables_of_interest = NArray(
        dtype=int,
        label="Raw Monitor sees all!!! Resistance is futile...",
        required=False)

    def _config_vois(self, simulator):
        self.voi = numpy.arange(len(simulator.model.variables_of_interest))

    def sample(self, start_step, n_steps,history_incomplete,history_delayed):
        times = []
        values = []
        for step in range(start_step, start_step + n_steps):
            times.append(step*self.dt)
            values.append(history_incomplete.query_state(step))
        return [numpy.array(times),numpy.array(values)]

class Coupling(Co_sim_Monitor):
    """
    A monitor that records the output raw data from a tvb simulation:
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based
        - all the integration time steps

    """

    _ui_name = "Raw recording co_sim"

    variables_of_interest = NArray(
        dtype=int,
        label="See only the coupling variable!!! Resistance is futile...",
        required=False)

    coupling = Attr(
        field_type=coupling.Coupling,
        label="Long-range coupling function",
        default=coupling.Linear(),
        required=True,
        doc="""The coupling function is applied to the activity propagated
        between regions by the ``Long-range connectivity`` before it enters the local
        dynamic equations of the Model. Its primary purpose is to 'rescale' the
        incoming activity to a level appropriate to Model.""")

    def _config_vois(self, simulator):
        self.voi = simulator.model.cvar

    def sample(self, start_step, n_steps,history_incomplete,history_delayed):
        times = []
        couplings = []
        for step in range(start_step, start_step + n_steps):
            times.append(step*self.dt)
            couplings.append(self.coupling(step, history_delayed))
        return [numpy.array(times),numpy.array(couplings)]