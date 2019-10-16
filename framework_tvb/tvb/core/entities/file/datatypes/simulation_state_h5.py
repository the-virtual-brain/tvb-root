from tvb.basic.neotraits.api import Attr, Int, NArray, Float
from tvb.core.neotraits.h5 import H5File, DataSet, Scalar
from tvb.simulator.integrators import IntegratorStochastic
from tvb.simulator.simulator import Simulator



class SimulationStateH5(H5File):

    def __init__(self, path):
        super(SimulationStateH5, self).__init__(path)
        self.history = DataSet(NArray(), self, name='history')
        self.current_state = DataSet(NArray(), self, name='current_state')
        self.current_step = Scalar(Int(), self, name='current_step')

        for i in range(1, 16):
            setattr(self, 'monitor_stock_%i' % i, DataSet(NArray(), self, name='monitor_stock_%i' % i))

        self.integrator_noise_rng_state_algo = Scalar(Attr(str), self, name='integrator_noise_rng_state_algo')
        self.integrator_noise_rng_state_keys = DataSet(NArray(dtype='uint32'), self, name='integrator_noise_rng_state_keys')
        self.integrator_noise_rng_state_pos = Scalar(Int(), self, name='integrator_noise_rng_state_pos')
        self.integrator_noise_rng_state_has_gauss = Scalar(Int(), self, name='integrator_noise_rng_state_has_gauss')
        self.integrator_noise_rng_state_cached_gauss = Scalar(Float(), self, name='integrator_noise_rng_state_cached_gauss')


    def store(self, simulator, scalars_only=False):
        # type: (Simulator, bool) -> None
        self.history.store(simulator.history.buffer.copy())
        self.current_step.store(simulator.current_step)
        self.current_state.store(simulator.current_state)

        for i, monitor in enumerate(simulator.monitors):
            field_name = "monitor_stock_" + str(i + 1)

            getattr(self, field_name).store(monitor._stock)

            # if hasattr(monitor, "_ui_name"):
            #     self.set_metadata({'monitor_name': monitor._ui_name}, field_name)
            # else:
            #     self.set_metadata({'monitor_name': monitor.__class__.__name__}, field_name)

        if isinstance(simulator.integrator, IntegratorStochastic):
            rng_state = simulator.integrator.noise.random_stream.get_state()
            self.integrator_noise_rng_state_algo.store(rng_state[0])
            self.integrator_noise_rng_state_keys.store(rng_state[1])
            self.integrator_noise_rng_state_pos.store(rng_state[2])
            self.integrator_noise_rng_state_has_gauss.store(rng_state[3])
            self.integrator_noise_rng_state_cached_gauss.store(rng_state[4])


    def load_into(self, simulator):
        """
        Populate a Simulator object from current stored-state.
        """
        simulator.history.initialize(self.history.load())
        simulator.current_step = self.current_step.load()
        simulator.current_state = self.current_state.load()

        for i, monitor in enumerate(simulator.monitors):
            monitor._stock = getattr(self, "monitor_stock_" + str(i + 1)).load()

        if self.integrator_noise_rng_state_algo is not None:
            rng_state = (
                self.integrator_noise_rng_state_algo.load(),
                self.integrator_noise_rng_state_keys.load(),
                self.integrator_noise_rng_state_pos.load(),
                self.integrator_noise_rng_state_has_gauss.load(),
                self.integrator_noise_rng_state_cached_gauss.load()
            )
            simulator.integrator.noise.random_stream.set_state(rng_state)

