from tvb.core.neotraits.h5 import H5File, DataSet, Scalar
from tvb.datatypes.simulation_state import SimulationState


class SimulationStateH5(H5File):

    def __init__(self, path):
        super(SimulationStateH5, self).__init__(path)
        self.history = DataSet(SimulationState.history, self)
        self.current_state = DataSet(SimulationState.current_state, self)
        self.current_step = Scalar(SimulationState.current_step, self)
        self.monitor_stock_1 = DataSet(SimulationState.monitor_stock_1, self)
        self.monitor_stock_2 = DataSet(SimulationState.monitor_stock_2, self)
        self.monitor_stock_3 = DataSet(SimulationState.monitor_stock_3, self)
        self.monitor_stock_4 = DataSet(SimulationState.monitor_stock_4, self)
        self.monitor_stock_5 = DataSet(SimulationState.monitor_stock_5, self)
        self.monitor_stock_6 = DataSet(SimulationState.monitor_stock_6, self)
        self.monitor_stock_7 = DataSet(SimulationState.monitor_stock_7, self)
        self.monitor_stock_8 = DataSet(SimulationState.monitor_stock_8, self)
        self.monitor_stock_9 = DataSet(SimulationState.monitor_stock_9, self)
        self.monitor_stock_10 = DataSet(SimulationState.monitor_stock_10, self)
        self.monitor_stock_11 = DataSet(SimulationState.monitor_stock_11, self)
        self.monitor_stock_12 = DataSet(SimulationState.monitor_stock_12, self)
        self.monitor_stock_13 = DataSet(SimulationState.monitor_stock_13, self)
        self.monitor_stock_14 = DataSet(SimulationState.monitor_stock_14, self)
        self.monitor_stock_15 = DataSet(SimulationState.monitor_stock_15, self)
