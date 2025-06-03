import math
import numpy as np
from typing import List

import tvb.basic.neotraits.api as t
from tvb.simulator.monitors import Monitor
from .network import NetworkSet


class Simulator(t.HasTraits):
    """Simulator for hybrid network models.
    
    The Simulator class manages the simulation of a hybrid network model,
    including configuration of monitors and time stepping.
    
    Attributes
    ----------
    nets : NetworkSet
        The network model to simulate
    monitors : list
        List of monitors for recording data
    simulation_length : float
        Total simulation time in milliseconds
    """

    nets: NetworkSet = t.Attr(NetworkSet)
    monitors: List[Monitor] = t.List(of=Monitor)
    simulation_length: float = t.Float()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_dts()
        self.validate_vois()

    def validate_vois(self):
        """Validate variables of interest across subnetworks."""
        if len(self.monitors) == 0:
            return
        nv0 = self.nets.subnets[0].model.variables_of_interest
        for sn in self.nets.subnets[1:]:
            msg = 'Variables of interest must have same size on all models.'
            assert len(nv0) == len(sn.model.variables_of_interest), msg
        for monitor in self.monitors:
            num_nodes = sum([sn.nnodes for sn in self.nets.subnets])
            monitor._config_stock(len(nv0), num_nodes, 1)

    def validate_dts(self):
        """Validate integration time steps across subnetworks."""
        self._dt0 = self.nets.subnets[0].scheme.dt
        for sn in self.nets.subnets[1:]:
            assert self._dt0 == sn.scheme.dt
        for monitor in self.monitors:
            monitor: Monitor
            monitor._config_dt(self._dt0)
            monitor.voi = slice(None)  # all vars

    def configure(self):
        """Configure the simulator and its monitors."""
        for subnet in self.nets.subnets:
            # Subnetwork.configure() also configures its IntraProjections
            subnet.configure() 
            # Configure recorders in each subnetwork
            for recorder in subnet.monitors:
                recorder.configure(self.simulation_length)
        
        # Configure the NetworkSet, which configures InterProjections
        self.nets.configure()

    def run(self):
        """Run the simulation.
        
        Returns
        -------
        list
            List of (time, data) tuples for each monitor, if any
        """
        # Configure if not already done
        if not hasattr(self, '_dt0'):
            self.configure()
        mts = [[] for _ in self.monitors]
        mxs = [[] for _ in self.monitors]
        x = self.nets.zero_states()
        stop = int(math.ceil(self.simulation_length / self._dt0))
        for step in range(0, stop):
            x = self.nets.step(step, x)
            if self.monitors:
                ox = self.nets.observe(x, flat=True)
                for mt, mx, mon in zip(mts, mxs, self.monitors):
                    maybe_tx = mon.record(step, ox)
                    if maybe_tx is not None:
                        mt.append(maybe_tx[0])
                        mx.append(maybe_tx[1])
        return [(np.array(t), np.array(x)) for t, x in zip(mts, mxs)]
