"""
Tests for the Subnetwork and Stim classes.
"""

import numpy as np
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.patterns import StimuliRegion
from tvb.simulator.models import JansenRit
from .test_base import BaseHybridTest


class TestSubnetwork(BaseHybridTest):
    """Tests for the Subnetwork class."""

    def test_subnetwork(self):
        """Test subnetwork state and coupling variable shapes"""
        conn, ix, c, t, a, nets = self.setup()
        self.assert_equal((6, (ix == a.CORTEX).sum(), 1), c.zero_states().shape)
        self.assert_equal((2, (ix == a.CORTEX).sum(), 1), c.zero_cvars().shape)
        self.assert_equal((4, (ix == a.THALAMUS).sum(), 3), t.zero_states().shape)


class TestStim(BaseHybridTest):
    """Tests for the Stim class."""

    def test_stim(self):
        """Test stimulus application to network"""
        conn = Connectivity.from_file()
        nn = conn.weights.shape[0]
        conn.configure()
        
        class MyStim(StimuliRegion):
            def __call__(self, t):
                return np.random.randn(self.connectivity.weights.shape[0])
                
        stim = MyStim(connectivity=conn)
        I = stim(5)
        self.assert_equal((nn,), I.shape)
        model = JansenRit()
        model.configure()
        # TODO: Add more stimulus tests 