import collections
import numpy as np
import tvb.basic.neotraits.api as t
from .subnetwork import Subnetwork
from .inter_projection import InterProjection


class NetworkSet(t.HasTraits):
    """A collection of subnetworks and their projections.

    A NetworkSet represents a complete hybrid model by collecting subnetworks
    and defining how they interact through projections.

    Attributes
    ----------
    subnets : list
        List of subnetworks
    projections : list
        List of projections between subnetworks
    States : namedtuple
        Named tuple class for accessing subnetwork states
    """

    subnets: [Subnetwork] = t.List(of=Subnetwork)
    projections: [Subnetwork] = t.List(of=InterProjection)
    # NOTE dynamically generated namedtuple based on subnetworks
    States: collections.namedtuple = None
    # TODO consider typing this as a tuple[ndarray[float]]?

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.States = collections.namedtuple(
            'States',
            ' '.join([_.name for _ in self.subnets]))
        self.States.shape = property(lambda self: [_.shape for _ in self])

    def configure(self):
        """Configure all inter-subnetwork projections within the network set."""
        for p in self.projections: # These are InterProjection instances
            p.configure()
        return self

    def zero_states(self, initial_states: list[np.ndarray] = None) -> States:
        """Create zero states or use provided initial states for all subnetworks.

        Parameters
        ----------
        initial_states : list[np.ndarray], optional
            A list of initial state arrays, one for each subnetwork.
            If None, zero states are created.

        Returns
        -------
        States
            Named tuple containing initial states for each subnetwork
        """
        if initial_states is not None:
            if len(initial_states) != len(self.subnets):
                raise ValueError(
                    f"Number of initial_states ({len(initial_states)}) "
                    f"must match number of subnetworks ({len(self.subnets)})."
                )
            # Ensure each initial_state has the correct shape for its subnetwork
            for i, sn in enumerate(self.subnets):
                expected_shape = (sn.model.nvar, ) + sn.var_shape
                if initial_states[i].shape != expected_shape:
                    raise ValueError(
                        f"Initial state for subnetwork '{sn.name}' has shape {initial_states[i].shape}, "
                        f"expected {expected_shape}."
                    )
            return self.States(*initial_states)
        else:
            return self.States(*[_.zero_states() for _ in self.subnets])

    def zero_cvars(self) -> States:
        """Create zero coupling variables for all subnetworks.

        Returns
        -------
        States
            Named tuple containing zero coupling variables for each subnetwork
        """
        return self.States(*[_.zero_cvars() for _ in self.subnets])

    def observe(self, states: States, flat=False) -> np.ndarray:
        """Compute observations across all subnetworks.

        Parameters
        ----------
        states : States
            Current states of all subnetworks
        flat : bool
            If True, flatten observations into a single array

        Returns
        -------
        ndarray
            Array of observations
        """
        obs = self.States(*[sn.model.observe(x).sum(axis=-1)[..., None]
                          for sn, x in zip(self.subnets, states)])
        if flat:
            obs = np.hstack(obs)
        return obs

    def cfun(self, step: int, eff: States) -> States:
        """Compute coupling between subnetworks.

        Parameters
        ----------
        step : int
            Current simulation step index.
        eff : States
            Current states of all subnetworks

        Returns
        -------
        States
            Coupling variables for each subnetwork
        """
        aff = self.zero_cvars()
        for p in self.projections:
            tgt = getattr(aff, p.target.name)
            src = getattr(eff, p.source.name)
            p.update_buffer(src, step)
            p.apply(tgt, step)
        return aff

    def step(self, step, xs: States) -> States:
        """Take a single integration step for all subnetworks.

        Parameters
        ----------
        step : int
            Current simulation step
        xs : States
            Current states of all subnetworks

        Returns
        -------
        States
            Next states after integration
        """
        cs = self.cfun(step, xs)
        nxs = self.zero_states()
        for sn, nx, x, c in zip(self.subnets, nxs, xs, cs):
            nx[:] = sn.step(step, x, c)
        return nxs
