"""
This module provides a reference backend implemented with NumPy.

"""
import numpy
import numpy as np
from Cython.Includes import numpy

from tvb.simulator.common import numpy_add_at

from .base import BaseBackend


class SurfaceRefBackend:
    "Surface/field-related methods."

    @staticmethod
    def surface_state_to_rois(_regmap, n_reg, state):
        region_state = numpy.zeros((n_reg, state.shape[0], state.shape[2]))  # temp (node, cvar, mode)
        numpy_add_at(region_state, _regmap, state.transpose((1, 0, 2)))  # sum within region
        region_state /= numpy.bincount(_regmap).reshape((-1, 1, 1))  # div by n node in region
        # TODO out= argument to avoid alloc
        state = region_state.transpose((1, 0, 2))  # (cvar, node, mode)
        return state


class ReferenceBackend(BaseBackend, SurfaceRefBackend):
    "Base reference backend, implemented in readable NumPy."
    pass