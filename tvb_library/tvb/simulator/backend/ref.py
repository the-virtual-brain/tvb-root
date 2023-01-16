# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
This module provides a reference backend implemented with NumPy.

"""
import numpy
import numpy as np
from .base import BaseBackend


class RefBase:
    "Base methods for reference NumPy backend"

    @staticmethod
    def evaluate(expr, global_dict=None, out=None):
        "Evaluate expression as in numexpr.evaluate."
        ns = {}
        ns.update(np.__dict__)
        if global_dict is None:
            import inspect
            frame = inspect.getcurrentframe()
            global_dict = frame.f_back.f_locals
        ns.update(global_dict)
        val = eval(expr, ns)
        if out is not None:
            out[:] = val
        return val

    try:
        import numexpr
        evaluate = staticmethod(numexpr.evaluate)
    except ImportError:
        pass

    # from tvb.sim.common
    def _add_at(dest, map, src):
        "workaround lack of ufunc at method for older NumPy versions"
        for i in numpy.unique(map):
            dest[i] += src[i == map].sum(axis=0)
        return dest

    try:
        add_at = staticmethod(numpy.add.at)
    except AttributeError:
        add_at = staticmethod(_add_at)

    @staticmethod
    def linear_interp1d(start_time, end_time, start_value, end_value, interp_point):
        """
        performs a one dimensional linear interpolation using two
        timepoints (start_time, end_time) for two floating point (possibly
        NumPy arrays) states (start_value, end_value) to return state at timepoint
        start_time < interp_point < end_time.

        """

        mean = (end_value - start_value) / (end_time - start_time)
        return start_value + (interp_point - start_time) * mean

    @staticmethod
    def heaviside(array):
        """
        heaviside() returns 1 if argument > 0, 0 otherwise.

        The copy operation here is necessary to ensure that the
        array passed in is not modified.

        """

        if type(array) == numpy.float64:
            return 0.0 if array < 0.0 else 1.0
        else:
            ret = array.copy()
            ret[array < 0.0] = 0.0
            ret[array > 0.0] = 1.0
            return ret

    @staticmethod
    def iround(x):
        """
        iround(number) -> integer
        Trying to get a stable and portable result when rounding a number to the
        nearest integer.

        NOTE: I'm introducing this because of the unstability we found when
        using int(round()). Should use always the same rounding strategy.

        Try :
        >>> int(4.999999999999999)
        4
        >>> int(4.99999999999999999999)
        5

        """
        y = round(x) - .5
        return int(y) + (y > 0)


class RefSurface(RefBase):
    "Surface/field-related methods."

    @staticmethod
    def surface_state_to_rois(_regmap, n_reg, state):
        region_state = numpy.zeros((n_reg, state.shape[0], state.shape[2]))  # temp (node, cvar, mode)
        # TODO how to handle this?
        RefBase.add_at(region_state, _regmap, state.transpose((1, 0, 2)))  # sum within region
        region_state /= numpy.bincount(_regmap).reshape((-1, 1, 1))  # div by n node in region
        # TODO out= argument to avoid alloc
        state = region_state.transpose((1, 0, 2))  # (cvar, node, mode)
        return state


class ReferenceBackend(BaseBackend, RefSurface):
    "Base reference backend, implemented in readable NumPy."
