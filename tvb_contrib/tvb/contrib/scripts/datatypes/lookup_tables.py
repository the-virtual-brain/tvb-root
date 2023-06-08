# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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

The LookUpTable datatype.

.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>

"""

import numpy
from tvb.basic.readers import try_get_absolute_path
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Int, Float


class LookUpTable(HasTraits):
    """
    Lookup Tables for storing pre-computed functions.
    Specific table subclasses are implemented below.
    """

    xmin = Float(
        label="x-min",
        doc="""Minimum value""")

    xmax = Float(
        label="x-max",
        doc="""Maximum value""")

    data = NArray(
        label="data",
        doc="""Tabulated values""")

    number_of_values = Int(
        label="Number of values",
        default=0,
        doc="""The number of values in the table """)

    df = NArray(
        label="df",
        doc=""".""")

    dx = Float(
        label="dx",
        default=0.,
        doc="""Tabulation step""")

    invdx = Float(
        label="invdx",
        default=0.,
        doc=""".""")

    @staticmethod
    def populate_table(result, source_file):
        source_full_path = try_get_absolute_path("tvb_data.tables", source_file)
        zip_data = numpy.load(source_full_path)

        result.df = zip_data['df']
        result.xmin = zip_data['min_max'][0]
        result.xmax = zip_data['min_max'][1]
        result.data = zip_data['f']
        return result

    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialization.
        """
        super(LookUpTable, self).configure()

        # Check if dx and invdx have been computed
        if self.number_of_values == 0:
            self.number_of_values = self.data.shape[0]

        self.compute_search_indices()

    def summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this dataType, if any ...
        """
        return {"Number of values": self.number_of_values}

    def compute_search_indices(self):
        """
        ...
        """
        self.dx = ((self.xmax - self.xmin) / (self.number_of_values) - 1)
        self.invdx = 1 / self.dx

    def search_value(self, val):
        """
        Search a value in this look up table
        """

        if self.xmin:
            y = val - self.xmin
        else:
            y = val

        ind = numpy.array(y * self.invdx, dtype=int)

        try:
            return self.data[ind] + self.df[ind] * (y - ind * self.dx)
        except IndexError:  # out of bounds
            return numpy.NaN
            # NOTE: not sure if we should return a NaN or make val = self.max
            # At the moment, we force the input values to be within a known range


class PsiTable(LookUpTable):
    """
    Look up table containing the values of a function representing the time-averaged gating variable
    :math:`\\psi(\\nu)` as a function of the presynaptic rates :math:`\\nu`

    """

    @staticmethod
    def from_file(source_file="psi.npz"):
        return LookUpTable.populate_table(PsiTable(), source_file)


class NerfTable(LookUpTable):
    """
    Look up table containing the values of Nerf integral within the :math:`\\phi`
    function that describes how the discharge rate vary as a function of parameters
    defining the statistical properties of the membrane potential in presence of synaptic inputs.

    """

    @staticmethod
    def from_file(source_file="nerf_int.npz"):
        return LookUpTable.populate_table(NerfTable(), source_file)
