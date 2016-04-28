# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Adapter that uses the traits module to generate interfaces for ... Analyzer.

.. moduleauthor:: Francesca Melozzi <france.melozzi@gmail.com>
.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""
import numpy as np
import tvb.datatypes.time_series as time_series
import tvb.datatypes.fcd as fcd
import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic
import tvb.basic.traits.util as util
from tvb.basic.logger.builder import get_logger


LOG = get_logger(__name__)

class FcdCalculator(core.Type):
    """
    Compute the the fcd of the timeseries

    Return a Fcd datatype, whose values of are between -1
    and 1, inclusive.

    """

    time_series = time_series.TimeSeries(
        label = "Time Series",
        required = True,
        doc = """The time-series for which the fcd matrices are calculated.""")

    sw = basic.Float(
        label="Sliding window length (ms)",
        default=120000,
        doc="""Sliding window length (ms)""")

    sp = basic.Float(
        label="Spanning between two consecutive sliding window (ms)",
        default=2000,
        doc="""Spanning between two consecutive sliding window (ms)""")

    def evaluate(self):
        cls_attr_name = self.__class__.__name__ + ".time_series"
        self.time_series.trait["data"].log_debug(owner=cls_attr_name)

        sp=float(self.sp)
        sw=float(self.sw)
        # Pass sp and sw in the right reference (means considering the sample period)
        #! the sample period is in second
        sp = sp  / self.time_series.sample_period
        sw = sw / self.time_series.sample_period
        # (fcd_points, fcd_points, state-variables, modes)
        input_shape = self.time_series.read_data_shape()
        result_shape = self.result_shape(input_shape)

        result = np.zeros(result_shape)

        FCstream = {}  # Here I will put before the stram of the FC
        start = -sp  # in order to well initialize the first starting point of the FC stream
        # One fcd matrix, for each state-var & mode.
        for mode in range(result_shape[3]):
            for var in range(result_shape[2]):
                for nfcd in range(result_shape[0]):
                    start += sp
                    current_slice = tuple([slice(int(start), int(start+sw) + 1), slice(var, var + 1),
                                           slice(input_shape[2]), slice(mode, mode + 1)])
                    data = self.time_series.read_data_slice(current_slice).squeeze()
                    #data = self.time_series.data[start:(start + sw), var, :, mode]
                    FC = np.corrcoef(data.T)
                    Triangular = np.triu_indices(len(FC),
                                                 1)  # I organize the triangular part of the FC as a vector excluding the diagonal (always ones)
                    FCstream[nfcd] = FC[Triangular]
                for i in range(result_shape[0]):
                    j = i
                    while j < result_shape[0]:
                        fci = FCstream[i]
                        fcj = FCstream[j]
                        result[i, j, var, mode] = np.corrcoef(fci, fcj)[0, 1]
                        result[j, i, var, mode] = result[i, j, var, mode]
                        j += 1

        util.log_debug_array(LOG, result, "result")

        # fcd_matrix = fcd.Fcd(source=self.time_series,
        #                      sp=self.sp,
        #                      sw=self.sw,
        #                      array_data=result,
        #                      use_storage=False)
        return result


    def result_shape(self, input_shape):
        """Returns the shape of the main result of ...."""
        sw=float(self.sw)
        sp=float(self.sp)
        sp = (sp)  / (self.time_series.sample_period)
        sw = (sw) / (self.time_series.sample_period)
        fcd_points = int((input_shape[0] - sw) / sp)
        result_shape = (fcd_points, fcd_points,
                        input_shape[1], input_shape[3])
        return result_shape


    def result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the main result of .
        """
        result_size = np.sum(map(np.prod, self.result_shape(input_shape))) * 8.0  # Bytes
        return result_size


    def extended_result_size(self, input_shape):
        """
        Returns the storage size in Bytes of the extended result of the ....
        That is, it includes storage of the evaluated ... attributes
        such as ..., etc.
        """
        extend_size = self.result_size(input_shape)  # Currently no derived attributes.
        return extend_size



