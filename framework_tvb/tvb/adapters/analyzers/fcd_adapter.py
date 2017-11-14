# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Adapter that uses the traits module to generate interfaces for FCD Analyzer.

.. moduleauthor:: Francesca Melozzi <france.melozzi@gmail.com>
.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import numpy as np
from tvb.analyzers.fcd_matrix import FcdCalculator
from tvb.basic.traits.util import log_debug_array
from tvb.basic.filters.chain import FilterChain
from tvb.core.adapters.abcadapter import ABCAsynchronous
from tvb.core.adapters.exceptions import LaunchException
from tvb.datatypes.fcd import Fcd
from tvb.datatypes.graph import ConnectivityMeasure



class FunctionalConnectivityDynamicsAdapter(ABCAsynchronous):
    """ TVB adapter for calling the Pearson CrossCorrelation algorithm. """

    _ui_name = "FCD matrix"
    _ui_description = "Functional Connectivity Dynamics metric"
    _ui_subsection = "fcd_calculator"


    def get_input_tree(self):
        """
        Return a list of lists describing the interface to the analyzer. This
        is used by the GUI to generate the menus and fields necessary for
        defining a simulation.
        """
        algorithm = FcdCalculator()
        algorithm.trait.bound = self.INTERFACE_ATTRIBUTES_ONLY
        tree = algorithm.interface[self.INTERFACE_ATTRIBUTES]
        tree[0]['conditions'] = FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                            operations=["=="], values=[4])
        return tree


    def get_output(self):
        return [Fcd, ConnectivityMeasure]


    def configure(self, time_series, sw, sp):
        """
        Store the input shape to be later used to estimate memory usage. Also create the algorithm instance.

        :param time_series: the input time-series for which fcd matrix should be computed
        :param sw: length of the sliding window
        :param sp: spanning time: distance between two consecutive sliding window
        """
        """
        Store the input shape to be later used to estimate memory usage. Also create the algorithm instance.
        """

        self.input_shape = time_series.read_data_shape()
        log_debug_array(self.log, time_series, "time_series")
        actual_sp = float(sp) / time_series.sample_period
        actual_sw = float(sw) / time_series.sample_period
        actual_ts_length = self.input_shape[0]

        if actual_sw >= actual_ts_length or actual_sp >= actual_ts_length or actual_sp >= actual_sw:
            raise LaunchException(
                "Spanning (Sp) and Sliding (Sw) window size parameters need to be less than the TS length, "
                "and Sp < Sw. After calibration with sampling period, current values are: Sp=%d, Sw=%d, Ts=%d). "
                "Please configure valid input parameters." % (actual_sp, actual_sw, actual_ts_length))

        # -------------------- Fill Algorithm for Analysis -------------------##
        self.algorithm = FcdCalculator(time_series=time_series, sw=sw, sp=sp)


    def get_required_memory_size(self, **kwargs):
        # We do not know how much memory is needed.
        return -1


    def get_required_disk_size(self, **kwargs):
        return 0


    def launch(self, time_series, sw, sp):
        """
           Launch algorithm and build results.

           :param time_series: the input time-series for which correlation coefficient should be computed
           :param sw: length of the sliding window
           :param sp: spanning time: distance between two consecutive sliding window
           :returns: the fcd matrix for the given time-series, with that sw and that sp
           :rtype: `Fcd`,`ConnectivityMeasure` 
        """

        result = []  # where fcd, fcd_segmented (eventually), and connectivity measures will be stored

        [fcd, fcd_segmented, eigvect_dict, eigval_dict, Connectivity] = self.algorithm.evaluate()

        # Create a Fcd dataType object.
        result_fcd = Fcd(storage_path=self.storage_path, source=time_series, sw=sw, sp=sp)
        result_fcd.array_data = fcd
        result.append(result_fcd)

        if np.amax(fcd_segmented) == 1.1:
            result_fcd_segmented = Fcd(storage_path=self.storage_path, source=time_series, sw=sw, sp=sp)
            result_fcd_segmented.array_data = fcd_segmented
            result.append(result_fcd_segmented)
        for mode in eigvect_dict.keys():
            for var in eigvect_dict[mode].keys():
                for ep in eigvect_dict[mode][var].keys():
                    for eig in range(3):
                        result_eig = ConnectivityMeasure(storage_path=self.storage_path)
                        result_eig.connectivity = Connectivity
                        result_eig.array_data = eigvect_dict[mode][var][ep][eig]
                        result_eig.title = "Epoch # %d, \n " \
                                           "eigenvalue = %s,\n " \
                                           "variable = %s,\n " \
                                           "mode = %s." % (ep, eigval_dict[mode][var][ep][eig], var, mode)
                        result.append(result_eig)
        return result
