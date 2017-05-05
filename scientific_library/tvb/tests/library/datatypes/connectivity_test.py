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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

if __name__ == "__main__":
    from tvb.tests.library import setup_test_console_env
    setup_test_console_env()

try:
    H5PY_SUPPORT = True
    import h5py as hdf5
except Exception:
    H5PY_SUPPORT = False

import os
import numpy
import unittest
from tvb.datatypes import connectivity
from tvb.tests.library.base_testcase import BaseTestCase


class ConnectivityTest(BaseTestCase):
    """
    Tests the defaults for `tvb.datatypes.connectivity` module.
    """

    def test_connectivity_surrogates(self):
        """
        Create a connectivity using generate_surrogate method and that fields get correctly populated
        """
        conn = connectivity.Connectivity()
        conn.generate_surrogate_connectivity(74)
        conn.configure()
        # Check for value from tvb_data/connectivity/o52r00_irp2008
        self.assertEqual(conn.weights.shape, (74, 74))
        self.assertEqual(conn.weights.max(), 1.0)
        self.assertEqual(conn.weights.min(), 0.0)
        self.assertEqual(conn.tract_lengths.shape, (74, 74))
        self.assertEqual(conn.tract_lengths.max(), 42.0)
        self.assertEqual(conn.tract_lengths.min(), 0.0)
        self.assertEqual(conn.centres.shape, (74, 3))
        self.assertEqual(conn.orientations.shape, (74, 3))
        self.assertEqual(conn.region_labels.shape, (74,))
        self.assertTrue(conn.areas is not None)
        self.assertEqual(conn.undirected, 0)
        self.assertEqual(conn.speed, numpy.array([3.0]))
        self.assertTrue(conn.cortical is not None)
        self.assertTrue(conn.hemispheres is not None)
        self.assertEqual(conn.idelays.shape, (0,))
        self.assertEqual(conn.delays.shape, (74, 74,))
        self.assertEqual(conn.number_of_regions, 74)
        self.assertEqual(conn.number_of_connections, 75)

    
    def test_connectivity_default(self):
        """
        Create a default connectivity and check that everything gets loaded
        """
        conn = connectivity.Connectivity(load_default=True)
        conn.configure()
        n = 76
        # Check for value from tvb_data/connectivity/o52r00_irp2008
        self.assertEqual(conn.weights.shape, (n, n))
        self.assertEqual(conn.weights.max(), 3.0)
        self.assertEqual(conn.weights.min(), 0.0)
        self.assertEqual(conn.tract_lengths.shape, (n, n))
        self.assertEqual(conn.tract_lengths.max(), 153.48574)
        self.assertEqual(conn.tract_lengths.min(), 0.0)
        self.assertEqual(conn.centres.shape, (n, 3))
        self.assertEqual(conn.orientations.shape, (n, 3))
        self.assertEqual(conn.region_labels.shape, (n,))
        self.assertEqual(conn.areas.shape, (n,))
        self.assertEqual(conn.undirected, 0)
        self.assertEqual(conn.speed, numpy.array([3.0]))
        self.assertTrue(conn.cortical.all())
        self.assertEqual(conn.hemispheres.shape, (n,))
        self.assertEqual(conn.idelays.shape, (0,))
        self.assertEqual(conn.delays.shape, (n, n,))
        self.assertEqual(conn.number_of_regions, n)
        self.assertEqual(conn.number_of_connections, 1560)
        self.assertTrue(conn.saved_selection is None)
        self.assertEqual(conn.parent_connectivity, '')
        summary = conn.summary_info
        self.assertEqual(summary['Number of regions'], n)
        ## Call connectivity methods and make sure no compilation or runtime erros
        conn.compute_tract_lengths()
        conn.compute_region_labels()
        conn.try_compute_hemispheres()
        self.assertEqual(conn.scaled_weights().shape, (n, n))
        for mode in ['none', 'tract', 'region']:
            # Empirical seems to fail on some scipy installations. Error is not pinned down
            # so far, it seems to only happen on some machines. Most relevant related to this:
            #
            # http://projects.scipy.org/scipy/ticket/1735
            # http://comments.gmane.org/gmane.comp.python.scientific.devel/14816
            # http://permalink.gmane.org/gmane.comp.python.numeric.general/42082
            #conn.switch_distribution(mode=mode)
            self.assertEqual(conn.scaled_weights(mode=mode).shape, (n, n))


    def test_connectivity_reload(self):
        """
        Reload a connectivity and check that defaults changes accordingly.
        """
        conn = connectivity.Connectivity.from_file("connectivity_192.zip")
        n = 192
        self.assertEqual(conn.weights.shape, (n, n))
        self.assertEqual(conn.weights.max(), 3.0)
        self.assertEqual(conn.weights.min(), 0.0)
        self.assertEqual(conn.tract_lengths.shape, (n, n))
        self.assertEqual(conn.tract_lengths.max(), 142.1458)
        self.assertEqual(conn.tract_lengths.min(), 0.0)
        self.assertEqual(conn.centres.shape, (n, 3))
        self.assertEqual(conn.orientations.shape, (n, 3))
        self.assertEqual(conn.region_labels.shape, (n,))
        self.assertEqual(conn.areas.shape, (n,))
        self.assertEqual(conn.undirected, 0)
        self.assertEqual(conn.speed, numpy.array([3.0]))
        self.assertEqual(conn.hemispheres.shape, (0,))
        self.assertEqual(conn.idelays.shape, (0,))
        self.assertEqual(conn.delays.shape, (0,))
        self.assertEqual(conn.number_of_regions, 0)
        self.assertTrue(conn.saved_selection is None)
        self.assertEqual(conn.parent_connectivity, '')


    @unittest.skipIf(not H5PY_SUPPORT, "HDF5 and H5PY not found on this system")
    def test_connectivity_h5py_reload(self):
        """
        Reload a connectivity and check that defaults changes accordingly.
        """
        h5_full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Edited_Connectivity.h5")
        conn = connectivity.Connectivity.from_file(h5_full_path)
        self.assertEqual(conn.weights.shape, (74, 74))
        self.assertEqual(conn.weights[0][0], 9.0)   # Edit set first weight to 9
        self.assertEqual(conn.weights.max(), 9.0)   # Edit has a weight of value 9
        self.assertEqual(conn.weights.min(), 0.0)
        self.assertEqual(conn.undirected, 0)
        self.assertEqual(conn.speed, numpy.array([3.0]))
        self.assertEqual(conn.hemispheres.shape, (74,))
        self.assertEqual(conn.idelays.shape, (0,))
        self.assertEqual(conn.delays.shape, (0,))
        self.assertEqual(conn.number_of_regions, 0)
        self.assertTrue(conn.saved_selection is None)
        self.assertEqual(conn.parent_connectivity, '')


    def test_connectivity_bzip_in_zip(self):
        conn = connectivity.Connectivity.from_file("connectivity_68.zip")
        conn.configure()
        self.assertEqual(conn.weights.shape, (68, 68))
        self.assertEqual(conn.weights.max(), 0.12053822)
        self.assertEqual(conn.weights.min(), 0.0)
        self.assertEqual(conn.tract_lengths.shape, (68, 68))
        self.assertEqual(conn.tract_lengths.max(), 252.90276)
        self.assertEqual(conn.tract_lengths.min(), 0.0)
        self.assertEqual(conn.centres.shape, (68, 3))
        self.assertEqual(conn.orientations.shape, (68, 3))
        self.assertEqual(conn.region_labels.shape, (68,))
        self.assertEqual(conn.areas.shape, (0,))
        self.assertEqual(conn.undirected, 1)
        self.assertEqual(conn.speed, numpy.array([3.0]))
        self.assertEqual(conn.hemispheres.shape, (68,))
        self.assertEqual(conn.idelays.shape, (0,))
        self.assertEqual(conn.delays.shape, (68, 68))
        self.assertEqual(conn.number_of_regions, 68)



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(ConnectivityTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE) 