# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
Define file storage changes for each release.

Created on Dec 10, 2012

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>

CONVENTIONS: 
    - Each file from the file_update_scripts must have a name following the pattern: ${VERSION_NR}_update_files. 
      ${VERSION_NR} is the file storage version, and NOT TVB release version or database version!!.
    
    - Each file corresponding to ${VERSION_NR} should have a public function `upgrade(input_file)` which takes as input
      a string representing a file path and does any changes required to bring that file from:
      ${VERSION_NR} - 1 to ${VERSION_NR}
"""  


