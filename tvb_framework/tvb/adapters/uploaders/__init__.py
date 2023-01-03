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
Define a list with all Python modules in which the introspection mechanism should search for Import Adapters.
"""

ALL_UPLOADERS = ["brco_importer", "connectivity_measure_importer", "csv_connectivity_importer",
                 "gifti_surface_importer", "gifti_timeseries_importer", "mat_timeseries_eeg_importer",
                 "mat_timeseries_importer", "networkx_importer", "nifti_importer", "obj_importer",
                 "projection_matrix_importer", "region_mapping_importer", "sensors_importer", "tract_importer",
                 "tumor_dataset_importer", "tvb_importer", "zip_connectivity_importer", "zip_surface_importer"]
