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

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


def execute_notebook(notebook):
    with open(notebook, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=None)
        ep.preprocess(nb)

    # with open(notebook, 'w+', encoding='utf-8') as f: # for debug only
    #     nbformat.write(nb, f)
    print(notebook + " - successful execution")


if __name__ == '__main__':

    if len(sys.argv) >= 2:
        in_path = sys.argv[1]
        sub_folders = sys.argv[2:]
        # compute relative paths
        sub_folders = [os.path.join(in_path, folder) for folder in sub_folders]
    else:
        raise AttributeError("please insert the input path")

    skipped_notebooks = [
        'exploring_time_series_interactive.ipynb',  # run separately because of other notebook dependency
        'export_encrypt_decrypt_data.ipynb',
        'interacting_with_rest_api_fire_simulation.ipynb',
        'interacting_with_rest_api_launch_operations.ipynb',
        'launching_bids_adapter.ipynb',
        'model_generation_using_dsl.ipynb',
        'RateML_CUDA_on_HPC.ipynb',
        'RateML_Python_TVB.ipynb',
        'simulate_surface_seeg_eeg_meg.ipynb',
        'Zerlaut_parametersweep_HPC.ipynb'
    ]

    notebooks = [os.path.join(in_path, file) for file in os.listdir(in_path) if
                 file[-6:] == ".ipynb" and file not in skipped_notebooks]
    for folder in sub_folders:
        notebooks += [os.path.join(folder, file) for file in os.listdir(folder) if
                      file[-6:] == ".ipynb" and file not in skipped_notebooks]

    first_notebook = os.path.join(in_path, 'exploring_time_series_interactive.ipynb')
    if os.path.exists(first_notebook):
        execute_notebook(first_notebook)

    # start as many threads as logical CPUs
    with ThreadPool(cpu_count()) as pool:
        pool.map(lambda notebook: execute_notebook(notebook), notebooks)
