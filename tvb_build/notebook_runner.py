# -*- coding: utf-8 -*-
#
# "TheVirtualBrain - Widgets" package
#
# (c) 2022-2023, TVB Widgets Team
#

import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count


def execute_notebook(in_path, notebook):
    with open(os.path.join(in_path, notebook), encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=None)

    ep.preprocess(nb)

    # with open(os.path.join(out_path, notebook), 'w+', encoding='utf-8') as f: # for debug only
    #     nbformat.write(nb, f)
    print(notebook + " - successful execution")


if __name__ == '__main__':

    if len(sys.argv) == 2:
        in_path = sys.argv[1]
    else:
        raise AttributeError("please insert the input path")

    if not os.getenv('CLB_AUTH'):
        os.environ['CLB_AUTH'] = 'abc'

    skipped_notebooks = [
                        'exploring_time_series_interactive.ipynb', # run separately because of other notebook dependency
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

    notebooks = [file for file in os.listdir(in_path) if file[-6:] == ".ipynb" and file not in skipped_notebooks]

    execute_notebook(in_path, 'exploring_time_series_interactive.ipynb')

    # start as many threads as logical cpus
    with ThreadPool(cpu_count()) as pool:
        pool.map(lambda notebook: execute_notebook(in_path, notebook), notebooks)
