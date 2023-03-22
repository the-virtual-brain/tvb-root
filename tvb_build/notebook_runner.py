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


def execute_notebook(in_path, out_path, notebook):
    with open(os.path.join(in_path, notebook), encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=1000)

    ep.preprocess(nb)

    with open(os.path.join(out_path, notebook), 'w+', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(notebook + " - successful execution")


if __name__ == '__main__':

    if len(sys.argv) == 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
    else:
        print("please insert the input and output paths")
        exit(-1)

    if not os.getenv('CLB_AUTH'):
        os.environ['CLB_AUTH'] = 'abc'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    skipped_notebooks = ['encrypt_data.ipynb', # path to public key is missing // de fixat
                         'export_encrypt_decrypt_data.ipynb', # path to public key is missing // de fixat
                         'interacting_with_Allen.ipynb', # infinite loop, user input // merge
                         'interacting_with_rest_api_fire_simulation.ipynb', # exclus
                         'interacting_with_rest_api_launch_operations.ipynb', # exclus
                         'interacting_with_the_framework.ipynb', # // de fixat
                         'launching_bids_adapter.ipynb', # exclus
                         'model_generation_using_dsl.ipynb',# exclus
                         'RateML_CUDA_on_HPC.ipynb', # exclus
                         'RateML_Python_TVB.ipynb', # exclus
                         'simulate_for_mouse.ipynb',
                         'simulate_surface_seeg_eeg_meg.ipynb',#lia
                         'simulate_zerlaut.ipynb', # no file is given, no such file connectivity_76
                         'Zerlaut_parametersweep_HPC.ipynb', # clb_oauth
                         'exploring_time_series_interactive.ipynb', #run separately because of other notebook dependency
                         'exploring_power_spectra_interactive.ipynb' #run separately because of other notebook dependency
                         ]

    notebooks = [file for file in os.listdir(in_path) if file[-6:] == ".ipynb" and file not in skipped_notebooks ]

    print(notebooks)

    execute_notebook(in_path, out_path, 'exploring_time_series_interactive.ipynb')
    execute_notebook(in_path, out_path, 'exploring_power_spectra_interactive.ipynb')

    # start as many threads as logical cpus
    pool = ThreadPool(cpu_count())
    pool.map(lambda notebook: execute_notebook(in_path, out_path, notebook), notebooks)
