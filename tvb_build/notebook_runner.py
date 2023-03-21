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


def execute_notebook(in_path, out_path, notebook):
    with open(os.path.join(in_path, notebook)) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=1000)

    ep.preprocess(nb)

    with open(os.path.join(out_path, notebook), 'w+', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(notebook + " - successful execution")


def swap_notebooks_execution_order(notebooks, notebook1, notebook2):
    """
    Swap 2 notebooks order of execution
    """
    index_1 = notebooks.index(notebook1)
    index_2 = notebooks.index(notebook2)
    notebooks[index_1], notebooks[index_2] = notebooks[index_2], notebooks[index_1]

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

    skipped_notebooks = ['encrypt_data.ipynb', # path to public key is missing
                         'export_encrypt_decrypt_data.ipynb', # path to public key is missing
                         'interacting_with_Allen.ipynb', # infinite loop, user input
                         'interacting_with_rest_api_fire_simulation.ipynb',
                         'interacting_with_rest_api_launch_operations.ipynb',
                         'interacting_with_the_framework.ipynb',
                         'launching_bids_adapter.ipynb' # a gui is needed
                         ]

    notebooks = [file for file in os.listdir(in_path) if file[-6:] == ".ipynb" and file not in skipped_notebooks]

    swap_notebooks_execution_order(notebooks,
                                   'exploring_power_spectra_interactive.ipynb',
                                   'exploring_time_series_interactive.ipynb')

    notebooks = notebooks[19:]
    print(notebooks)
    for notebook in notebooks:
        execute_notebook(in_path, out_path, notebook)

