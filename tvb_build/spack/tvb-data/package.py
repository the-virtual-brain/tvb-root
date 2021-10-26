# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class TvbData(PythonPackage):
    """
    Various demonstration datasets for use with The Virtual Brain are provided here.
    """

    homepage = "https://zenodo.org/record/4263723"
    url = 'https://zenodo.org/record/4263723/files/tvb_data.zip'

    maintainers = ['paulapopa']

    version('2.0.3', '1e02cdc21147f46644c57b14429f564f')

    # python_requires
    depends_on('python@3.8:', type=('build', 'run'))

    # setup_requires
    depends_on('py-setuptools', type='build')
