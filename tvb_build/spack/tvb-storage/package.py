# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class TvbStorage(PythonPackage):
    """
    "The Virtual Brain" Project (TVB Project) has the purpose of offering modern tools to the Neurosciences community,
    for computing, simulating and analyzing functional and structural data of human brains, brains modeled at the level
    of population of neurons.
    """

    homepage = "https://www.thevirtualbrain.org/"
    pypi = 'tvb-storage/tvb-storage-2.3.tar.gz'

    maintainers = ['paulapopa']

    version('2.3', 'e10bb40a486771703ba2776555ea5a453e619b07f10e4dea0d347043dee8f54b')

    # python_requires
    depends_on('python@3.8:', type=('build', 'run'))

    # setup_requires
    depends_on('py-setuptools', type='build')

    # install_requires
    depends_on('py-h5py', type=('build', 'run'))
    depends_on('py-numpy', type=('build', 'run'))
    depends_on('py-scipy', type=('build', 'run'))
    depends_on('tvb-library', type=('build', 'run'))
