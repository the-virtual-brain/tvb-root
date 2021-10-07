# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class TvbLibrary(PythonPackage):
    """
    "The Virtual Brain" Project (TVB Project) has the purpose of offering modern tools to the Neurosciences community,
    for computing, simulating and analyzing functional and structural data of human brains, brains modeled at the level
    of population of neurons.
    """

    homepage = "https://www.thevirtualbrain.org/"
    pypi = 'tvb-library/tvb-library-2.3.tar.gz'

    maintainers = ['paulapopa']

    version('2.3', '0595f2eca95e5ed4c7a1c88425914cc71d0ea7a9f4ec575b6a315ca2408ea801')
    version('2.2', 'de70440b0cfd517e49a8ef52aa01f3bfde87a907abbf68ae5d85fef9a28000dd')
    version('2.1', '99c2817d9d341abd6d7ff07c4a827f58462d85a4dbb975e514840f362b3ca5cc')
    version('2.0.10', '27ece9ec3a79125b037fdd67963da23dc4fad7bd7154b884faa6c26c2775a1b8')
    version('2.0.9', '0c3109c03665e3dd516fda12ba2155d718cd9933fb25b9f7bd0906851e937f39')
    version('2.0.8', '41e912723b66fe7beeff79f6b760d8ae8c34b0e80e668e81ce59c380ad00506d')
    version('2.0.7', 'b9a6c03b8b7c55e512b0a601260934c984870c875f958423c360e3813e70100b')
    version('2.0.6', 'f1ee168939e522f698b2fe18c2b7013b827ace198d4af777b38cf80fc2ab5db3')
    version('2.0.5', 'a1af85c3a376b52daa140416320f59794263fac99796a3a9e47ec1db46bda160')
    version('2.0.3', 'f4ebe3f3bba13dd6b32568cc016cb218251002a8c20acb992e030ab2ca5b30c8')
    version('2.0.2', '1893e641108c8fdbafc7c3400ac95b0f5f0a714fd0a378258ac065d11d2de071')
    version('2.0', 'a89bcd1949788d35722a1dc1e3bb8d5e32fa02a4eef977fd476ab6df18285e9b')

    # python_requires
    depends_on('python@3.8:', type=('build', 'run'))

    # setup_requires
    depends_on('py-setuptools', type='build')

    # install_requires
    depends_on('py-autopep8', type=('build', 'run'))
    depends_on('py-mako', type=('build', 'run'))
    depends_on('py-matplotlib', type=('build', 'run'))
    depends_on('py-numpy', type=('build', 'run'))
    depends_on('py-numba', type=('build', 'run'))
    depends_on('py-numexpr', type=('build', 'run'))
    depends_on('py-requests', type=('build', 'run'))
    depends_on('py-scipy', type=('build', 'run'))
    depends_on('py-six', type=('build', 'run'))
    depends_on('tvb-data', type=('run'))

    # extra_requires
    # ["h5py", "mpl_toolkits", "tvb-gdist"]

    # test_requires
    # ["pytest", "pytest-benchmark"]
