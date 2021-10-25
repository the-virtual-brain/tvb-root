# Copyright 2013-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class TvbFramework(PythonPackage):
    """
    "The Virtual Brain" Project (TVB Project) has the purpose of offering modern tools to the Neurosciences community,
    for computing, simulating and analyzing functional and structural data of human brains, brains modeled at the level
    of population of neurons.
    """

    homepage = "https://www.thevirtualbrain.org/"
    pypi = 'tvb-framework/tvb-framework-2.3.tar.gz'

    maintainers = ['paulapopa']

    version('2.3', '0f3386135cdbd80bfd7d31f2e056b015a27ff6d081492db16274deed581b0aac')
    version('2.2', 'dfcdecee7325dd9b75eb63ddca56d046a4a6f3a20a9bc71a609315b6f151d68b')
    version('2.0.10', '082c8c742680804a62fd20b80d0ac0fa7421b0e1cd3a54293ff2fec2abc4f15d')
    version('2.0.9', '8a3af6d52e057db901d38c524d4bf1c7f78e793b990b27950d0765b8edf03c61')
    version('2.0.8', '95a8585b3095eecbdc8ca389269471a350b6876e732f62c260d75b615d10a237')

    # python_requires
    depends_on('python@3.8:', type=('build', 'run'))

    # setup_requires
    depends_on('py-setuptools', type='build')

    # install_requires
    depends_on('py-alembic', type=('build', 'run'))
    depends_on('py-cherrypy', type=('build', 'run'))
    depends_on('py-formencode', type=('build', 'run'))
    # depends_on('py-gevent', type=('build', 'run'))
    depends_on('py-h5py', type=('build', 'run'))
    depends_on('py-jinja2', type=('build', 'run'))
    depends_on('py-nibabel', type=('build', 'run'))
    depends_on('py-numpy', type=('build', 'run'))
    depends_on('py-pandas', type=('build', 'run'))
    depends_on('py-pillow', type=('build', 'run'))
    depends_on('py-psutil', type=('build', 'run'))
    depends_on('py-pyaescrypt', type=('build', 'run'))
    depends_on('py-requests', type=('build', 'run'))
    depends_on('py-scikit-learn', type=('build', 'run'))
    depends_on('py-scipy', type=('build', 'run'))
    depends_on('py-simplejson', type=('build', 'run'))
    depends_on('py-six', type=('build', 'run'))
    depends_on('py-sqlalchemy', type=('build', 'run'))
    depends_on('tvb-data', type='run')
    depends_on('tvb-library', type=('build', 'run'))
    depends_on('tvb-storage', type=('build', 'run'), when='@2.3')
    depends_on('py-werkzeug', type=('build', 'run'))
