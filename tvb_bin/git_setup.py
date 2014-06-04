# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
import sys
import os
from tvb.core.decorators import user_environment_execution
from tvb.basic.config.settings import TVBSettings as cfg


@user_environment_execution
def _checkout_git():

    git_repo = sys.argv[1]
    destination_folder = git_repo.split('/')[-1].replace('.git', '')

    os.system('git clone %s %s' % (git_repo, destination_folder))
    os.chdir(destination_folder)
    os.system('git checkout trunk')
    TVB_PATH = cfg.TVB_PATH
    if TVB_PATH:
        TVB_PATH = TVB_PATH + os.pathsep + os.getcwd()
    else:
        TVB_PATH = os.getcwd()
    cfg.add_entries_to_config_file({cfg.KEY_TVB_PATH: TVB_PATH})



WINDOWS_USAGE = "Usage is 'contributor_setup.bat your.github.repository.link'"
UNIX_USAGE = "Usage is 'sh contributor_setup.sh your.github.repository.link'"

if len(sys.argv) < 2:
    if sys.platform == 'win32':
        raise Exception(WINDOWS_USAGE)
    raise Exception(UNIX_USAGE)

if os.system('git --version') != 0:
    raise Exception("You need to have git installed in order to set up TVB for contributions.")


if sys.platform == 'darwin':
    parent_folder = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    sys.path = [parent_folder, os.path.join(parent_folder, 'site-packages.zip'),
                os.path.join(parent_folder, 'lib-dynload')] + sys.path
    _checkout_git()
else:
    os.chdir('..')
    os.chdir('..')
    _checkout_git()
    
