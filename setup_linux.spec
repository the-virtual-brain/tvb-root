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
Accepted optional argument: 'cluster'

Usage should be `python pyinstaller.py setup_linux.spec

The resulting structure of the distribution will be:
{current folder} / TVB_Linux_{version}_x{32|64}_web.zip
"""
import os
import shutil
import zipfile
from tvb_bin.build_pyinstaller import PyInstallerPacker, PYTHON_EXE



def create_start_scripts(base_folder, data_folder, python_exe, arg_cluster):
    """
    Any startup scripts that are created after package generation should go here.
    
    @param app_name: the name of the resulted executable 
    """
    bin_folder = os.path.join(base_folder, 'bin')
    if not os.path.exists(bin_folder):
        os.mkdir(bin_folder)
        
    SCRIPT_PREPARE_TEXT = """# make sure system knows where our libraries are
if [ ${LD_LIBRARY_PATH+1} ]; then
  export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
else
  export LD_LIBRARY_PATH=`pwd`
fi

if [ ${LD_RUN_PATH+1} ]; then
  export LD_RUN_PATH=`pwd`:LD_RUN_PATH
else
  export LD_RUN_PATH=`pwd`
fi

# Some environment variables are also set in the initialize method of
# the deployment profile. We just need these here to get rid of ugly 
# warning messages on the first step.
export PYTHONHOME=`pwd`
export PYTHONPATH=`pwd`

"""

    def _create_script_file(command_file_name, contents):
        """
        Private script which generated a command file inside tvb-bin distribution folder.
        Unfortunately it can not be defined outside this function, or else it's not visible with PyInstaller.
        """
        pth = os.path.join(bin_folder, command_file_name + ".sh")
        with open(pth, 'w') as command_file:
            command_file.write('#!/bin/bash\n')
            command_file.write('cd "$(dirname "$0")"\n')
            command_file.write(contents + '\n')
        os.chmod(pth, 0775)


    def _create_file_with_tvb_paths(command_file_name, command):
        """
        Private script which adds the common part of a script TVB file.
        Unfortunately it can not be defined outside this function, or else it's not visible with PyInstaller.
        """
        tvb_command_text = 'cd ../' + data_folder + '\n' + \
                            SCRIPT_PREPARE_TEXT + '\n' + \
                            command + '\n'
        _create_script_file(command_file_name, tvb_command_text)


    _create_file_with_tvb_paths('distribution', './' + python_exe + ' -m tvb_bin.app $@')
    _create_file_with_tvb_paths('contributor_setup', './' + python_exe + ' -m tvb_bin.git_setup $1')

    _create_script_file('tvb_start', 'sh distribution.sh start')
    _create_script_file('tvb_clean', 'sh distribution.sh clean')
    _create_script_file('tvb_stop', 'sh distribution.sh stop')

    
    if arg_cluster in os.environ:
        COMMAND = open(os.path.join(bin_folder, 'clusterLauncher'), 'w')
        COMMAND.write(open("tvb_bin/cluster_launch.sh", 'r').read())
        COMMAND.close()
        os.chmod(os.path.join(bin_folder, 'clusterLauncher'), 0775)
        #Get out of bin folder and up to tvb root
        os.chdir('..')
        COMMAND = open(os.path.join(bin_folder, 'clusterSetup.sh'), 'w')
        COMMAND.write('#!/bin/sh\n')
        COMMAND.write("echo 'export TVB_ROOT=`pwd`' >> $HOME/.profile\n")
        COMMAND.write("echo '\n' >> $HOME/.profile\n")
        COMMAND.write("echo 'export PATH=$TVB_ROOT:$PATH' >> $HOME/.profile\n")
        COMMAND.write(". $HOME/.profile\n")
        COMMAND.close()
        os.chmod(os.path.join(bin_folder, 'clusterSetup.sh'), 0775)


#--------------------------- Setup variable declarations for PyInstaller starts here   --------------------------------

EXTRA_DEPENDENCIES = ['pysqlite2']
EXTRA_BINARIES = [('libXfixes.so.3', '/usr/lib/libXfixes.so.3', 'BINARY')]

EXCLUDE_FILES = ['grp.so', 'audioop.so', 'libjpeg.so.62', 'tvb._speedups.history.so', 'tvb._speedups.models.so']
EXCLUDE_DIRS = ['dist-packages', 'ensurepip', 'qt4_plugins']
INCLUDE_FILES = ['decorator', 'cmath']
ARG_CLUSTER = 'cluster'

#---------------------------   Setup variable declarations for PyInstaller ends here   --------------------------------

# Get path to Python executable since we need to copy it into distribution
PYTHON_PATH = None
if 'PYTHONPATH' in os.environ:
    PYTHON_PATH = os.environ['PYTHONPATH']
else:
    try:
        PYTHON_PATH = os.path.dirname(os.__file__).replace('/lib/', '/bin/')
    except Exception:
        PYTHON_PATH = None
if PYTHON_PATH is None:
    print 'PYTHONPATH environment variable not set, and Python executable location could not be deduced.'
    print 'Please set "export PYTHONPATH=$path/$to/python" then try again!'
    exit()

#--------------------------- Actual package generation flow starts here   ---------------------------------------------

PyInstallerPacker.set_pyinstaller_globals(globals())
PyInstallerPacker.gather_tvb_dependencies(EXTRA_DEPENDENCIES, EXTRA_BINARIES)
PyInstallerPacker.copy_additional_libraries(PYTHON_PATH, INCLUDE_FILES)
PyInstallerPacker.clean_up_files(EXCLUDE_FILES, EXCLUDE_DIRS)
create_start_scripts(PyInstallerPacker.RESULT_BASE_FOLDER, PyInstallerPacker.DATA_FOLDER_NAME, PYTHON_EXE, ARG_CLUSTER)
PyInstallerPacker.generate_package('TVB_Linux')

#--------------------------- Actual package generation flow ends here   -----------------------------------------------
