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


# __future__ import must be kept at the beginning of the file
from __future__ import with_statement
"""
Usage should be `python pyinstaller.py setup_windows.spec`
You can also set 'cluster' as an environmental variable beforehand 

The resulting structure of the distribution will be:
{current folder} / TVB_Windows_{version}_x32_web.zip
"""
import os
import shutil
import zipfile
from build_pyinstaller import PyInstallerPacker, PYTHON_EXE



def create_start_scripts(base_folder, data_folder, python_exe):
    """
    Any startup scripts that are created after package generation should go here.
    
    @param app_name: the name of the resulted executable. 
        NOTE: For the current state of the Windows 64-bit build machine it seems
        that the generated app_name (tvb_start.exe) has some problem and raises an
            `ImportError: No module named pythoncom`
        A quick fix to this would be to just use the `python.exe` that we distribute
        anyway, so app_name should become `python.exe tvb_bin\app.py`
    """
    app_name = python_exe + ' -m tvb_bin.app'
    bin_folder = os.path.join(base_folder, 'bin')
    if not os.path.exists(bin_folder):
        os.mkdir(bin_folder)
    
    def _create_script_file(bin_folder, file_name, data_folder, command, end_message="Done"):
        """
        Private script which adds the common part of a command file.
        Unfortunately it can not be defined outside this function, or else it's not visible.
        """
        command_file = open(os.path.join(bin_folder, file_name + '.bat'), 'w')
        command_file.write('@echo off \n')
        command_file.write('rem Executing ' + file_name + ' \n')
        command_file.write('cd ..\\' + data_folder + ' \n')
        command_file.write('set PATH=%cd%;%path%; \n')
        command_file.write('set PYTHONPATH=%cd%; \n')
        command_file.write('set PYTHONHOME=%cd%; \n')
        command_file.write('cd exe \n')
        command_file.write(command + ' \n')
        command_file.write('echo "' + end_message + '" \n')
        command_file.close()
        os.chmod(os.path.join(bin_folder, file_name + '.bat'), 0775)
    
    _create_script_file(bin_folder, 'tvb_start', data_folder, app_name, "Starting...")
    _create_script_file(bin_folder, 'tvb_clean', data_folder, app_name + ' clean')
    _create_script_file(bin_folder, 'tvb_stop', data_folder, app_name + ' stop')
    _create_script_file(bin_folder, 'tvb_command', data_folder, app_name + ' console')
    _create_script_file(bin_folder, 'tvb_library', data_folder, app_name + ' library')
    #_create_script_file(bin_folder, 'tvb_model_validation', data_folder, app_name + ' validate %1\n')
    _create_script_file(bin_folder, 'contributor_setup', data_folder, python_exe + ' -m tvb_bin.git_setup %1\n')
   

#--------------------------- Setup variable declarations for PyInstaller starts here   --------------------------------

EXTRA_DEPENDENCIES = ['sqlite3', 'winshell']

EXCLUDE_DIRS = [ #On windows python standard libraries like os, site, runpy are located
                 #as same level as site-packages, so pyinstaller will also add site-packages.
                'site-packages', 
                 #Standard library for python on windows, not needed
                 'msilib', 'pydoc_data']

EXCLUDE_FILES = [#Additional DLL's that don't seem to be used
                 'msvcp90.dll', 'msvcr100.dll', 'nsi.dll', 'psapi.dll', 'secur32.dll', 'winnsi.dll', 'wldap32.dll', 
                 'wtsapi32.dll', 'gdiplus.dll', 'mfc90.dll', 'mfc90u.dll', 'mfcm90.dll', 'mfcm90u.dll',
                 'wxbase28uh_net_vc.dll', 'wxbase28uh_vc.dll', 'wxmsw28uh_adv_vc.dll',
                 'wxmsw28uh_core_vc.dll', 'wxmsw28uh_html_vc.dll', 'LIBEAY32.dll', 'SSLEAY32.dll',
                 'perfmon.pyd', 'servicemanager.pyd',
                 # wx files probably included after idle dependency but does not seem to be required
                 'wx._windows_.pyd', 'wx._misc_.pyd', 'wx._gdi_.pyd', 'wx._core_.pyd', 'wx._controls_.pyd'
                 ]
INCLUDE_FILES = ['decorator']

#--------------------------- Setup variable declarations for PyInstaller ends here   ----------------------------------

#Get path to python executable since we need to copy it into distribution
PYTHON_PATH = None
#This is the name under which we copy the python executable we distribute with the package
#Give it fixed name so we don't have anything like python2.6.exe or pythonw.exe that can mess
#up the commands afterwards.
if 'PYTHONHOME' in os.environ:
    PYTHON_PATH = os.environ['PYTHONHOME']
else:
    try:
        PYTHON_PATH = os.path.split(os.path.dirname(os.__file__))[0]
    except Exception:
        PYTHON_PATH = None
if PYTHON_PATH is None:
    print 'PYTHONHOME environment variable not set, and python executable location could not be deduced.'
    print 'Please set "export PYTHONHOME=$path/$to/python.exe" then try again!'
    exit()
PYTHON_PATH = os.path.join(PYTHON_PATH, PYTHON_EXE)


#--------------------------- Actual package generation flow starts here   ---------------------------------------------

PyInstallerPacker.set_pyinstaller_globals(globals())
PyInstallerPacker.gather_tvb_dependencies(EXTRA_DEPENDENCIES)
PyInstallerPacker.copy_additional_libraries(PYTHON_PATH, INCLUDE_FILES)
PyInstallerPacker.clean_up_files(EXCLUDE_FILES, EXCLUDE_DIRS)
create_start_scripts(PyInstallerPacker.RESULT_BASE_FOLDER, PyInstallerPacker.DATA_FOLDER_NAME, PYTHON_EXE)
PyInstallerPacker.generate_package('TVB_Windows')

#--------------------------- Actual package generation flow ends here   -----------------------------------------------
