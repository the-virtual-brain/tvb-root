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

Usage should be `python pyinstaller.py setup_linux.spec ['cluster']`
You can also set 'cluster' as an environmental variable beforehand 

The resulting structure of the distribution will be:
{current folder} / TVB_Linux_{version}_x32_web.zip
"""
import os
import shutil
import zipfile
from build_pyinstaller import PyInstallerPacker, PYTHON_EXE



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

cd exe
"""

    def _create_script_file(command_file_name, contents):
        pth = os.path.join(bin_folder, command_file_name + ".sh")
        with open(pth, 'w') as command_file:
            command_file.write('#!/bin/bash\n')
            command_file.write('cd "$(dirname "$0")"\n')
            command_file.write(contents + '\n')
        os.chmod(pth, 0775)


    def _create_command_file(command_file_name, command):
        """
        Private script which adds the common part of a script file.
        Unfortunately it can not be defined outside this function, or else it's not visible.
        """
        _create_script_file(command_file_name,
            'cd ../' + data_folder + '\n' +
            SCRIPT_PREPARE_TEXT + '\n' +
            command + '\n'
        )


    _create_command_file('distribution', './' + python_exe + ' -m tvb_bin.app $@')
    _create_command_file('contributor_setup', './' + python_exe + ' -m tvb_bin.git_setup $1')

    _create_script_file('tvb_start', 'source distribution.sh start')
    _create_script_file('tvb_clean', 'source distribution.sh clean')
    _create_script_file('tvb_stop', 'source distribution.sh stop')

    
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

EXCLUDE_FILES = [# Below are *so's from TVB-linux 32-bit IDLE related which don't seem to be needed
                 'libncursesw.so.5', 'libxcb-render.so.0', 'libxcb-render-util.so.0', 'libXcomposite.so.1',
                 'libXcursor.so.1', 'libXdamage.so.1', 'libXinerama.so.1', 'libXi.so.6', 'libXrandr.so.2',
                 # End of TVB 32-bit IDLE related
                 'audioop.so', 'dbm.so', 'gdbm.so', 'grp.so', 'libamd.so.2.2.0', 'libgdbm.so.3',
                 'libgdbm_compat.so.3', 'libjpeg.so.62', 'libumfpack.so.5.4.0', 'mx.DateTime.mxDateTime.mxDateTime.so',
                 # Below here are so files that only appear in old python2.6 packages
                 'sgmlop.so', 'pango.so', 'pangocairo.so', 'libpangocairo-1.0.so.0', 'libpangoft2-1.0.so.0', 
                 'libpango-1.0.so.0', 'wx._windows_.so', 'wx._misc_.so', 'wx._gdi_.so', 'wx._core_.so', 
                 'wx._controls_.so', 'libwx_gtk2u_xrc-2.8.so.0', 'libwx_gtk2u_richtext-2.8.so.0', 
                 'libwx_gtk2u_qa-2.8.so.0', 'libwx_gtk2u_html-2.8.so.0', 'libwx_gtk2u_core-2.8.so.0', 
                 'libwx_gtk2u_aui-2.8.so.0', 'libwx_gtk2u_adv-2.8.so.0', 'libwx_baseu_xml-2.8.so.0', 
                 'libwx_baseu_net-2.8.so.0', 'libwx_baseu-2.8.so.0', 'gtk._gtk.so', 'gtk.glade.so', 
                 'gobject._gobject.so', 'glib._glib.so', 'gio.unix.so', 'gio._gio.so', 'cairo._cairo.so', 'atk.so', 
                 'libaudio.so.2', 'libffi.so.5', 'libcairo.so.2', 'libatk-1.0.so.0', 'libgdk-x11-2.0.so.0', 
                 'libgdk_pixbuf-2.0.so.0', 'libglib-2.0.so.0', 'libgmodule-2.0.so.0', 'libglade-2.0.so.0', 
                 'libgio-2.0.so.0', 'libICE.so.6', 'libgtk-x11-2.0.so.0', 'libgthread-2.0.so.0', 'libgobject-2.0.so.0',
                 'libpixman-1.so.0', 'libpcre.so.3', 'libxml2.so.2', 'libtiff.so.4', 'libSM.so.6', 'libselinux.so.1', 
                 'libpyglib-2.0-python2.6.so.0', 'PyQt4.QtCore.so', 
                 'PyQt4.QtGui.so', 'sip.so', 'libQt3Support.so.4', 'libQtGui.so.4', 'libQtSql.so.4', 'libQtXml.so.4', 
                 'libQtCore.so.4', 'libQtNetwork.so.4', 'libQtSvg.so.4'
                 ]

EXCLUDE_DIRS = ['dist-packages', 'qt4_plugins']
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
