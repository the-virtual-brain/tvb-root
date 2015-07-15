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
Create TVB distribution package for Mac OS.

Execute:
    python setup_py2app.py py2app

"""

#Prepare TVB code and dependencies.
import os
import sys
import shutil
import setuptools
from tvb_bin.build_base import FW_FOLDER, DIST_FOLDER
from tvb_bin.build_pyinstaller import PyInstallerPacker



def _create_command_file(command_file_path, command, before_message, done_message=False):
    """
    Private script which adds the common part of a command file.
    """
    pth = command_file_path + ".command"
    with open(pth, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('cd "$(dirname "$0")"\n')
        f.write('echo "' + before_message + '"\n')
        f.write(command + "\n")
        if done_message:
            f.write('echo "Done."\n')


def _copy_collapsed(to_copy):
    """
    Merge multiple src folders, and filter some resources which are not needed (e.g. svn folders)
    """
    for module_path, destination_folder in to_copy.iteritems():
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for sub_folder in os.listdir(module_path):
            src = os.path.join(module_path, sub_folder)
            dest = os.path.join(destination_folder, sub_folder)

            if not os.path.isdir(src) and not os.path.exists(dest):
                shutil.copy(src, dest)

            if os.path.isdir(src) and not sub_folder.startswith('.') and not os.path.exists(dest):
                ignore_patters = shutil.ignore_patterns('.svn')
                shutil.copytree(src, dest, ignore=ignore_patters)

#--------------------------- PY2APP specific configurations--------------------------------------------

PY2APP_PACKAGES = ['cherrypy', 'email', 'h5py', 'IPython', 'idlelib', 'migrate', 'minixsv',
                   'numpy', 'scipy', 'sklearn', 'tables', 'tornado', 'tvb']

PY2APP_INCLUDES = ['apscheduler', 'apscheduler.scheduler', 'cfflib', 'cmath', 'contextlib', 'formencode', 'gdist',
                   'genshi', 'genshi.template', 'genshi.template.loader', 'jinja2', 'jsonschema', 'logging.config',
                   'lxml.etree', 'lxml._elementpath', 'markupsafe', 'matplotlib', 'minixsv', 'mod_pywebsocket',
                   'mplh5canvas.backend_h5canvas', 'mpl_toolkits.axes_grid', 'nibabel', 'numexpr', 'os', 'psycopg2',
                   'runpy', 'sqlite3', 'sqlalchemy', 'sqlalchemy.dialects.sqlite', 'sqlalchemy.dialects.postgresql',
                   'simplejson', 'StringIO', 'xml.dom', 'xml.dom.minidom', 'zlib', 'zmq']

PY2APP_EXCLUDES = ['_markerlib', 'coverage', 'cython', 'Cython', 'tvb_data', 'docutils', 'lib2to3',
                   'nose', 'opengl', 'PyOpenGL', 'PyQt4', 'sphinx', 'wx']

PY2APP_OPTIONS = {'iconfile': 'build_resources/icon.icns',
                  'plist': 'build_resources/info.plist',
                  'packages': PY2APP_PACKAGES,
                  'includes': PY2APP_INCLUDES,
                  'frameworks': ['Tcl', 'Tk'],
                  'resources': [],
                  'excludes': PY2APP_EXCLUDES,
                  'argv_emulation': True,
                  'strip': True,  # TRUE is the default
                  'optimize': '0'}


#This is a list of all the dynamic libraries identified so far that are added by py2app even though apparently
#they are not used by TVB. We will exclude them from package so as not to worry about licenses.
EXCLUDED_DYNAMIC_LIBS = ['libbz2.1.0.dylib', 'libdb-4.6.dylib', 'libexslt.0.dylib',
                         'libintl.8.dylib', 'liblzma.5.dylib', 'libpng15.15.dylib', 'libtiff.3.dylib',
                         'libsqlite3.0.dylib', 'libXss.1.dylib', 'libxml2.2.dylib', 'libxslt.1.dylib']

EXCLUDE_INTROSPECT_FOLDERS = [folder for folder in os.listdir(FW_FOLDER)
                              if os.path.isdir(os.path.join(FW_FOLDER, folder)) and folder != "tvb"]

#-------------- Finish configuration, starting build-script execution ---------------------------------

print "Running pre-py2app:"
print " - Cleaning old builds"

if os.path.exists('build'):
    shutil.rmtree('build')
if os.path.exists(DIST_FOLDER):
    shutil.rmtree(DIST_FOLDER)

print "PY2APP starting ..."
# Log everything from py2app in a log file
REAL_STDOUT, REAL_STDERR = sys.stdout, sys.stderr
sys.stdout = open('PY2APP.log', 'w')
sys.stderr = open('PY2APP_ERR.log', 'w')

FW_NAME = os.path.split(FW_FOLDER)[1]

setuptools.setup(name="tvb",
                 version=PyInstallerPacker.VERSION,
                 packages=setuptools.find_packages(FW_NAME, exclude=EXCLUDE_INTROSPECT_FOLDERS),
                 package_dir={'': FW_NAME},
                 license="GPL v2",
                 options={'py2app': PY2APP_OPTIONS},
                 include_package_data=True,
                 extras_require={'postgres': ["psycopg2"]},
                 app=['tvb_bin/app.py'],
                 setup_requires=['py2app'])

sys.stdout = REAL_STDOUT
sys.stderr = REAL_STDERR
print "PY2APP finished."

print "Running post-py2app build operations:"
print "- Start creating startup scripts..."

os.mkdir(os.path.join(DIST_FOLDER, "bin"))
os.mkdir(os.path.join(DIST_FOLDER, "demo_scripts"))

_create_command_file(os.path.join(DIST_FOLDER, "bin", 'distribution'),
                     '../tvb.app/Contents/MacOS/tvb $@', '')
_create_command_file(os.path.join(DIST_FOLDER, "bin", 'tvb_start'),
                     'source ./distribution.command start', 'Starting TVB Web Interface')
_create_command_file(os.path.join(DIST_FOLDER, "bin", 'tvb_clean'),
                     'source ./distribution.command clean', 'Cleaning up old TVB data.', True)
_create_command_file(os.path.join(DIST_FOLDER, "bin", 'tvb_stop'),
                     'source ./distribution.command stop', 'Stopping TVB related processes.', True)

IPYTHON_COMMAND = 'export PYTHONPATH=../tvb.app/Contents/Resources/lib/python2.7:' \
                  '../tvb.app/Contents/Resources/lib/python2.7/site-packages.zip:' \
                  '../tvb.app/Contents/Resources/lib/python2.7/lib-dynload\n' \
                  '../tvb.app/Contents/MacOS/python -m tvb_bin.run_ipython notebook '
_create_command_file(os.path.join(DIST_FOLDER, "bin", 'ipython_notebook'),
                      IPYTHON_COMMAND + '../demo_scripts', '')
_create_command_file(os.path.join(DIST_FOLDER, "demo_scripts", 'ipython_notebook'), IPYTHON_COMMAND, '')

_create_command_file(os.path.join(DIST_FOLDER, "bin", 'contributor_setup'),
                     'cd ..\n'
                     'export PYTHONPATH=tvb.app/Contents/Resources/lib/python2.7:'
                     'tvb.app/Contents/Resources/lib/python2.7/site-packages.zip:'
                     'tvb.app/Contents/Resources/lib/python2.7/lib-dynload\n'
                     './tvb.app/Contents/MacOS/python  '
                     'tvb.app/Contents/Resources/lib/python2.7/tvb_bin/git_setup.py $1 $2\n'
                     'cd bin\n',
                     'Setting-up contributor environment', True)

#py2app should have a --exclude-dynamic parameter but it doesn't seem to work until now
for entry in EXCLUDED_DYNAMIC_LIBS:
    path = os.path.join(DIST_FOLDER, "tvb.app", "Contents", "Frameworks", entry)
    if os.path.exists(path):
        os.remove(path)

DESTINATION_SOURCES = os.path.join("tvb.app", "Contents", "Resources", "lib", "python2.7")

_copy_collapsed({os.path.join("tvb_documentation_new", "demos"): os.path.join(DIST_FOLDER, "demo_scripts")})
                 # os.path.join("tvb_documentation_new", "tutorials"): os.path.join(DIST_FOLDER, "demo_scripts")})

PyInstallerPacker.add_sitecustomize(DIST_FOLDER, DESTINATION_SOURCES)
PyInstallerPacker.add_tvb_bin_folder(DIST_FOLDER, DESTINATION_SOURCES)
PyInstallerPacker.generate_final_zip("TVB_MacOS", DESTINATION_SOURCES)

## Clean after install      
shutil.rmtree(os.path.join(FW_FOLDER, 'tvb.egg-info'), True)    
    


