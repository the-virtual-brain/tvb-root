# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
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

Execute in root SVN:

    python tvb_build/setup_mac.py py2app

"""

import os
import sys
import shutil
import setuptools
import locale
import importlib
import tvb_bin
from glob import glob
from zipfile import ZipFile, ZIP_DEFLATED
from tvb.basic.profile import TvbProfile
from tvb.basic.config.environment import Environment
from tvb_build.third_party_licenses.build_licenses import generate_artefact


BIN_FOLDER = os.path.dirname(tvb_bin.__file__)
TVB_ROOT = os.path.dirname(os.path.dirname(__file__))
DIST_FOLDER = os.path.join(TVB_ROOT, "dist")
DIST_FOLDER_FINAL = "TVB_Distribution"
STEP1_RESULT = os.path.join(TVB_ROOT, "tvb_build", "build", "TVB_build_step1.zip")

FW_FOLDER = os.path.join(TVB_ROOT, "framework_tvb")
VERSION = TvbProfile.current.version.BASE_VERSION

FOLDERS_TO_DELETE = ['.svn', '.project', '.settings']
FILES_TO_DELETE = ['.DS_Store', 'dev_logger_config.conf']
EXCLUDED_DYNAMIC_LIBS = []

# Modules to be copied manually into "packages" locations, as they are not found by py2app.
# These are six modules.
EXTRA_MODULES = ['six.moves.BaseHTTPServer']

# --------------------------- PY2APP specific configurations--------------------------------------------

PY2APP_PACKAGES = ['cherrypy', 'email', 'h5py', 'IPython', 'ipykernel', 'ipykernel_launcher', 'nbformat',
                   'lib2to3', "llvmlite", 'migrate', 'numba', 'notebook', 'numpy', 'pkg_resources',
                   'PyObjCTools', 'scipy', 'sklearn', 'tables', 'tornado', 'tvb']

PY2APP_INCLUDES = ['allensdk', 'cfflib', 'cmath', 'contextlib', 'formencode', 'gdist',
                   'jinja2', 'jinja2.ext', 'jsonschema',
                   'logging.config', 'markupsafe', 'matplotlib', 'mpl_toolkits.axes_grid', 'nibabel',
                   'numexpr', 'os', 'pandas._libs.skiplist', 'psycopg2',
                   'pygments.formatters.html', 'pygments.lexers.python', 'runpy', 'sqlite3', 'sqlalchemy',
                   'sqlalchemy.dialects.sqlite', 'sqlalchemy.dialects.postgresql', 'sqlalchemy.sql.default_comparator',
                   'simplejson', 'six', 'StringIO', 'xml.dom', 'xml.dom.minidom', 'zlib', 'zmq']

PY2APP_EXCLUDES = ['_markerlib', 'altgraph', 'coverage', 'cython', 'Cython', 'tvb_data', 'docutils',
                   'macholib', 'modulegraph', 'nose', 'OpenGL', 'py2app', 'PyOpenGL', 'PyQt4',
                   'sphinx', 'test', 'testpath', 'wx']

PY2APP_OPTIONS = {'iconfile': 'tvb_build/icon.icns',
                  'plist': 'tvb_build/info.plist',
                  'packages': PY2APP_PACKAGES,
                  'includes': PY2APP_INCLUDES,
                  'frameworks': ['Tcl', 'Tk'],
                  'resources': [],
                  'excludes': PY2APP_EXCLUDES,
                  'argv_emulation': True,
                  'strip': True,  # TRUE is the default
                  'optimize': '0'}


# --------------------------- Start defining functions: --------------------------------------------

def _create_command_file(command_file_path, command, before_message, done_message=False):
    """
    Private script which adds the common part of a command file.
    """
    pth = command_file_path + ".command"
    with open(pth, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('cd "$(dirname "$0")"\n')
        f.write('echo "' + before_message + '"\n')
        f.write('if [ -z "$LANG" ]; then\n')
        f.write("    export LANG=en_US.UTF-8\n")
        f.write('fi\n')
        f.write("export LC_ALL=$LANG\n")
        f.write("export PYTHONIOENCODING=utf8 \n")
        f.write("# export TVB_USER_HOME=`pwd`\n")
        f.write(command + "\n")
        if done_message:
            f.write('echo "Done."\n')


def _copy_tvb_sources(library_folder):
    """
    Make sure all TVB folders are collapsed together in one folder in the distribution.
    """
    print("- Start to copy TVB resources into: " + library_folder)
    import tvb

    destination_folder = os.path.join(library_folder, 'tvb')
    for module_path in tvb.__path__:
        for sub_folder in os.listdir(module_path):
            src = os.path.join(module_path, sub_folder)
            dest = os.path.join(destination_folder, sub_folder)
            if os.path.isdir(src) and not (sub_folder.startswith('.') or sub_folder.startswith("tests")):
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                print("  Copy: " + str(src))
                ignore_patterns = shutil.ignore_patterns('*.pyc', '.svn')
                shutil.copytree(src, dest, ignore=ignore_patterns)

    tests_folder = os.path.join(destination_folder, "tests")
    if os.path.exists(tests_folder):
        shutil.rmtree(tests_folder, True)
        print("  Removed: " + str(tests_folder))

    for excluded in [os.path.join(destination_folder, "simulator", "doc"),
                     os.path.join(destination_folder, "simulator", "demos")]:
        if os.path.exists(excluded):
            shutil.rmtree(excluded, True)
            print("  Removed: " + str(excluded))


def _copy_module(module_str, destination_folder):
    """Import module, find its origin __file__ and copy it into distribution"""
    imported_module = importlib.import_module(module_str)
    six_extra_src = imported_module.__file__
    print("- Copying " + six_extra_src + " into " + destination_folder)
    shutil.copy2(six_extra_src, destination_folder)


def _introspect_licenses(destination_folder, root_introspection, extra_licenses_check=None):
    """Generate archive with 3rd party licenses"""
    print("- Introspecting for dependencies..." + str(root_introspection))

    try:
        locale.getdefaultlocale()
    except Exception:
        os.environ['LANG'] = 'en_US.UTF-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'

    zip_name = generate_artefact(root_introspection, extra_licenses_check=extra_licenses_check)
    ZipFile(zip_name).extractall(destination_folder)
    os.remove(zip_name)
    print("- Dependencies archive with licenses done.")


def _zipdir(basedir, archivename):
    """Create ZIP archive from folder"""
    assert os.path.isdir(basedir)
    with ZipFile(archivename, "w", ZIP_DEFLATED) as z_file:
        for root, _, files in os.walk(basedir):
            # NOTE: ignore empty directories
            for file_nname in files:
                absfn = os.path.join(root, file_nname)
                zfn = absfn[len(basedir) + len(os.sep):]
                z_file.write(absfn, zfn)


def _clean_up(folder_path, to_delete):
    """
    Remove any read only permission for certain files like those in .svn, then delete the files.
    """
    # Add Write access on folder
    folder_name = os.path.split(folder_path)[1]
    will_delete = False
    os.chmod(folder_path, 0o777)
    if to_delete or folder_name in FOLDERS_TO_DELETE:
        will_delete = True

    # step through all the files/folders and change permissions
    for file_ in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_)
        os.chmod(file_path, 0o777)
        # if it is a directory, do a recursive call
        if os.path.isdir(file_path):
            _clean_up(file_path, to_delete or will_delete)
        # for files merely call chmod
        else:
            if file_ in FILES_TO_DELETE:
                os.remove(file_path)

    if to_delete or will_delete:
        shutil.rmtree(folder_path)


def _generate_distribution(final_name, library_path, version, extra_licensing_check=None):
    # merge sources
    library_abs_path = os.path.join(DIST_FOLDER, library_path)

    _copy_tvb_sources(library_abs_path)

    bct_src = os.path.join("externals", "BCT")
    bct_dst = os.path.join(DIST_FOLDER, library_path, "externals", "BCT")
    print("- Copying " + bct_src + " to " + bct_dst)
    shutil.copytree(bct_src, bct_dst)

    for extra in EXTRA_MODULES:
        _copy_module(extra, library_abs_path)

    bin_src = os.path.join("tvb_bin", "tvb_bin")
    bin_dst = os.path.join(library_abs_path, "tvb_bin")
    print("- Copying " + bin_src + " to " + bin_dst)
    shutil.copytree(bin_src, bin_dst)

    demo_data_src = os.path.join(DIST_FOLDER, "_tvb_data")
    demo_data_dst = os.path.join(library_abs_path, "tvb_data")
    print("- Moving " + demo_data_src + " to " + demo_data_dst)
    os.rename(demo_data_src, demo_data_dst)

    online_help_src = os.path.join(DIST_FOLDER, "_help")
    online_help_dst = os.path.join(library_abs_path, "tvb", "interfaces", "web", "static", "help")
    print("- Moving " + online_help_src + " to " + online_help_dst)
    os.rename(online_help_src, online_help_dst)

    print("- Cleaning up non-required files...")
    _clean_up(DIST_FOLDER, False)
    if os.path.exists(DIST_FOLDER_FINAL):
        shutil.rmtree(DIST_FOLDER_FINAL)
    os.rename(DIST_FOLDER, DIST_FOLDER_FINAL)
    shutil.rmtree('tvb.egg-info', True)
    shutil.rmtree('build', True)
    for file_zip in glob('*.zip'):
        os.unlink(file_zip)

    print("- Creating required folder structure...")
    if os.path.exists(final_name):
        shutil.rmtree(final_name)
    os.mkdir(final_name)
    shutil.move(DIST_FOLDER_FINAL, final_name)

    if extra_licensing_check:
        extra_licensing_check = extra_licensing_check.split(';')
        for idx in range(len(extra_licensing_check)):
            extra_licensing_check[idx] = os.path.join(final_name, DIST_FOLDER_FINAL, extra_licensing_check[idx])
    _introspect_licenses(os.path.join(final_name, DIST_FOLDER_FINAL, 'THIRD_PARTY_LICENSES'),
                         os.path.join(final_name, DIST_FOLDER_FINAL, library_path), extra_licensing_check)
    print("- Creating the ZIP folder of the distribution...")
    zip_name = final_name + "_" + version + ".zip"
    if os.path.exists(zip_name):
        os.remove(zip_name)
    _zipdir(final_name, zip_name)
    if os.path.exists(final_name):
        shutil.rmtree(final_name)
    print('- Finish creation of distribution ZIP')


def prepare_py2app_dist():
    print("Running pre-py2app:")
    print(" - Cleaning old builds")

    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists(DIST_FOLDER):
        shutil.rmtree(DIST_FOLDER)

    print(" - Decompressing " + STEP1_RESULT + " into '" + DIST_FOLDER)
    step1_tmp_dist_folder = os.path.join(TVB_ROOT, 'TVB_Distribution')
    if os.path.exists(step1_tmp_dist_folder):
        shutil.rmtree(step1_tmp_dist_folder)
    os.mkdir(step1_tmp_dist_folder)
    ZipFile(STEP1_RESULT).extractall(step1_tmp_dist_folder)
    # the above created a TVB_Distribution/ we need a dist folder
    shutil.move(step1_tmp_dist_folder, DIST_FOLDER)
    # make needed directory structure that is not in the step1 zip
    # bin dir is initially empty, step1 does not support empty dirs in the zip
    os.mkdir(os.path.join(DIST_FOLDER, 'bin'))

    print("PY2APP starting ...")
    # Log everything from py2app in a log file
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = open('PY2APP.log', 'w')
    sys.stderr = open('PY2APP_ERR.log', 'w')

    fw_name = "framework_tvb"

    setuptools.setup(name="tvb",
                     version=VERSION,
                     packages=setuptools.find_packages(fw_name),
                     package_dir={'': fw_name},
                     license="GPL v3",
                     options={'py2app': PY2APP_OPTIONS},
                     include_package_data=True,
                     extras_require={'postgres': ["psycopg2"]},
                     app=['tvb_bin/tvb_bin/app.py'],
                     setup_requires=['py2app'])

    sys.stdout = real_stdout
    sys.stderr = real_stderr
    print("PY2APP finished.")

    print("Running post-py2app build operations:")
    print("- Start creating startup scripts...")

    _create_command_file(os.path.join(DIST_FOLDER, "bin", 'distribution'),
                         '../tvb.app/Contents/MacOS/tvb $@', '')
    _create_command_file(os.path.join(DIST_FOLDER, "bin", 'tvb_start'),
                         'source ./distribution.command start', 'Starting TVB Web Interface')
    _create_command_file(os.path.join(DIST_FOLDER, "bin", 'tvb_clean'),
                         'source ./distribution.command clean', 'Cleaning up old TVB data.', True)
    _create_command_file(os.path.join(DIST_FOLDER, "bin", 'tvb_stop'),
                         'source ./distribution.command stop', 'Stopping TVB related processes.', True)

    jupyter_command = 'export PYTHONPATH=../tvb.app/Contents/Resources/lib/' + Environment.PYTHON_FOLDER + ':' \
                      '../tvb.app/Contents/Resources/lib/' + Environment.PYTHON_FOLDER + '/site-packages.zip:' \
                      '../tvb.app/Contents/Resources/lib/' + Environment.PYTHON_FOLDER + '/lib-dynload\n' \
                      '../tvb.app/Contents/MacOS/python -m tvb_bin.run_jupyter notebook '
    _create_command_file(os.path.join(DIST_FOLDER, "bin", 'jupyter_notebook'),
                         jupyter_command + '../demo_scripts', 'Launching IPython Notebook from TVB Distribution')

    # py2app should have a --exclude-dynamic parameter but it doesn't seem to work until now
    for entry in EXCLUDED_DYNAMIC_LIBS:
        path = os.path.join(DIST_FOLDER, "tvb.app", "Contents", "Frameworks", entry)
        if os.path.exists(path):
            os.remove(path)

    destination_sources = os.path.join("tvb.app", "Contents", "Resources", "lib", Environment.PYTHON_FOLDER)
    _generate_distribution("TVB_MacOS", destination_sources, VERSION)

    # cleanup after install
    shutil.rmtree(os.path.join(FW_FOLDER, 'tvb.egg-info'), True)


if __name__ == '__main__':
    prepare_py2app_dist()
