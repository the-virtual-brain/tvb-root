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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""
import importlib
import os
import re
import sys
import shutil
import pkg_resources
import zipfile
from types import ModuleType


PYTHON_VERSION = "%s.%s.%s" % sys.version_info[:3]

EXCLUDES = [
    # part of tvb
    'tvb_bin', 'tvb_data', 'tvb', 'gdist',
    # part of python and setuptools
    'distutils', 'packaging', 'pkg_resources', 'setuptools', 'socketserver', '_thread', '_dummy_thread',
    # python 3 backports (PSF licensed)
    'futures', 'libfuturize', 'libpasteurize', 'lib2to3', 'past', 'ordereddict',
    'singledispatch', 'funcsigs', 'enum34', 'enum', 'reprlib', 'backports-abc', 'backports.functools-lru-cache',
    'backports.ssl-match-hostname', 'backports.shutil-get-terminal-size', 'copyreg', 'winreg',
    # others
    '_builtinsuites', 'builtins', 'bsddb', 'carbon', 'compiler', 'config', 'http', 'html',
    'hotshot', 'lib-dynload', '_markupbase', '_pytest',
    'openglcontext', 'pydoc_data', 'pysqlite2', 'pyximport',  # part of cython
    'queue', 'stdsuites', 'wxpython',
    # We exclude bellow shorter names for packages already introspected (mainly Mac)
    "foundation", "exceptionhandling", "pytest_cov", "pypiwin32", "pyyaml", "msgpack-python", "tlz",
    "objc", "appkit", "pyobjctools", "cocoa",
    "ipykernel", "ipython_genutils", "nbformat", "nbconvert",
    'finder', 'unittest', 'email', 'encodings', 'multiprocessing', 'json', 'curses', 'importlib', 'xml', 'logging'
]

EXCLUDES_DLL = []

# Windows *.pyds that are part of python standard libs
EXCLUDES_PYD = [
    '_psutil_windows.pyd', 'gdist.pyd', "_cffi_backend.pyd", "_scandir.pyd", "_yaml.pyd"
]

# libpq dependencies on dynamic psycopg linux 32
EXCLUDES_SO = [
    '_psutil_linux.so', '_psutil_posix.so', 'gdist.so', '_scandir.so',
    '_posixsubprocess.so' # already reported as subprocess32
]

EXCLUDES_DYLIB = [
    'libcrypto.1.0.0.dylib', 'libncursesw.5.dylib', 'libpq.5.dylib', 'libpq.5.8.dylib',
    'libssl.1.0.0.dylib', 'libsz.2.0.0.dylib',
    re.compile(r'libpython3.*\.dylib'),
    re.compile(r'libgcc.*\.dylib'),
    # Public domain:
    'liblzma.5.dylib',
    # Libz is needed by psycopg2
    re.compile(r'libz.*\.dylib'),
    # Already included
    re.compile(r'libhdf5_hl.*\.dylib'), re.compile(r'libhdf5.*\.dylib'), 'libsqlite3.0.dylib', 'liblcms.1.0.19.dylib',
    re.compile(r'libxslt.*\.dylib'), re.compile(r'libexslt.*\.dylib')  # Come with with libxml2
]

# py2app adds some frameworks to package that we also need to check for licenses.
EXCLUDES_FRAMEWORK = ['python.framework']

INIT = ['__init__.py', '__init__.pyc']
EXTRA_SEARCH_FOLDERS = ['site-packages', 'site-packages.zip']

EXTRA_MODULES = {
    'jquery': '2.1.1',
    'flot': '0.8.3',
    'hdf5': '1.8.17',
    'jit': '2.0.1',
    'd3': '3',
    'bct': '2017',
    'python': PYTHON_VERSION,
    'zlib': '1.0',
    'mathjax': '2.0'
}

ANACONDA_VERSION = "4.2"

# These file-name pattern should not be found in TVB distribution:
LICENSE_INTERDICTIONS = [re.compile(".*lzo.*")]

SETUPTOOLS_PACKAGE_VERSION = {}


def _get_dll_version_number(filename):
    """Read windows file version number from properties"""
    try:
        from win32api import GetFileVersionInfo, LOWORD, HIWORD
        info = GetFileVersionInfo(filename, "\\")
        version_ms = info['FileVersionMS']
        version_ls = info['FileVersionLS']
        tmp = (HIWORD(version_ms), LOWORD(version_ms),
               HIWORD(version_ls), LOWORD(version_ls))
        return ".".join([str(i) for i in tmp])
    except ImportError:
        print('Warning, win32api not found. All dll versions set to unknown')
    except Exception:
        print('DLL ' + filename + 'does not contain any information avout version')
    return 'unknown'


def _extract_all(zip_name, dest):
    """ Extract a ZIP archive to introspect"""
    zip_ = zipfile.ZipFile(zip_name)
    for file_ in zip_.namelist():
        if file_ == "./":
            continue
        if file_.endswith('/'):
            os.makedirs(os.path.join(dest, file_))
        else:
            zip_.extract(file_, dest)


def _get_module_version(module_name):
    """For a package name, return its version"""
    try:
        module = importlib.import_module(str(module_name))
        if hasattr(module, '__version__'):
            return str(module.__version__)
        if hasattr(module, 'version') and not isinstance(module.version, ModuleType):
            return str(module.version)
        if hasattr(module, 'version') and isinstance(module.version, ModuleType) and hasattr(module.version, 'version'):
            return str(module.version.version)
        pkg_search_name = module_name.replace('-', '').replace('_', '').lower()
        if pkg_search_name in SETUPTOOLS_PACKAGE_VERSION:
            return SETUPTOOLS_PACKAGE_VERSION[pkg_search_name]
    except ImportError:
        pass
    return 'unknown'


def _find_extra_modules(extra, modules_dict, excludes):
    """ Introspect supplementary non-python sub-folder"""
    for module in extra:
        if excludes is not None and module in excludes:
            continue
        modules_dict[module.lower()] = _get_module_version(module)
        if modules_dict[module] == 'unknown':
            modules_dict[module.lower()] = extra[module]


def _is_excluded(input_lib, excluded_list):
    """
    Check if the passed library, dylib, dll or so is in the excludes list in which
    case we don't need to do license checking.
    
    @param input_lib: the entry library, dylib, dll or so to be checked against the EXCLUDES list
    @param excluded_list: the list of libraries to check against 
    """
    for entry in excluded_list:
        if re.match(entry, input_lib.lower()):
            return True
    return False


def _find_modules(root_, modules_dict):
    """Introspect Python module in current specified root."""
    all_files = os.listdir(root_)
    for entry in all_files:
        full_path = os.path.join(root_, entry)
        # Validate current file not to be in the INTERDICTION list.
        for regex in LICENSE_INTERDICTIONS:
            if regex.match(entry):
                raise Exception("%s file has unacceptable license!!! " % (entry,))

        if (os.path.isdir(full_path) and not _is_excluded(entry, EXCLUDES) and (INIT[0] in os.listdir(full_path)
                                                                                or INIT[1] in os.listdir(full_path))):
            modules_dict[entry.lower()] = _get_module_version(entry)
        if entry in EXTRA_SEARCH_FOLDERS:
            if entry.endswith('.zip'):
                temp_folder = os.path.join(root_, 'TEMP_DEP_FOLDER')
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
                os.makedirs(temp_folder)
                _extract_all(full_path, temp_folder)
                _find_modules(temp_folder, modules_dict)
                shutil.rmtree(temp_folder)
            else:
                _find_modules(full_path, modules_dict)
        if entry.endswith('.dll') and not _is_excluded(entry, EXCLUDES_DLL):
            modules_dict[entry.lower()] = _get_dll_version_number(full_path)
        if entry.endswith('.pyd') and not _is_excluded(entry, EXCLUDES_PYD):
            modules_dict[entry.lower()] = 'unknown'
        if (entry.endswith('.so') or '.so.' in entry) and not _is_excluded(entry, EXCLUDES_SO):
            modules_dict[entry.lower()] = 'unknown'
        if entry.endswith('.dylib') and not _is_excluded(entry, EXCLUDES_DYLIB):
            modules_dict[entry.lower()] = 'unknown'
        if entry.endswith('.framework') and not _is_excluded(entry, EXCLUDES_FRAMEWORK):
            modules_dict[entry.lower()] = 'unknown'


def _find_pkg_modules(modules_dict):
    """
    Discovers the versions of all packages installed by setuptools
    """
    # pkg_resources is nice but it works for the current interpreter, which is not the distribution interpreter
    # O conda builds it is virtually the same
    # todo:Review this on the mac builds
    for dist in pkg_resources.working_set:
        key = dist.project_name.lower()
        if key not in modules_dict:
            if not _is_excluded(key, EXCLUDES):
                modules_dict[key] = dist.version


def parse_tree_structure(root_, excludes=None):
    """Main method, to return Python packages from a distribution folder"""

    # The working set will also include packages that are not in the virtual env.
    # IMPORTANT: this will list installed setuptools packages for the current interpreter!
    for dist in pkg_resources.working_set:
        if '0' <= dist.version[0] <= '9':  # why?
            key = dist.project_name.strip().replace('-', '').replace('_', '').lower()
            SETUPTOOLS_PACKAGE_VERSION[key] = dist.version.strip()

    modules_dict = {}
    _find_modules(root_, modules_dict)
    _find_extra_modules(EXTRA_MODULES, modules_dict, excludes)
    # TODO: make this work on the py2app build. This will require "switching" to the packages interpreter
    if not sys.platform == 'darwin':
        _find_pkg_modules(modules_dict)

    if 'anaconda' in sys.version.lower() or 'conda' in sys.executable:
        # TODO retrieve this from the build machine
        modules_dict['anaconda'] = ANACONDA_VERSION

    if sys.platform == 'darwin':
        # ----- Go into Contents/Frameworks and look for dlybs/so's-----
        path = os.path.split(root_)[0]
        path = os.path.split(path)[0]
        path = os.path.join(os.path.split(path)[0], 'Frameworks')
        if os.path.exists(path):
            _find_modules(path, modules_dict)
    return modules_dict


# Test case for Windows or Mac
if __name__ == '__main__':
    ROOT = 'D:\Projects\Brain\dist-repo\TVB_distribution\library.zip'
    ROOT_MAC = '../TVB_MacOS_dist/TVB_distribution/tvb.app/Contents/Resources/lib/python3.7'
    print(parse_tree_structure(ROOT))
