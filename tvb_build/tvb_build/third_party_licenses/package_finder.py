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
    'tvb_bin', 'tvb_data', 'tvb', 'mplh5canvas', 'gdist',
    # part of python and setuptools
    'distutils', 'pkg_resources', 'setuptools',
    # python 3 backports
    'futures', 'backports.ssl-match-hostname',
    # others
    '_builtinsuites', 'bsddb', 'carbon', 'compiler', 'config',
    'hotshot', 'lib-dynload',
    'openglcontext', 'pydoc_data', 'pysqlite2', 'pyximport', # part of cython
    'stdsuites', 'wxpython',
    ## We exclude bellow shorter names for packages already introspected.
    "foundation", "objc", "appkit", "exceptionhandling", "pyobjctools"
]

# todo: understand why this is needed on mac
if sys.platform == 'darwin':
    EXCLUDES.extend([
        'xml', 'logging', 'importlib', 'curses', 'unittest', 'multiprocessing',
        'json', 'encodings', 'test', 'email', 'finder'
    ])

EXCLUDES_DLL = ['libeay32.dll', 'msvcm90.dll', 'msvcr90.dll', 'python27.dll',
                'pywintypes27.dll', 'sqlite3.dll', 'ssleay32.dll', 'zlib1.dll', 'zlib.dll',
                ## match any of the dll, so or dylib from hdf5 hl library, since the license is already
                ## included by the libhdf5.* equivalent
                'hdf5_hldll.dll', 'hdf5_hl.dll', 'h5py_hdf5_hl.dll', 'h5py_hdf5.dll', 'pythoncom27.dll',
                ## Exclude numpy distributed dll's
                'libifcoremd.dll', 'libiomp5md.dll', 'libmmd.dll',
                ## These come from psycopg2 postgresql
                'libintl-8.dll', 'libpq.dll',
                ]
EXCLUDES_PYD = [## Windows *.pyds that are part of python standard libs
                '_bsddb.pyd', '_ctypes.pyd', '_hashlib.pyd', '_imaging.pyd',
                '_multiprocessing.pyd', '_socket.pyd', '_ssl.pyd', '_tkinter.pyd',
                'pyexpat.pyd', 'select.pyd', 'unicodedata.pyd', re.compile(r'win32.*\.pyd'),
                ## Windows *.pyds already included in licensing due to python package
                '_psutil_mswindows.pyd', '_sqlite3.pyd', 'bz2.pyd', 'gdist.pyd', 'genshi._speedups.pyd',
                re.compile(r'h5py.*\.pyd'), re.compile(r'matplotlib.*\.pyd'), 'numexpr.interpreter.pyd',
                re.compile(r'numpy.*\.pyd'), 'openssl.*\.pyd', 'psycopg2._psycopg.pyd', 'pil._imaging',
                re.compile(r'scipy.*\.pyd'), 'sqlalchemy.cresultproxy.pyd', 'sqlalchemy.cprocessors.pyd',
                'simplejson._speedups.pyd', '_psutil_windows.pyd',
                ]

EXCLUDES_SO = [  # libpq dependencies on dynamic psycopg linux 32
                 'libcom_err.so.2', 'libgssapi_krb5.so.2', 'libldap_r-2.4.so.2', 'libkrb5support.so.0',
                 'libk5crypto.so.3', 'libkeyutils.so.1', 'liblber-2.4.so.2', 'libtasn1.so.3', 'libgcrypt.so.11',
                 'libgpg-error.so.0',
                 # This are the so' which are only needed by the linux 32 python 2.6 machine
                 '_bytesio.so', '_fileio.so', 'libblt.2.4.so.8.5', 'libuuid.so.1', 'libxcb-render-util.so.0',
                 'libxcb-render.so.0', 'libxcomposite.so.1', 'libxcursor.so.1', 'libxdamage.so.1', 'libxfixes.so.3',
                 'libxi.so.6', 'libxinerama.so.1', 'libxrandr.so.2', 'libxt.so.6', 'openssl.crypto.so',
                 'openssl.rand.so', 'pil._imaging.so',
                 # SO's which we can exclude from license checking being either python standard or
                 # part of library that is already in packages_accepted.xml
                 '_bisect.so', '_collections.so', '_functools.so', '_hashlib.so', '_io.so', '_locale.so',
                 '_random.so', '_socket.so', '_ssl.so', '_struct.so', '_weakref.so', 'unicodedata.so',
                 'array.so', 'cpickle.so', 'cstringio.so', 'select.so', 'fcntl.so', 'binascii.so', 'future_builtins.so',
                 'operator.so', 'zlib.so', 'time.so', 'itertools.so', 'math.so', 'strop.so', 'syslog.so',
                 '_bsddb.so', '_codecs_cn.so', '_codecs_hk.so', '_codecs_iso2022.so', '_codecs_jp.so',
                 '_codecs_kr.so', '_codecs_tw.so', '_csv.so', '_ctypes.so', '_curses.so', 'datetime.so', '_heapq.so',
                 '_imaging.so', '_json.so', '_lsprof.so', '_lsprof.so', '_multibytecodec.so', '_multiprocessing.so',
                 '_psutil_linux.so', '_psutil_posix.so', '_psutil_osx.so', '_sqlite3.so', '_tkinter.so', 'cmath.so',
                 'libcrypto.so.0.9.8', 'libgcc_s.so.1', 'libssl.so.0.9.8', 'libstdc\\+\\+.so.6',
                 'libz.so', 'mmap.so', 'parser.so', 'pyexpat.so', 'readline.so', 'resource.so', 'termios.so',
                 re.compile(r'libncurses.*'), re.compile(r'libreadline\..*'), re.compile(r'libpython2\..*\.so.*'),
                 'gdist.so', 'libgfortran.so.3', 'libfontconfig.so.1', 'libsqlite3.so.0', 'numexpr.interpreter.so',
                 'genshi._speedups.so', 'psycopg2._psycopg.so', 'pysqlite2._sqlite.so', 'simplejson._speedups.so',
                 re.compile(r'sqlalchemy\..*\.so'), re.compile(r'scipy\..*\.so'), re.compile(r'libpq.so\.*'),
                 re.compile(r'matplotlib\..*\.so'), re.compile(r'numpy\..*\.so'), re.compile(r'_psutil\.*\.so'),
                 'libxau.so.6', 'libxcb.so.1', 'libxdmcp.so.6', 'libxext.so.6', 'libxft.so.2', 'libxrender.so.1',
                 'libxss.so.1', re.compile(r'h5py\..+\.so'), re.compile(r'libhdf5_hl.*\.so')]

EXCLUDES_DYLIB = ['libcrypto.1.0.0.dylib', 'libgcc_s.1.dylib',
                  'libgfortran.2.dylib', 'libgfortran.2.0.0.dylib', 'libgfortran.3.dylib',
                  'libncursesw.5.dylib', 'libncurses.5.dylib', 'libpq.5.dylib', 'libpq.5.6.dylib', 'libssl.1.0.0.dylib',
                  ##Dependencies of Tkinter
                  'libfontconfig.1.dylib', 'libxau.6.dylib', 'libxcb.1.dylib', 'libxdmcp.6.dylib',
                  'libxext.6.dylib', 'libxft.2.dylib', 'libxrender.1.dylib',
                  ## Libz is needed by psycopg2
                  re.compile(r'libz.*\.dylib'),
                  ## Already included
                  re.compile(r'libhdf5_hl.*\.dylib'), re.compile(r'libhdf5.*\.dylib')]

# py2app adds some frameworks to package that we also need to check for licenses.
EXCLUDES_FRAMEWORK = ['python.framework']

INIT = ['__init__.py', '__init__.pyc']
EXTRA_SEARCH_FOLDERS = ['site-packages', 'site-packages.zip']
EXTRA_MODULES = {'jquery': '2.1.1',
                 'flot': '0.8.3',
                 'hdf5': '1.8.15',
                 'jit': '2.0.1',
                 'd3': '3',
                 'bct': '43',
                 'python': PYTHON_VERSION,
                 'zlib': '1.0',
                 'mathjax': '2.0'}
ANACONDA_VERSION = "2.3.0"

## These file-name pattern should not be found in TVB distribution.
LICENSE_INTERDICTIONS = [re.compile(".*lzo.*")] ##, re.compile(".*szip.*")]

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
        print 'Warning, win32api not found. All dll versions set to unknown'
    except Exception:
        print 'DLL ' + filename + 'does not contain any information avout version'
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
        module = __import__(str(module_name), globals(), locals(), [])
        if hasattr(module, '__version__'):
            return module.__version__
        if hasattr(module, 'version') and not isinstance(module.version, ModuleType):
            return module.version
        if hasattr(module, 'version') and isinstance(module.version, ModuleType) and hasattr(module.version, 'version'):
            return module.version.version
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
        ### Validate current file not to be in the INTERDICTION list.
        for regex in LICENSE_INTERDICTIONS:
            if regex.match(entry):
                raise Exception("%s file has unacceptable license!!! " % (entry, ))

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
    #TODO: make this work on the py2app build. This will require "switching" to the packages interpreter
    if not sys.platform == 'darwin':
        _find_pkg_modules(modules_dict)

    if 'anaconda' in sys.version.lower() or 'conda' in sys.executable:
        # TODO retrieve this from the build machine
        modules_dict['anaconda'] = ANACONDA_VERSION

    if sys.platform == 'darwin':
        #----- Go into Contents/Frameworks and look for dlybs/so's-----
        path = os.path.split(root_)[0]
        path = os.path.split(path)[0]
        path = os.path.join(os.path.split(path)[0], 'Frameworks')
        if os.path.exists(path):
            _find_modules(path, modules_dict)
    return modules_dict



# Test case for Windows or Mac   
if __name__ == '__main__':
    ROOT = 'D:\Projects\Brain\dist-repo\TVB_distribution\library.zip'
    ROOT_MAC = '../TVB_MacOS_dist/TVB_distribution/tvb.app/Contents/Resources/lib/python2.7'
    print parse_tree_structure(ROOT)

