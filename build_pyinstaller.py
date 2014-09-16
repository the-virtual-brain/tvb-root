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
Common part, for building TVB packages for multiple supported OS.
"""

import os
import shutil
import zipfile
import datetime
import matplotlib
import tvb_bin
import tvb.interfaces.web as tvb_web
from subprocess import Popen
from build_base import DIST_FOLDER
from tvb.basic.profile import TvbProfile
from tvb.basic.config.settings import TVBSettings


# The required packages for TVB. 
# We need to copy them separately because of PyInstaller limitations on including
# data files and using them in processes opened by the internal 'python2.6'.
TVB_DEPENDENCIES = ['apscheduler', 'cherrypy', 'cfflib', 'dateutil', 'decorator', 'formencode',
                    'genshi', 'genxmlif', 'h5py', 'idlelib', 'Image', 'lxml', 'matplotlib', 'migrate', 'minixsv',
                    'mpl_toolkits', 'mod_pywebsocket', 'mplh5canvas', 'numpy', 'networkx', 'nibabel', 'numexpr', 'pytz',
                    'psutil', 'psycopg2', 'sqlalchemy', 'simplejson', 'scipy', 'sklearn', 'tables', 'tvb', 'tempita']

# List of files that can be safely removed from both Windows and Linux packages 
# since they are not actually needed.
BASE_EXCLUDE_FILES = []

BASE_EXCLUDE_DIRS = ['lib2to3', 'test', 'config_d', 'config']
# This will be the name under which we copy the actual python interpreter
PYTHON_EXE = TvbProfile.get_python_exe_name()


class PyInstallerPacker():
    """
    A helper class that should handle all PyInstaller related packaging, as well
    as part of the post-processing on the resulted package up to the point where
    the package is ready for license checking, documentation and archiving as ZIP.
    """

    START_TIME = datetime.datetime.now()

    VERSION = TVBSettings.BASE_VERSION

    BIN_FOLDER = os.path.dirname(tvb_bin.__file__)

    ## The root folder in which the various distribution packages will be gathered. 
    ## This folder is just an intermediate and will later become `TVB_Distribution`.
    RESULT_BASE_FOLDER = os.path.join(os.path.dirname(__file__), 'dist-tvb')

    ## A sub-folder that will hold all the gathered data, as to keep resulting package better organized. 
    ## While the top level will also hold scripts, license and documentation, this is only with code.
    DATA_FOLDER_NAME = "tvb_data"


    @staticmethod
    def set_pyinstaller_globals(pyinst_globals):
        """
        This method stores specific PyInstaller classes that are used in the packaging
        process on the PyInstallerPacker class so they can be used later on in this 
        namespace. This is needed since it seems that PyInstaller uses some kind of 
        `eval` mechanism and it's classes are valid only in the *.spec namespace.
        
        :param pyinst_globals: python's globals() dictionary from the *.spec file that
            uses this class for packaging.
         
        NOTE: This was needed in PyInstaller 2.0 version.
        """
        PyInstallerPacker.Tree = pyinst_globals['Tree']
        PyInstallerPacker.Analysis = pyinst_globals['Analysis']
        PyInstallerPacker.PYZ = pyinst_globals['PYZ']
        PyInstallerPacker.TOC = pyinst_globals['TOC']
        PyInstallerPacker.EXE = pyinst_globals['EXE']
        PyInstallerPacker.COLLECT = pyinst_globals['COLLECT']
        ## Prepare specific TVB hook in PyInstaller.
        print "  Copying TVB-hook in PyInstaller..."
        shutil.copy("hook-tvb.py", os.path.join("PyInstaller", "hooks"))


    @staticmethod
    def create_trees_from_dependencies(dep_list):
        """
        Given a dependency list, create pyInstaller specific Tree objects that will be used by 
        the COLLECT class. Needed since PyInstaller misses some dependencies that turn out to be required.
        :param dep_list: a list of packages that are either not included at all, or not fully included
            by default PyInstaller setup. 
        
        For each passed package, extract if it is stored as *.egg then create a PyInstaller Tree
        that will later on be used in the final COLLECT. 
        """
        result = []
        #This new import is required here, as this method will be executed in a different context
        import zipfile
        for one_dep in dep_list:
            module_import = __import__(one_dep, globals(), locals())
            _file_path = module_import.__file__
            tree_path = os.path.dirname(_file_path)
            if not os.path.isdir(tree_path):
                dir_path = os.path.dirname(_file_path)
                _zip_name, actual_module = dir_path.split('.egg')
                actual_module = actual_module[1:]
                _zip_name += '.egg'
                _dest_path = 'temp'
                _source_zip = zipfile.ZipFile(_zip_name, 'r')
                for _name in _source_zip.namelist():
                    if not _name.endswith('.pyc'):
                        _source_zip.extract(_name, _dest_path)
                _source_zip.close()
                #EGG are usually: package_name.egg/package_name/actual_package
                tree_path = os.path.join(_dest_path, actual_module)
                print "Expanding egg for %s and including in TOC tree." % tree_path
            result.append(PyInstallerPacker.Tree(tree_path, prefix=one_dep,
                                                 excludes=['*.pyc', '.svn', '.DStore', '*.svn*', '*.DS*']))
        return result


    @staticmethod
    def gather_tvb_dependencies(extra_dependencies, extra_binaries=None):
        """
        This method does the standard flow that is required in order for 
        PyInstaller to generate a distribution package. 
        
        @param extra_dependencies: a list with any OS specific dependencies that are 
            required extra to the TVB_DEPENDENCIES that are already declared here. 
        @param extra_binaries: a list of pyinstaller specific tuples of the form:
            (dynamic_lib_name, dynamic_lib_filepath, 'BINARY')
        """
        analisys_tvb = PyInstallerPacker.Analysis([os.path.join(PyInstallerPacker.BIN_FOLDER, 'app.py'),
                                                   os.path.join(PyInstallerPacker.BIN_FOLDER, 'git_setup.py'),
                                                   os.path.join(os.path.dirname(tvb_web.__file__), 'run.py')],
                                                  excludes=['cython', "Cython"])

        pyinstaller_pyz = PyInstallerPacker.PYZ(PyInstallerPacker.TOC(analisys_tvb.pure))

        python_tree = PyInstallerPacker.Tree(os.path.dirname(os.__file__),
                                             excludes=['site-packages', '*.pyc*', '*.svn*'])
        if extra_dependencies:
            TVB_DEPENDENCIES.extend(extra_dependencies)
        additional_trees = PyInstallerPacker.create_trees_from_dependencies(TVB_DEPENDENCIES)
        if extra_binaries is not None:
            analisys_tvb.binaries.extend(extra_binaries)

        PyInstallerPacker.COLLECT(analisys_tvb.binaries, analisys_tvb.zipfiles, analisys_tvb.datas,
                                  pyinstaller_pyz, python_tree, *additional_trees, strip=False,
                                  upx=True, name=os.path.join(PyInstallerPacker.RESULT_BASE_FOLDER,
                                                              PyInstallerPacker.DATA_FOLDER_NAME))


    @staticmethod
    def add_tvb_bin_folder(base_folder, data_folder):
        """
        Add our custom 'tvb_bin' python package to the distribution pack.
        """
        bin_package_folder = os.path.join(base_folder, data_folder, 'tvb_bin')
        if os.path.isdir(bin_package_folder):
            shutil.rmtree(bin_package_folder)
        os.mkdir(bin_package_folder)

        for file_n in os.listdir(PyInstallerPacker.BIN_FOLDER):
            if file_n.endswith('.py'):
                shutil.copy(os.path.join(PyInstallerPacker.BIN_FOLDER, file_n), bin_package_folder)


    @staticmethod
    def __copy_matplotlib_code(base_folder, data_folder):
        """
        Pyinstaller has problems adding all required matplotlib data so we
        need to copy it manually. In the process also copy pylab since this is
        also not added properly.
        @param base_folder: the root folder in which the various distribution packages will be
            gathered. This folder is just an intermediate and will later become `TVB_Distribution`.
        @param data_folder: a subfolder that will hold all the gathered data, as to keep resulting
            package better organized. The top level will only hold scripts that use the data from this 
            subfolder, aswell as licensing and documentation.
        """
        mpl_data_folder = os.path.join(base_folder, data_folder, 'mpl-data')
        if os.path.exists(mpl_data_folder):
            shutil.rmtree(mpl_data_folder)
        matplotlib.use('TkAgg')
        import pylab
        shutil.copytree(matplotlib._get_data_path(), mpl_data_folder)
        shutil.copy(pylab.__file__.replace('.pyc', '.py'), os.path.join(base_folder, data_folder, 'pylab.py'))
        shutil.copy(matplotlib.matplotlib_fname(), os.path.join(base_folder, data_folder, 'matplotlibrc'))


    @staticmethod
    def __copy_pkg_resources(base_folder, data_folder):
        """
        PyInstaller is not adding pkg_resources which is needed when we start
        our other python processes.
        @param base_folder: the root folder in which the various distribution packages will be
            gathered. This folder is just an intermediate and will later become `TVB_Distribution`.
        @param data_folder: a subfolder that will hold all the gathered data, as to keep resulting
            package better organized. The top level will only hold scripts that use the data from this 
            subfolder, aswell as licensing and documentation.
        """
        import pkg_resources
        file_path = pkg_resources.__file__
        if not os.path.isdir(os.path.dirname(file_path)):
            zip_name = file_path.split('.egg')[0]
            zip_name += '.egg'
            dest_path = os.path.join('temp', 'pkg_res')
            source_zip = zipfile.ZipFile(zip_name, 'r')
            for name in source_zip.namelist():
                if name.endswith('pkg_resources.py'):
                    source_zip.extract(name, dest_path)
                    file_path = os.path.join(dest_path, name)
            source_zip.close()
        shutil.copy(file_path.replace('.pyc', '.py'), os.path.join(base_folder, data_folder, 'pkg_resources.py'))


    @staticmethod
    def copy_additional_libraries(python_exe_path, extra_includes=None):
        """
        Add the Python executable into the package. Also copy any additional required packages that 
        were not added by PyInstaller but are still needed.
        
        @param python_exe_path: the path from the build machine that should point towards the Python
            executable.
        @param extra_includes: a list with any modules that are for some reason still not included up
            until this point.
        """
        #Add the Python executable to the distribution
        base_folder = PyInstallerPacker.RESULT_BASE_FOLDER
        data_folder = PyInstallerPacker.DATA_FOLDER_NAME
        if not os.path.exists(os.path.join(base_folder, data_folder)):
            os.makedirs(os.path.join(base_folder, data_folder))

        os.mkdir(os.path.join(base_folder, data_folder, 'exe'))
        shutil.copy(python_exe_path, os.path.join(base_folder, data_folder, 'exe', PYTHON_EXE))
        PyInstallerPacker.add_tvb_bin_folder(base_folder, data_folder)
        PyInstallerPacker.__copy_matplotlib_code(base_folder, data_folder)
        PyInstallerPacker.__copy_pkg_resources(base_folder, data_folder)
        if extra_includes:
            for module_string in extra_includes:
                imported_module = __import__(module_string, globals(), locals())
                module_file = imported_module.__file__
                shutil.copy(module_file, os.path.join(base_folder, data_folder, os.path.split(module_file)[1]))


    @staticmethod
    def clean_up_files(exclude_files=None, exclude_dirs=None):
        """
        Since some files and folders are added by pyinstaller even tho they are not needed,
        this is the place to get rid of them.
        
        @param exclude_files: a list of files that should be deleted after the package generation. These
            should have the full file name with extension since we want to delete *.dll, *.so, *.py or *.pyd 
            and only a module name is not enough.
        @param exclude_dirs: a list of directories that should be deleted from the package.
        """
        if exclude_files:
            BASE_EXCLUDE_FILES.extend(exclude_files)
        if exclude_dirs:
            BASE_EXCLUDE_DIRS.extend(exclude_dirs)
        for entry in BASE_EXCLUDE_FILES:
            file_n = os.path.join(PyInstallerPacker.RESULT_BASE_FOLDER, PyInstallerPacker.DATA_FOLDER_NAME, entry)
            if os.path.isfile(file_n):
                os.remove(file_n)
        for entry in BASE_EXCLUDE_DIRS:
            dir_n = os.path.join(PyInstallerPacker.RESULT_BASE_FOLDER, PyInstallerPacker.DATA_FOLDER_NAME, entry)
            if os.path.isdir(dir_n):
                shutil.rmtree(dir_n, True)


    @staticmethod
    def generate_package(package_name):
        """
        Do other required post-processing like checking licenses, adding doc and get the final 
        zip of the distribution.
        
        @param package_name: this will be used in the final zip name. Resulting package will have the 
            name generated as {package_name}{version}_x[32|64]_web.zip
        """
        dist_folder = os.path.join(os.path.dirname(PyInstallerPacker.RESULT_BASE_FOLDER), DIST_FOLDER)
        if os.path.exists(dist_folder):
            shutil.rmtree(dist_folder)
        os.rename(PyInstallerPacker.RESULT_BASE_FOLDER, dist_folder)
        shutil.rmtree('temp', True)
        PyInstallerPacker.generate_final_zip(package_name, PyInstallerPacker.DATA_FOLDER_NAME)


    @staticmethod
    def generate_final_zip(final_name, library_path, extra_check_license_folders=None):
        """
        Now start distribution packer into a new process.
        This was necessary because on Windows, py2exe changes Python compiler to a 
        custom one and some imports fails.
        """
        operation = ["python", "-m", "build_base", "--final_name", final_name,
                     "--library_path", library_path, "--version", PyInstallerPacker.VERSION]
        if extra_check_license_folders:
            operation.append("--extra_licences_check")
            operation.append(extra_check_license_folders)
        process = Popen(operation)
        process.wait()

        if process.returncode != 0:
            raise Exception("Problem encountered while building distribution ZIP file.")
        duration = (datetime.datetime.now() - PyInstallerPacker.START_TIME).seconds

        print 'It took %d seconds to generate distribution ZIP.' % duration


