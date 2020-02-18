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
import glob
import shutil
import zipfile
import os.path
from os.path import join
from tvb.basic.config.settings import VersionSettings
from tvb.basic.config.environment import Environment
from tvb_build.third_party_licenses.build_licenses import generate_artefact


class Config:
    def __init__(self, platform_name, anaconda_env_path, site_packages_suffix, commands_map, command_factory):
        # System paths:
        self.anaconda_env_path = anaconda_env_path

        # Build result & input
        self.platform_name = platform_name
        self.build_folder = "build"
        # the step 1 zip is expected to be placed in this path. Hudson will place it there via the copy artifact plugin.
        self.step1_result = join(self.build_folder, "TVB_build_step1.zip")
        self.target_root = join(self.build_folder, "TVB_Distribution")
        self.target_before_zip = join(self.build_folder, "TVB_Build")

        # Inside Distribution paths:
        self.target_library_root = join(self.target_root, "tvb_data")
        self.target_3rd_licences_folder = join(self.target_root, 'THIRD_PARTY_LICENSES')
        self.target_site_packages = join(self.target_library_root, site_packages_suffix)
        self.easy_install_pth = join(self.target_site_packages, "easy-install.pth")
        self.to_read_licenses_from = [os.path.dirname(self.target_library_root)]

        # TVB sources and specify where to copy them in distribution
        self.tvb_sources = {
            join("..", "framework_tvb", "tvb"): join(self.target_site_packages, "tvb"),
            join("..", "scientific_library", "tvb"): join(self.target_site_packages, "tvb"),
            join("..", "tvb_bin", "tvb_bin"): join(self.target_site_packages, "tvb_bin"),
            join("..", "externals", "BCT"): join(self.target_site_packages, "externals", "BCT"),
        }

        self.commands_map = commands_map
        self.command_factory = command_factory

        self.artifact_name = "TVB_" + platform_name + "_" + VersionSettings.BASE_VERSION + ".zip"
        _artifact_glob = "TVB_" + platform_name + "_*.zip"
        self.artifact_glob = join(self.build_folder, _artifact_glob)  # this is used to match old artifacts
        self.artifact_pth = join(self.build_folder, self.artifact_name)


    @staticmethod
    def mac64():
        # TODO check paths
        set_path = 'cd ../tvb_data \n' + \
                   'export PATH=`pwd`/bin:$PATH \n' + \
                   'export PYTHONPATH=`pwd`/lib/' + Environment.PYTHON_FOLDER + \
                   ':`pwd`/lib/' + Environment.PYTHON_FOLDER + '/site-packages \n' + \
                   'export PYTHONIOENCODING=utf8 \n' + \
                   'unset PYTHONHOME \n\n' + \
                   '# export TVB_USER_HOME=`pwd` \n'

        commands_map = {
            'bin/distribution.command': set_path + '../tvb_data/bin/python -m tvb_bin.app $@',
            'bin/tvb_start.command': 'source ./distribution.command start',
            'bin/tvb_clean.command': 'source ./distribution.command clean',
            'bin/tvb_stop.command': 'source ./distribution.command stop',
            'bin/jupyter_notebook.sh': set_path + '../tvb_data/bin/python -m tvb_bin.run_jupyter notebook ../demo_scripts',
            'demo_scripts/jupyter_notebook.sh': set_path + '../tvb_data/bin/python -m tvb_bin.run_jupyter notebook'
        }

        return Config("MacOS", "/anaconda/envs/tvb-run3", join("lib", Environment.PYTHON_FOLDER, "site-packages"),
                      commands_map, _create_unix_command)


    @staticmethod
    def win64():
        set_path = 'cd ..\\tvb_data \n' + \
                   'set PATH=%cd%;%cd%\\Library\\mingw-w64\\bin;%cd%\\Library\\bin;%cd%\\Scripts;%path%; \n' + \
                   'set PYTHONPATH=%cd%\\Lib;%cd%\\Lib\\site-packages \n' + \
                   'set PYTHONIOENCODING=utf8 \n' + \
                   'set PYTHONHOME=\n\n' + \
                   'REM set TVB_USER_HOME=%cd% \n'

        commands_map = {
            'bin\\distribution.bat': set_path + 'python.exe -m tvb_bin.app %1 %2 %3 %4 %5 %6\ncd ..\\bin',
            'bin\\tvb_start.bat': 'distribution start',
            'bin\\tvb_clean.bat': 'distribution clean',
            'bin\\tvb_stop.bat': 'distribution stop',
            'bin\\jupyter_notebook.bat': set_path + 'cd ..\\bin\n..\\tvb_data\\Scripts\\ipython notebook ..\\demo_scripts',
            'demo_scripts\\jupyter_notebook.bat': set_path + 'cd ..\\demo_scripts\n..\\tvb_data\\Scripts\\ipython notebook'
        }

        return Config("Windows", "C:\\Users\\tvb\\.conda\\envs\\tvb-run",
                      join("Lib", "site-packages"), commands_map, _create_windows_script)


    @staticmethod
    def linux64():
        set_path = 'cd ../tvb_data \n' + \
                   'export PATH=`pwd`/bin:$PATH \n' + \
                   'export PYTHONPATH=`pwd`/lib/' + Environment.PYTHON_FOLDER + \
                   ':`pwd`/lib/' + Environment.PYTHON_FOLDER + '/site-packages \n' + \
                   'export PYTHONIOENCODING=utf8 \n' + \
                   'unset PYTHONHOME \n\n' + \
                   '# export TVB_USER_HOME=`pwd` \n'

        for env_name in ["LD_LIBRARY_PATH", "LD_RUN_PATH"]:
            set_path += "if [ ${" + env_name + "+1} ]; then\n" + \
                        "  export " + env_name + "=`pwd`/lib:`pwd`/bin:$" + env_name + "\n" + \
                        "else\n" + \
                        "  export " + env_name + "=`pwd`/lib:`pwd`/bin\n" + \
                        "fi\n"

        commands_map = {
            'bin/distribution.sh': set_path + './bin/python -m tvb_bin.app $@\ncd ../bin',
            'bin/tvb_start.sh': 'bash ./distribution.sh start',
            'bin/tvb_clean.sh': 'bash ./distribution.sh clean',
            'bin/tvb_stop.sh': 'bash ./distribution.sh stop',
            'bin/jupyter_notebook.sh': set_path + 'cd ../bin\n../tvb_data/bin/python -m tvb_bin.run_jupyter notebook ../demo_scripts',
            'demo_scripts/jupyter_notebook.sh': set_path + 'cd ../demo_scripts\n../tvb_data/bin/python -m tvb_bin.run_jupyter notebook'
        }

        return Config("Linux", "/opt/conda/envs/tvb-run", join("lib", Environment.PYTHON_FOLDER, "site-packages"),
                      commands_map, _create_unix_command)


def _log(indent, msg):
    """
    Produce a feedback about the current build procedure.
    """
    if indent == 1:
        print(" - ", msg)
    else:
        print("  " * indent, msg)


def _compress(folder_to_zip, result_name):
    """
    Create ZIP archive from folder
    """
    assert os.path.isdir(folder_to_zip)
    with zipfile.ZipFile(result_name, "w", zipfile.ZIP_DEFLATED) as z_file:
        for root, _, files in os.walk(folder_to_zip):
            for file_nname in files:
                absfn = join(root, file_nname)
                zfn = absfn[len(folder_to_zip) + len(os.sep):]
                z_file.write(absfn, zfn)


def _copy_collapsed(config):
    """
    Merge multiple src folders, and filter some resources which are not needed (tests, docs, svn folders)
    """
    for module_path, destination_folder in config.tvb_sources.items():
        _log(2, module_path + " --> " + destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for sub_folder in os.listdir(module_path):
            src = join(module_path, sub_folder)
            dest = join(destination_folder, sub_folder)

            if not os.path.isdir(src) and not os.path.exists(dest):
                shutil.copy(src, dest)

            if os.path.isdir(src) and not (sub_folder.startswith('.')
                                           or sub_folder.startswith("tests")) and not os.path.exists(dest):
                ignore_patters = shutil.ignore_patterns('.svn', '*.ipynb', 'tutorials')
                shutil.copytree(src, dest, ignore=ignore_patters)

            for excluded in [join(destination_folder, "simulator", "doc"),
                             join(destination_folder, "simulator", "demos")]:
                if os.path.exists(excluded):
                    shutil.rmtree(excluded, True)
                    _log(3, "Removed: " + str(excluded))


def _create_unix_command(target_file, command):
    """
    Private script which adds the common part of a command file.
    """
    _log(2, target_file)

    with open(target_file, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('cd "$(dirname "$0")"\n')
        f.write('echo "Executing ' + os.path.split(target_file)[1] + '"\n')
        f.write(command + "\n")
        f.write('echo "Done."\n')
    os.chmod(target_file, 0o755)


def _create_windows_script(target_file, command):
    """
    Private script which generates a windows launch script.
    """
    _log(2, target_file)

    with open(target_file, 'w') as f:
        f.write('@echo off \n')
        f.write('echo "Executing ' + os.path.split(target_file)[1] + '" \n')
        f.write(command + ' \n')
        f.write('echo "Done."\n')
    os.chmod(target_file, 0o755)


def _modify_pth(pth_name):
    """
    Replace tvb links with paths
    """
    tvb_markers = ["tvb_root", "tvb-root", "framework_tvb", "scientific_library", "third_party_licenses", "tvb_data",
                   "Hudson", "hudson"]
    tvb_replacement = "./tvb\n./tvb_bin\n./tvb_data\n"
    new_content = ""
    first_tvb_replace = True

    with open(pth_name) as fp:
        for line in fp:
            is_tvb = False
            for m in tvb_markers:
                if m in line:
                    is_tvb = True
                    break
            if is_tvb:
                _log(3, "ignoring " + line)
                if first_tvb_replace:
                    new_content += tvb_replacement
                    first_tvb_replace = False
                continue
            new_content += line

    _log(2, "PTH result: \n" + new_content)
    with open(pth_name, 'w') as fw:
        fw.write(new_content)


def prepare_anaconda_dist(config):
    """
    Main method for building from Anaconda (This requires TVB_Distribution - step1 ZIP to have been generated before
    """
    _log(0, "Generating ANACONDA-based TVB_Distribution! " + config.platform_name)

    _log(1, "Removing old artifacts")
    for ar in glob.glob(config.artifact_glob):
        os.remove(ar)
    shutil.rmtree(config.target_root, True)

    _log(1, "Decompressing " + config.step1_result + " into '" + config.target_root + "' ...")
    zipfile.ZipFile(config.step1_result).extractall(config.target_root)

    # make needed directory structure that is not in the step1 zip
    # bin dir is initially empty, step1 does not support empty dirs in the zip
    os.mkdir(join(config.target_root, 'bin'))

    _log(1, "Copying anaconda ENV folder" + config.anaconda_env_path + " into '" + config.target_library_root + "'...")
    shutil.copytree(config.anaconda_env_path, config.target_library_root)

    _log(1, "Copying TVB sources into site-packages ...")
    _copy_collapsed(config)

    demo_data_src = join(config.target_root, "_tvb_data")
    demo_data_dst = join(config.target_site_packages, "tvb_data")
    _log(2, "Moving " + demo_data_src + " to " + demo_data_dst)
    os.rename(demo_data_src, demo_data_dst)

    online_help_src = join(config.target_root, "_help")
    online_help_dst = join(config.target_site_packages, "tvb", "interfaces", "web", "static", "help")
    _log(2, "Moving " + online_help_src + " to " + online_help_dst)
    os.rename(online_help_src, online_help_dst)

    _log(1, "Modifying PTH " + config.easy_install_pth)
    _modify_pth(config.easy_install_pth)

    _log(1, "Creating command files:")
    for target_file, content in config.commands_map.items():
        config.command_factory(join(config.target_root, target_file), content)

    _log(1, "Introspecting 3rd party licenses...")
    zip_name = generate_artefact(config.target_site_packages, extra_licenses_check=config.to_read_licenses_from)
    zipfile.ZipFile(zip_name).extractall(config.target_3rd_licences_folder)
    os.remove(zip_name)

    _log(1, "Packing final ZIP...")
    if os.path.exists(config.target_before_zip):
        shutil.rmtree(config.target_before_zip, True)
    os.mkdir(config.target_before_zip)
    shutil.move(config.target_root, config.target_before_zip)
    _compress(config.target_before_zip, config.artifact_pth)
    shutil.rmtree(config.target_before_zip, True)
    _log(1, "Done TVB package " + config.artifact_pth)


if __name__ == "__main__":

    if Environment.is_mac():
        prepare_anaconda_dist(Config.mac64())

    elif Environment.is_windows():
        prepare_anaconda_dist(Config.win64())

    else:
        prepare_anaconda_dist(Config.linux64())
