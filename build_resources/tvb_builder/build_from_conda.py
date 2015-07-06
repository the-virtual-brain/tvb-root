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

import sys
import shutil
import zipfile
import platform
import os.path
from contextlib import closing
from tvb.basic.config.settings import VersionSettings
from tvb.basic.config.environment import Environment
from third_party_licenses.build_licenses import generate_artefact


class Config:
    def __init__(self, platform_name, anaconda_env_path, site_packages_suffix, commands_map, command_factory):
        # System paths:
        self.anaconda_env_path = anaconda_env_path

        dp = os.path.join("..", "..")
        self.tvb_sources = {os.path.join(dp, "framework_tvb", "tvb"): "tvb",
                            os.path.join(dp, "scientific_library", "tvb"): "tvb",
                            os.path.join(dp, "externals", "BCT"): os.path.join("externals", "BCT")}

        # Build result & input
        self.platform_name = platform_name
        self.build_folder = "build"
        self.step1_result = os.path.join(self.build_folder, "TVB_Distribution_a.zip")
        self.target_root = os.path.join(self.build_folder, "TVB_Distribution")
        self.target_before_zip = os.path.join(self.build_folder, "TVB_Build")

        # Inside Distribution paths:
        self.target_library_root = os.path.join(self.target_root, "tvb_data")
        self.target_3rd_licences_folder = os.path.join(self.target_root, 'THIRD_PARTY_LICENSES')
        self.target_site_packages = os.path.join(self.target_library_root, site_packages_suffix)
        self.easy_install_pth = os.path.join(self.target_site_packages, "easy-install.pth")
        self.to_read_licenses_from = [os.path.dirname(self.target_library_root)]

        self.commands_map = commands_map
        self.command_factory = command_factory

    @staticmethod
    def mac64():
        commands_map = {'distribution': '../tvb_data/bin/python -m tvb_bin.app $@',
                        'tvb_start': 'source ./distribution.command start',
                        'tvb_clean': 'source ./distribution.command clean',
                        'tvb_stop': 'source ./distribution.command stop',
                        'contributor_setup': '../tvb_data/bin/python tvb_bin.git_setup $1 $2'}

        return Config("MacOS", "/anaconda/envs/tvb-run3", os.path.join("lib", "python2.7", "site-packages"),
                      commands_map,
                      _create_mac_command)

    @staticmethod
    def win64():
        set_path = 'cd ..\\tvb_data \n' + \
                   'set PATH=%cd%;%path%; \n'
                   # 'set PYTHONPATH=%cd%; \n' + \
                   # 'set PYTHONHOME=%cd%; \n'

        commands_map = {'distribution': set_path + 'python.exe -m tvb_bin.app %1 %2 %3 %4 %5 %6\ncd ..\\bin',
                        'tvb_start': 'distribution start',
                        'tvb_clean': 'distribution clean',
                        'tvb_stop': 'distribution stop',
                        'contributor_setup': set_path + 'python.exe -m  tvb_bin.git_setup %1 %2\ncd ..\\bin'}

        return Config("Windows", "C:\\anaconda\\envs\\tvb-run", os.path.join("Lib", "site-packages"), commands_map,
                      _create_windows_script)


def _log(indent, msg):
    """
    Produce a feedback about the current build procedure.
    """
    if indent == 1:
        print " - ", msg
    else:
        print "  " * indent, msg


def _compress(folder_to_zip, result_name):
    """
    Create ZIP archive from folder
    """
    assert os.path.isdir(folder_to_zip)
    with closing(zipfile.ZipFile(result_name, "w", zipfile.ZIP_DEFLATED)) as z_file:
        for root, _, files in os.walk(folder_to_zip):
            for file_nname in files:
                absfn = os.path.join(root, file_nname)
                zfn = absfn[len(folder_to_zip) + len(os.sep):]
                z_file.write(absfn, zfn)


def _copy_collapsed(config):
    """
    Merge multiple src folders, and filter some resources which are not needed (tests, docs, svn folders)
    """
    for module_path, suffix in config.tvb_sources.iteritems():
        destination_folder = os.path.join(config.target_site_packages, suffix)
        _log(2, module_path + " --> " + destination_folder)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for sub_folder in os.listdir(module_path):
            src = os.path.join(module_path, sub_folder)
            dest = os.path.join(destination_folder, sub_folder)

            if not os.path.isdir(src) and not os.path.exists(dest):
                shutil.copy(src, dest)

            if os.path.isdir(src) and not (sub_folder.startswith('.')
                                           or sub_folder.startswith("tests")) and not os.path.exists(dest):
                ignore_patters = shutil.ignore_patterns('.svn', "tutorials")
                shutil.copytree(src, dest, ignore=ignore_patters)

            simulator_doc_folder = os.path.join(destination_folder, "simulator", "doc")
            if os.path.exists(simulator_doc_folder):
                shutil.rmtree(simulator_doc_folder, True)
                _log(3, "Removed: " + str(simulator_doc_folder))


def _create_mac_command(target_bin_folder, command_file_name, command):
    """
    Private script which adds the common part of a command file.
    """
    _log(2, command_file_name)
    pth = os.path.join(target_bin_folder, command_file_name + ".command")

    with open(pth, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('cd "$(dirname "$0")"\n')
        f.write('echo "' + command_file_name + '"\n')
        f.write(command + "\n")
        f.write('echo "Done."\n')


def _create_windows_script(target_bin_folder, command_file_name, command):
    """
    Private script which generated a command file inside tvb-bin distribution folder.
    """
    _log(2, command_file_name)
    pth = os.path.join(target_bin_folder, command_file_name + '.bat')

    with open(pth, 'w') as f:
        f.write('@echo off \n')
        f.write('rem Executing ' + command_file_name + ' \n')
        f.write(command + ' \n')
        f.write('echo "Done."\n')
    os.chmod(pth, 0775)


def _compute_final_zip_name(platform_name):
    """
    Compute resulting ZIP name
    """
    architecture = '_x32_'
    if sys.maxint > 2 ** 32 or platform.architecture()[0] == '64bit':
        architecture = '_x64_'
    return os.path.join("build",
                        "TVB_" + platform_name + "_" + VersionSettings.BASE_VERSION + architecture + "web.zip")


def _modify_pth(pth_name):
    """
    Replace tvb links with paths
    """
    tvb_markers = ["tvb_root", "tvb-root", "framework_tvb", "scientific_library", "third_party_licenses", "tvb_data",
                   "Hudson"]
    tvb_replacement = "./tvb\n./tvb_bin\n./tvb_data\n"
    _log(1, "Modifying PTH " + pth_name)
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
    # Cleanup
    shutil.rmtree(config.target_root, True)
    final_zip_name = _compute_final_zip_name(config.platform_name)
    if os.path.exists(final_zip_name):
        os.remove(final_zip_name)
    # Unzip skeleton TVB_Distribution
    _log(1, "Decompressing " + config.step1_result + " into '" + config.build_folder + "' ...")
    zipfile.ZipFile(config.step1_result).extractall(config.build_folder)

    # Copy anaconda ENV folder
    _log(1, "Copying anaconda ENV " + config.anaconda_env_path + " into '" + config.target_library_root + "'...")
    shutil.copytree(config.anaconda_env_path, config.target_library_root)

    # Copy tvb sources
    _log(1, "Copying TVB sources into site-packages ...")
    _copy_collapsed(config)

    bin_src = os.path.join(config.target_root, "_tvb_bin")
    bin_dst = os.path.join(config.target_site_packages, "tvb_bin")
    _log(2, "Moving " + bin_src + " to " + bin_dst)
    os.rename(bin_src, bin_dst)

    demo_data_src = os.path.join(config.target_root, "_tvb_data")
    demo_data_dst = os.path.join(config.target_site_packages, "tvb_data")
    _log(2, "Moving " + demo_data_src + " to " + demo_data_dst)
    os.rename(demo_data_src, demo_data_dst)

    online_help_src = os.path.join(config.target_root, "_help")
    online_help_dst = os.path.join(config.target_site_packages, "tvb", "interfaces", "web", "static", "help")
    _log(2, "Moving " + online_help_src + " to " + online_help_dst)
    os.rename(online_help_src, online_help_dst)

    # Modify easy_install.pth
    _modify_pth(config.easy_install_pth)

    # write command scripts for current OS in BIN folder
    _log(1, "Creating command files:")
    for key, value in config.commands_map.iteritems():
        config.command_factory(os.path.join(config.target_root, "bin"), key, value)

    # introspect licenses from distribution
    zip_name = generate_artefact(config.target_site_packages, extra_licenses_check=config.to_read_licenses_from)
    zipfile.ZipFile(zip_name).extractall(config.target_3rd_licences_folder)
    os.remove(zip_name)

    if os.path.exists(config.target_before_zip):
        shutil.rmtree(config.target_before_zip, True)
    os.mkdir(config.target_before_zip)
    shutil.move(config.target_root, config.target_before_zip)
    _compress(config.target_before_zip, final_zip_name)
    shutil.rmtree(config.target_before_zip, True)


if __name__ == "__main__":

    if Environment.is_mac():
        prepare_anaconda_dist(Config.mac64())
    elif Environment.is_windows():
        prepare_anaconda_dist(Config.win64())
