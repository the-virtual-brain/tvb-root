# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
The distribution process is split in three.

The zero'th phase is to build an anaconda environment with all tvb's dependencies.
This preliminary phase is not used by the mac build as it is not anaconda based.

This script is the first phase.
It should be run when bundled data, documentation or tvb_bin/ scripts change.
This should happen rarely.

The second phase includes the source code and depends on the zip produced by this file.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import os
import shutil
import sys
import requests

import tvb_bin
from subprocess import Popen, PIPE

# source paths
BIN_FOLDER = os.path.dirname(tvb_bin.__file__)
TVB_ROOT = os.path.dirname(os.path.dirname(BIN_FOLDER))
FW_FOLDER = os.path.join(TVB_ROOT, 'tvb_framework')
LICENSE_PATH = os.path.join(FW_FOLDER, 'LICENSE')
RELEASE_NOTES_PATH = os.path.join(TVB_ROOT, 'tvb_documentation', 'RELEASE_NOTES')


DEMOS_MATLAB_FOLDER = os.path.join(TVB_ROOT, 'tvb_documentation', 'matlab')

# dest paths
DIST_FOLDER = os.path.join(os.path.dirname(__file__), 'build', 'TVB_Distribution')


def _copy_demos_collapsed(to_copy):
    """
    Merge multiple src folders, and filter some resources which are not needed (e.g. svn folders)
    """
    for module_path in to_copy.keys():
        destination_folder = to_copy[module_path]
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for sub_folder in os.listdir(module_path):
            src = os.path.join(module_path, sub_folder)
            dest = os.path.join(destination_folder, sub_folder)

            if not os.path.isdir(src) and not os.path.exists(dest):
                if not (sub_folder.startswith('.') or sub_folder.endswith(".rst")):
                    shutil.copy(src, dest)

            if os.path.isdir(src) and not sub_folder.startswith('.')and not sub_folder == 'sandbox' and not os.path.exists(dest):
                ignore_patters = shutil.ignore_patterns('.svn', '*.rst')
                shutil.copytree(src, dest, ignore=ignore_patters)


def fetch_current_revision(branch, token, revision_increment):
    url = f'https://api.github.com/repos/the-virtual-brain/tvb-root/commits'
    params = {'per_page': 1, 'sha': branch}
    resp = requests.get(url, params, headers={'Authorization': f"Bearer {token}"})
    last_link = resp.links.get('last')
    branch_revision = int(last_link['url'].split('&page=')[1])
    return revision_increment + branch_revision


def ensure_tvb_current_revision(branch=None, token=None):
    """
    Enforce later revision number is written in 'tvb.version' file
    """
    import tvb.basic.config
    from tvb.basic.config.settings import VersionSettings
    config_folder = os.path.dirname(os.path.abspath(tvb.basic.config.__file__))

    if branch is None:
        branch = "master"

    print('Current branch {}'.format(branch))
    real_version_number = fetch_current_revision(branch, token, VersionSettings.SVN_GIT_MIGRATION_REVISION)

    with open(os.path.join(config_folder, 'tvb.version'), 'r') as version_file:
        version_line = version_file.read()
        try:
            written_tvb_number = VersionSettings.parse_revision_number(version_line)
        except ValueError:
            written_tvb_number = 0

    if written_tvb_number == real_version_number:
        print("We will not change file tvb.version")
        return

    tvb_version_path = "../tvb_library/tvb/basic/config/tvb.version"
    print("Tvb version file path: {}".format(tvb_version_path))
    paths = [tvb_version_path, os.path.join(config_folder, 'tvb.version')]
    for path in paths:
        with open(path, 'w') as version_file:
            new_text = "Revision: " + str(real_version_number + 1)
            version_file.write(new_text)
            print("Updating tvb.version content in %s to: %s because %d != %d" % (path, new_text, written_tvb_number, real_version_number))


def build_step1():
    # Import only after TVB Revision number has been changed, to get the latest number in generated documentation
    from tvb_build.tvb_documentor.doc_generator import DocGenerator

    build_folder = os.path.dirname(DIST_FOLDER)
    if os.path.exists(build_folder):
        shutil.rmtree(build_folder)
    os.makedirs(DIST_FOLDER)

    # make top level dirs
    top_level_folders = ['docs']
    for d in top_level_folders:
        os.mkdir(os.path.join(DIST_FOLDER, d))

    # make help HTML, PDF manual and documentation site
    print("Starting to populate %s" % DIST_FOLDER)
    doc_generator = DocGenerator(TVB_ROOT, DIST_FOLDER)
    doc_generator.generate_pdfs()
    doc_generator.generate_online_help()
    doc_generator.generate_site()

    shutil.copy2(LICENSE_PATH, os.path.join(DIST_FOLDER, 'LICENSE_TVB.txt'))
    shutil.copy2(RELEASE_NOTES_PATH, os.path.join(DIST_FOLDER, 'docs', 'RELEASE_NOTES.txt'))
    shutil.copytree(DEMOS_MATLAB_FOLDER, os.path.join(DIST_FOLDER, 'matlab'),
                    ignore=shutil.ignore_patterns('.svn', '*.rst'))


    _copy_demos_collapsed({os.path.join("..", "tvb_documentation", "demos"): os.path.join(DIST_FOLDER, "demo_scripts"),
                           os.path.join("..", "tvb_documentation", "tutorials"):
                               os.path.join(DIST_FOLDER, "demo_scripts")})

    shutil.rmtree(os.path.join(DIST_FOLDER, DocGenerator.API))
    shutil.make_archive('TVB_build_step1', 'zip', DIST_FOLDER)
    shutil.rmtree(DIST_FOLDER)
    shutil.move('TVB_build_step1.zip', build_folder)


if __name__ == '__main__':
    branch = None
    if len(sys.argv) > 1:
        branch = sys.argv[1]
        if branch.startswith("origin/"):
            branch = branch.replace("origin/","")
    token = None
    if len(sys.argv) > 2:
        token = sys.argv[2]
    ensure_tvb_current_revision(branch, token)
    build_step1()
