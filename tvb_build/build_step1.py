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
import tvb_bin
import tvb_data
from subprocess import Popen, PIPE

# source paths
BIN_FOLDER = os.path.dirname(tvb_bin.__file__)
TVB_ROOT = os.path.dirname(os.path.dirname(BIN_FOLDER))
FW_FOLDER = os.path.join(TVB_ROOT, 'framework_tvb')
LICENSE_PATH = os.path.join(FW_FOLDER, 'LICENSE')
RELEASE_NOTES_PATH = os.path.join(TVB_ROOT, 'tvb_documentation', 'RELEASE_NOTES')
DATA_SRC_FOLDER = os.path.dirname(tvb_data.__file__)
DEMOS_MATLAB_FOLDER = os.path.join(TVB_ROOT, 'tvb_documentation', 'matlab')

# dest paths
DIST_FOLDER = os.path.join(os.path.dirname(__file__), 'build', 'TVB_Distribution')

DATA_INSIDE_FOLDER = os.path.join(DIST_FOLDER, '_tvb_data')

INCLUDED_INSIDE_DATA = [
    "__init__.py",
    "Default_Project.zip",

    "connectivity/connectivity_76.zip",
    "connectivity/paupau.zip",
    "connectivity/connectivity_66.zip",
    "connectivity/connectivity_192.zip",
    "connectivity/__init__.py",

    "projectionMatrix/projection_eeg_62_surface_16k.mat",
    "projectionMatrix/projection_eeg_65_surface_16k.npy",
    "projectionMatrix/projection_meg_276_surface_16k.npy",
    "projectionMatrix/projection_seeg_588_surface_16k.npy",
    "projectionMatrix/__init__.py",

    "regionMapping/__init__.py",
    "regionMapping/regionMapping_16k_76.txt",
    "regionMapping/regionMapping_80k_80.txt",
    "regionMapping/regionMapping_16k_192.txt",

    "sensors/eeg_unitvector_62.txt.bz2",
    "sensors/eeg_brainstorm_65.txt",
    "sensors/meg_151.txt.bz2",
    "sensors/meg_brainstorm_276.txt",
    "sensors/seeg_39.txt.bz2",
    "sensors/seeg_brainstorm_960.txt",
    "sensors/seeg_588.txt",
    "sensors/__init__.py",

    "surfaceData/__init__.py",
    "surfaceData/cortex_80k.zip",
    "surfaceData/cortex_16384.zip",
    "surfaceData/outer_skin_4096.zip",
    "surfaceData/inner_skull_4096.zip",
    "surfaceData/outer_skull_4096.zip",
    "surfaceData/scalp_1082.zip",
    "surfaceData/face_8614.zip",

    "local_connectivity/__init__.py",
    "local_connectivity/local_connectivity_16384.mat",
    "local_connectivity/local_connectivity_80k.mat",

    "obj/__init__.py",
    "obj/face_surface.obj",
    "obj/eeg_cap.obj",

    "mouse/allen_2mm/Connectivity.h5",
    "mouse/allen_2mm/Volume.h5",
    "mouse/allen_2mm/StructuralMRI.h5",
    "mouse/allen_2mm/RegionVolumeMapping.h5",
]


def _copy_dataset(dataset_files, dataset_destination):
    for pth in dataset_files:
        rel_pth = pth.split('/')
        origin = os.path.join(DATA_SRC_FOLDER, *rel_pth)
        destination = os.path.join(dataset_destination, *rel_pth)
        destination_folder = os.path.dirname(destination)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        print("Copying %s into %s" % (origin, destination))
        shutil.copyfile(origin, destination)


def copy_distribution_dataset():
    """
    Copy the required data file from tvb_data folder:
    - inside TVB library package (for internal usage).
        Will be used during TVB functioning: import default project,
        load default for console profile, or code update events
    - in tvb_data folder, as example for users.
    """
    _copy_dataset(INCLUDED_INSIDE_DATA, DATA_INSIDE_FOLDER)


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


def ensure_svn_current_version():
    """
    Enforce later revision number is written in 'tvb.version' file
    """
    import tvb.basic.config
    from tvb.basic.config.settings import VersionSettings
    config_folder = os.path.dirname(os.path.abspath(tvb.basic.config.__file__))

    svn_variable = 'SVN_REVISION'
    if svn_variable in os.environ:
        real_svn_number = int(os.environ[svn_variable])
    else:
        _proc = Popen(["svnversion", "."], stdout=PIPE)
        real_svn_number = VersionSettings.parse_svn_version(str(_proc.communicate()[0]))

    with open(os.path.join(config_folder, 'tvb.version'), 'r') as version_file:
        version_line = version_file.read()
        try:
            written_svn_number = VersionSettings.parse_svn_version(version_line)
        except ValueError:
            written_svn_number = 0

    if written_svn_number == real_svn_number:
        print("We will not change file tvb.version")
        return

    with open(os.path.join(config_folder, 'tvb.version'), 'w') as version_file:
        new_text = "Revision: " + str(real_svn_number + 1)
        version_file.write(new_text)
        print("Updating tvb.version content to: %s because %d != %d" % (new_text, written_svn_number, real_svn_number))

    # Update SVN_REVISION in the current build, as we are creating a new commit
    os.environ[svn_variable] = str(real_svn_number + 1)
    _proc = Popen(["svn", "commit", "../scientific_library/tvb/basic/config/tvb.version", "-m",
                   "Update SVN revision number automatically from Hudson", "--trust-server-cert"], stdout=PIPE)
    print(_proc.communicate()[0])


def build_step1():
    ## Import only after SVN_REVISION has been changed, to get the latest number in generated documentation
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

    copy_distribution_dataset()

    _copy_demos_collapsed({os.path.join("..", "tvb_documentation", "demos"): os.path.join(DIST_FOLDER, "demo_scripts"),
                           os.path.join("..", "tvb_documentation", "tutorials"):
                               os.path.join(DIST_FOLDER, "demo_scripts")})

    shutil.rmtree(os.path.join(DIST_FOLDER, DocGenerator.API))
    shutil.make_archive('TVB_build_step1', 'zip', DIST_FOLDER)
    shutil.rmtree(DIST_FOLDER)
    shutil.move('TVB_build_step1.zip', build_folder)


if __name__ == '__main__':
    # TODO TVB-2415
    # ensure_svn_current_version()
    build_step1()
