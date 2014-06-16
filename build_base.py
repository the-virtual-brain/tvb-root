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
This module simply creates distribution package (ZIP).

.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

from __future__ import with_statement
import os
import sys
import shutil
import tvb_data
import platform
from glob import glob
from contextlib import closing
from zipfile import ZipFile, ZIP_DEFLATED
from subprocess import Popen, PIPE
from optparse import OptionParser
from tvb_documentor.doc_generator import DocGenerator


FW_FOLDER = "framework_tvb"
DIST_FOLDER = "dist"
DOCS_RESULT_FOLDER = "docs"

FOLDERS_TO_DELETE = ['.svn', '.project', '.settings']
FILES_TO_DELETE = ['.DS_Store', 'dev_logger_config.conf']

RELEASE_NOTES_PATH = os.path.join("tvb_documentation", 'RELEASE_NOTES')
LICENSE_PATH = os.path.join(FW_FOLDER, "LICENSE_TVB.txt")
DIST_FOLDER_FINAL = "TVB_Distribution"



def generate_distribution(final_name, library_path, version, extra_licensing_check=None):
    """ 
    Clean files, generate final ZIP, with 3rd party licenses included.
    """
    print "- Adding Docs, Demo Data and Externals..."

    current_folder = os.path.dirname(__file__)
    dist_folder = os.path.join(current_folder, DIST_FOLDER)
    if not os.path.exists(dist_folder):
        os.mkdir(dist_folder)

    # Copy library code before doc generation in order to avoid merging folders
    copy_simulator_library(os.path.join(DIST_FOLDER, library_path))
    os.mkdir(os.path.join(dist_folder, DOCS_RESULT_FOLDER))
    # Now generate TVB manuals and Online Help
    doc_generator = DocGenerator(current_folder, dist_folder, os.path.join(DIST_FOLDER, library_path))
    doc_generator.generate_all_docs()

    shutil.copy2(LICENSE_PATH, os.path.join(DIST_FOLDER, 'LICENSE_TVB.txt'))
    shutil.copy2(RELEASE_NOTES_PATH, os.path.join(DIST_FOLDER, DOCS_RESULT_FOLDER, 'RELEASE_NOTES.txt'))
    shutil.copytree(os.path.join("externals", "BCT"), os.path.join(DIST_FOLDER, library_path, "externals", "BCT"))
    copy_distribution_dataset(DIST_FOLDER, library_path)
    write_svn_current_version(os.path.join(DIST_FOLDER, library_path))

    print "- Cleaning up non-required files..."
    clean_up(dist_folder, False)
    if os.path.exists(DIST_FOLDER_FINAL):
        shutil.rmtree(DIST_FOLDER_FINAL)
    os.rename(DIST_FOLDER, DIST_FOLDER_FINAL)
    shutil.rmtree('tvb.egg-info', True)
    shutil.rmtree('build', True)
    for file_zip in glob('*.zip'):
        os.unlink(file_zip)

    print "- Creating required folder structure..."
    if os.path.exists(final_name):
        shutil.rmtree(final_name)
    os.mkdir(final_name)
    shutil.move(DIST_FOLDER_FINAL, final_name)

    if extra_licensing_check:
        extra_licensing_check = extra_licensing_check.split(';')
        for idx in xrange(len(extra_licensing_check)):
            extra_licensing_check[idx] = os.path.join(final_name, DIST_FOLDER_FINAL, extra_licensing_check[idx])
    introspect_licenses(os.path.join(final_name, DIST_FOLDER_FINAL, 'THIRD_PARTY_LICENSES'),
                        os.path.join(final_name, DIST_FOLDER_FINAL, library_path), extra_licensing_check)
    print "- Creating the ZIP folder of the distribution..."
    architecture = '_x32_'
    if sys.maxint > 2 ** 32 or platform.architecture()[0] == '64bit':
        architecture = '_x64_'
    zip_name = final_name + "_" + version + architecture + "web.zip"
    if os.path.exists(zip_name):
        os.remove(zip_name)
    zipdir(final_name, zip_name)
    if os.path.exists(final_name):
        shutil.rmtree(final_name)
    print '- Finish creation of distribution ZIP'



def copy_distribution_dataset(dist_path, library_path):
    """
    Copy the required data file from tvb_data folder:
        - inside TVB library package (for internal usage).
        Will be used during TVB functioning at default-project create action.
        - in tvb_data folder, as example for users.
    """
    included_data = [("__init__.py", "$INSIDE.__init__.py"),
                     ("cff.dataset_74.cff", "$OUTSIDE.dataset_74.cff"),
                     ("connectivity.connectivity_96.zip", "$OUTSIDE.connectivity_regions_96.zip"),
                     ("sensors.EEG_unit_vectors_BrainProducts_62.txt.bz2", "$OUTSIDE.EEG_Sensors.txt.bz2"),
                     ("sensors.meg_channels_reg13.txt.bz2", "$OUTSIDE.MEG_Sensors.txt.bz2"),

                     ("cff.dataset_74.cff", "$INSIDE.cff.dataset_74.cff"),
                     ("cff.__init__.py", "$INSIDE.cff.__init__.py"),

                     ("connectivity.connectivity_74.zip", "$INSIDE.connectivity.connectivity_74.zip"),
                     ("connectivity.dti_pipeline_regions.txt", "$INSIDE.connectivity.dti_pipeline_regions.txt"),
                     ("connectivity.__init__.py", "$INSIDE.connectivity.__init__.py"),

                     ("projectionMatrix.surface_reg_13_eeg_62.mat", "$INSIDE.projectionMatrix.surface_reg_13_eeg_62.mat"),
                     ("projectionMatrix.region_conn_74_eeg_1020_62.mat", "$INSIDE.projectionMatrix.region_conn_74_eeg_1020_62.mat"),
                     ("projectionMatrix.__init__.py", "$INSIDE.projectionMatrix.__init__.py"),

                     ("sensors.EEG_unit_vectors_BrainProducts_62.txt.bz2", "$INSIDE.sensors.EEG_unit_vectors_BrainProducts_62.txt.bz2"),
                     ("sensors.meg_channels_reg13.txt.bz2", "$INSIDE.sensors.meg_channels_reg13.txt.bz2"),
                     ("sensors.internal_39.txt.bz2", "$INSIDE.sensors.internal_39.txt.bz2"),
                     ("sensors.__init__.py", "$INSIDE.sensors.__init__.py"),

                     ("surfaceData.__init__.py", "$INSIDE.surfaceData.__init__.py"),
                     ("surfaceData.cortex_reg13.surface_cortex_reg13.zip", "$INSIDE.surfaceData.cortex_reg13.surface_cortex_reg13.zip"),
                     ("surfaceData.outer_skin_4096.zip", "$INSIDE.surfaceData.outer_skin_4096.zip"),
                     ("surfaceData.inner_skull_4096.zip", "$INSIDE.surfaceData.inner_skull_4096.zip"),
                     ("surfaceData.outer_skull_4096.zip", "$INSIDE.surfaceData.outer_skull_4096.zip"),
                     ("surfaceData.eeg_skin_surface.zip", "$INSIDE.surfaceData.eeg_skin_surface.zip"),
                     ("surfaceData.face_surface_old.zip", "$INSIDE.surfaceData.face_surface_old.zip"),
                     ("surfaceData.cortex_reg13.all_regions_cortex_reg13.txt", "$INSIDE.surfaceData.cortex_reg13.all_regions_cortex_reg13.txt"),
                     ("surfaceData.cortex_reg13.local_connectivity_surface_cortex_reg13.mat", "$INSIDE.surfaceData.cortex_reg13.local_connectivity_surface_cortex_reg13.mat"),

                     ("obj.__init__.py", "$INSIDE.obj.__init__.py"),
                     ("obj.face_surface.obj", "$INSIDE.obj.face_surface.obj"), ]

    source_folder = os.path.dirname(tvb_data.__file__)
    ### Copy demo-data inside TVB package, for usage from code (e.g. default import when creating a new project).
    destination_folder_inside = os.path.join(dist_path, library_path, 'tvb_data')
    ### Copy in User's visible area, all in one folder.
    top_destination_folder = os.path.join(dist_path, "demo_data")

    for coded_origin, coded_destination in included_data:
        origin = _prepare_name(coded_origin, destination_folder_inside, top_destination_folder, source_folder)
        destination = _prepare_name(coded_destination, destination_folder_inside, top_destination_folder, source_folder)
        shutil.copyfile(origin, destination)



def _prepare_name(coded_name, inside_folder, outside_folder, origin_folder):
    """
    Convert in absolute path and create folder if needed.
    """
    parts = coded_name.split(".")
    if parts[-1] == "bz2":
        parts[-2] = parts[-2] + "." + parts[-1]
        parts = parts[:-1]
    parts[-2] = parts[-2] + "." + parts[-1]
    parts = parts[:-1]

    if parts[0] == "$INSIDE":
        parts[0] = inside_folder
    elif parts[0] == "$OUTSIDE":
        parts[0] = outside_folder
    else:
        return os.path.join(origin_folder, *parts)

    exact_file = os.path.join(parts[0], *parts[1:])
    destination_folder = os.path.dirname(exact_file)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    return exact_file



def write_svn_current_version(library_folder):
    """Read current subversion number"""
    try:
        svn_variable = 'SVN_REVISION'
        if svn_variable in os.environ:
            version = os.environ[svn_variable]
        else:
            _proc = Popen(["svnversion", "."], stdout=PIPE)
            version = _proc.communicate()[0]
        file_ = open(os.path.join(library_folder, 'tvb_bin', 'tvb.version'), 'w')
        file_.write(version)
        file_.close()
    except Exception, excep:
        print "-- W: Could not get or persist revision number because: ", excep



def copy_simulator_library(library_folder):
    """
    Make sure all TVB folders are collapsed together in one folder in distribution.
    """
    import tvb

    destination_folder = os.path.join(library_folder, 'tvb')
    for module_path in tvb.__path__:
        for sub_folder in os.listdir(module_path):
            src = os.path.join(module_path, sub_folder)
            dest = os.path.join(destination_folder, sub_folder)
            if os.path.isdir(src) and not (sub_folder.startswith('.')
                                           or sub_folder.startswith("tests")) and not os.path.exists(dest):
                print "  Copying TVB: " + str(src)
                shutil.copytree(src, dest)

    tests_folder = os.path.join(destination_folder, "tests")
    if os.path.exists(tests_folder):
        shutil.rmtree(tests_folder, True)
        print "  Removed: " + str(tests_folder)

    simulator_doc_folder = os.path.join(destination_folder, "simulator", "doc")
    if os.path.exists(simulator_doc_folder):
        shutil.rmtree(simulator_doc_folder, True)
        print "  Removed: " + str(simulator_doc_folder)



def introspect_licenses(destination_folder, root_introspection, extra_licenses_check=None):
    """Generate archive with 3rd party licenses"""
    print "- Introspecting for dependencies..." + str(root_introspection)
    import locale

    try:
        locale.getdefaultlocale()
    except Exception:
        os.environ['LANG'] = 'en_US.UTF-8'
        os.environ['LC_ALL'] = 'en_US.UTF-8'
    from third_party_licenses.build_licenses import generate_artefact

    zip_name = generate_artefact(root_introspection, extra_licenses_check=extra_licenses_check)
    ZipFile(zip_name).extractall(destination_folder)
    os.remove(zip_name)
    print "- Dependencies archive with licenses done."



def zipdir(basedir, archivename):
    """Create ZIP archive from folder"""
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z_file:
        for root, _, files in os.walk(basedir):
            #NOTE: ignore empty directories
            for file_nname in files:
                absfn = os.path.join(root, file_nname)
                zfn = absfn[len(basedir) + len(os.sep):]
                z_file.write(absfn, zfn)



def clean_up(folder_path, to_delete):
    """
    Remove any read only permission for certain files like those in .svn, then delete the files.
    """
    #Add Write access on folder
    folder_name = os.path.split(folder_path)[1]
    will_delete = False
    os.chmod(folder_path, 0o777)
    if to_delete or folder_name in FOLDERS_TO_DELETE:
        will_delete = True

    #step through all the files/folders and change permissions
    for file_ in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_)
        os.chmod(file_path, 0o777)
        #if it is a directory, do a recursive call
        if os.path.isdir(file_path):
            clean_up(file_path, to_delete or will_delete)
        #for files merely call chmod 
        else:
            if file_ in FILES_TO_DELETE:
                os.remove(file_path)

    if to_delete or will_delete:
        shutil.rmtree(folder_path)



if __name__ == '__main__':
    USAGE = "usage: %prog [options]"
    PARSER = OptionParser(usage=USAGE)
    PARSER.add_option("-n", "--final_name", dest="name", help="ZIP file name", metavar="FILE")
    PARSER.add_option("-l", "--library_path", dest="library_path", help="Distribution library path")
    PARSER.add_option("-v", "--version", dest="version", help="TVB Version")
    PARSER.add_option("-x", "--extra_licences_check", dest="extra_licenses_check",
                      help="A ; separated list of folders relative to the package root to check for valid licensing.")
    (OPTIONS, _ARGS) = PARSER.parse_args()

    if OPTIONS.name is None:
        raise Exception("Please provide a name for current distribution")

    if OPTIONS.library_path is None:
        raise Exception("Please provide path to library folder for current distribution")

    if OPTIONS.version is None:
        raise Exception("Please provide version for current distribution")

    generate_distribution(OPTIONS.name, OPTIONS.library_path, OPTIONS.version, OPTIONS.extra_licenses_check)
