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
Generate PDF or HTML with third party licenses.
"""
import docutils.core
import os
from tvb_build import third_party_licenses
import tvb_build.third_party_licenses.deps_xml_parser as parser
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
from tvb_build.third_party_licenses.package_finder import parse_tree_structure

CURRENT_FOLDER = third_party_licenses.__path__[0]
LICENSES_FOLDER = "license"

RESULT_FILE_NAME = os.path.join(CURRENT_FOLDER, "_THIRD_PARTY_LICENSES")
PACKAGES_USED_XML = os.path.join(CURRENT_FOLDER, "..", "..", "..", "packages_used.xml")

FULL_TEMPLATE = "===========================\
                \nThe Virtual Brain Project\
                \n===========================\
                \n\n" + \
                ("------------------------------------------------------" * 10) + \
                "\nThe Virtual Brain Project uses a number of third party \
                  packages. This document lists these packages together with \
                  their copyrights, links to the respective projects to obtain\
                   sources and the original license texts. These packages are \
                   distributed with TVB for convenience and are not \
                   licensed under the TVB-License, but remain under their \
                   respective licenses.\n" + \
                ("------------------------------------------------------" * 10) + \
                "\n\n$$DEPENDENCIES$$\n\
                \n.. _BSD: http://en.wikipedia.org/wiki/BSD_licenses\
                \n.. _MIT: http://opensource.org/licenses/mit-license.php\
                \n.. _Apache: http://www.apache.org/licenses/LICENSE-2.0.html\
                \n.. _ZPL: https://opensource.org/licenses/ZPL-2.0\
                \n.. _PSF: http://docs.python.org/license.html\
                \n.. _LGPLv2: http://www.gnu.org/licenses/lgpl-2.1.html\
                \n.. _LGPLv3: http://www.gnu.org/licenses/lgpl.html\
                \n.. _GPLv2: http://www.gnu.org/licenses/gpl-2.0.html\
                \n.. _GPLv3: http://www.gnu.org/licenses/gpl.html\
                \n.. _ISC: http://opensource.org/licenses/ISC\
                \n.. _CC0: http://creativecommons.org/publicdomain/zero/1.0/\
                \n.. _MPLv2: http://mozilla.org/MPL/2.0/\
                \n\n .. footer:: Includes $$COUNT$$ licenses. \
                                 Generated at $$DATE$$."
ONE_LIB_TEMPLATE = "Library: `$$FULL-NAME$$ <$$PROJECT-HOME$$>`_  \
                   \n\n *Version:* **$$VERSION$$**  \
                   \n\n *License Type:* $$LICENSE-TYPE$$\
                   - `Original License Text <$$LICENSE$$>`__ - *Usage Type:* $$USAGE$$ \
                   \n\n *Copyright Notice:* $$COPYRIGHT-NOTICE$$  \
                   \n\n $$DESCRIPTION$$ \
                   \n\n\n"


def _invalid_version(expected, actual):
    """ 
    Check if current found version of 3rd party library is 
    acceptable from the licensing point of view.
    """
    if actual.startswith(expected):
        return False
    if expected.startswith('['):
        expected = expected[1:-1].split(',')
        return actual not in expected
    return True


def generate_artefact(root_folder_to_introspect, excludes=None, actual_libs=None, extra_licenses_check=None):
    """
    Compare accepted licenses with introspected licenses.
    """
    accepted_libs = parser.read_default()
    if actual_libs is None:
        actual_libs = parse_tree_structure(root_folder_to_introspect, excludes)
    if extra_licenses_check is not None:
        for entry in extra_licenses_check:
            actual_libs.update(parse_tree_structure(entry, excludes))
    dep_text = ""
    files2zip = []
    exceptions = dict()

    for lib_name in sorted(actual_libs, key=str.lower):
        lib_name = lib_name.lower()

        if lib_name not in accepted_libs:
            exceptions[lib_name] = actual_libs[lib_name]
            continue

        accepted_lib = accepted_libs[lib_name]
        actual_lib = actual_libs[lib_name]

        if _invalid_version(accepted_lib[parser.KEY_VERSION], str(actual_lib)):
            exceptions[lib_name] = actual_libs[lib_name]
            continue

        accepted_lib[parser.KEY_VERSION] = actual_lib
        lib_text = ONE_LIB_TEMPLATE
        for att in accepted_lib:
            lib_text = lib_text.replace("$$" + att.upper() + "$$", accepted_lib[att])
        if parser.KEY_LICENSE in accepted_lib:
            files2zip.append(os.path.join(CURRENT_FOLDER, LICENSES_FOLDER, accepted_lib[parser.KEY_LICENSE]))
        dep_text += lib_text

    if exceptions:
        print("Libraries: " + str(exceptions) + "\n are included in TVB package but their license were not validated!")

    if os.path.exists(RESULT_FILE_NAME + '.rst'):
        os.remove(RESULT_FILE_NAME + '.rst')

    dep_text = FULL_TEMPLATE.replace("$$DEPENDENCIES$$", dep_text)
    dep_text = dep_text.replace('$$COUNT$$', str(len(actual_libs)))
    dep_text = dep_text.replace('$$DATE$$', datetime.now().strftime('%Y/%m/%d'))
    dep_text = dep_text.replace("[unknown,", '[-,').replace(",unknown]", ',-]')
    dep_text = dep_text.replace("*Version:* **unknown**", "*Version:* **-**")

    print(" - Writing used dependencies as xml")
    # this is used for logging purposes
    # Gathering all these files from the build machines will show TVB's dependencies
    parser.write_used_on_this_platform(accepted_libs, actual_libs, path=PACKAGES_USED_XML)

    dep_html = docutils.core.publish_string(dep_text, writer_name='html4css1', settings=None,
                                            settings_overrides={
                                                'stylesheet_path': os.path.join(CURRENT_FOLDER, 'scheme.css')})
    with open(RESULT_FILE_NAME + '.html', 'wb') as f:
        f.write(dep_html)
    # Create ZIP with included dependencies.
    if os.path.exists(RESULT_FILE_NAME + '.zip'):
        os.remove(RESULT_FILE_NAME + '.zip')
    files2zip.append(RESULT_FILE_NAME + '.html')
    files2zip.append(os.path.join(CURRENT_FOLDER, 'back_brain.png'))
    files2zip = set(files2zip)

    with ZipFile(RESULT_FILE_NAME + '.zip', "w", ZIP_DEFLATED) as z_file:
        for file_name in files2zip:
            z_file.write(file_name, os.path.split(file_name)[1])

    os.remove(RESULT_FILE_NAME + '.html')
    print(" - THIRD_PARTY_LICENSES were generated")
    return RESULT_FILE_NAME + '.zip'


if __name__ == "__main__":
    # Run stand-alone.
    all_libs = parser.read_default()
    to_exclude = []
    for libname, libattr in all_libs.iteritems():
        if 'Desktop' in libattr["description"] or 'unused' in libattr['env']:
            to_exclude.append(libname)
        all_libs[libname] = libattr["version"]

    for lib in to_exclude:
        del all_libs[lib]

    print("Generating Dep-list with " + str(len(all_libs)) + " libraries.")
    RESULT_ZIP = generate_artefact(os.path.expanduser(os.path.join('~', 'Downloads', 'TVB_DISTRIBUTION')), [])
    os.rename(RESULT_ZIP, "_ALL_THIRD_PARTY.zip")
