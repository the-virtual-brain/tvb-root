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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""

import fileinput
import glob
import json
import logging
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import time
import dmgbuild
import magic
import six
from tvb.basic.profile import TvbProfile

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging.getLogger()

# ===============================================================================
# General settings applicable to all apps
# ===============================================================================
TVB_ROOT = os.path.dirname(os.path.dirname(__file__))
# The short version string
VERSION = TvbProfile.current.version.BASE_VERSION
# Name of the app
APP_NAME = "tvb-{}".format(VERSION)
# should match an Apple Developer defined identifier
IDENTIFIER = "ro.codemart.tvb"
# KEY for the ENV variable where we expect the signing identity to be defined
KEY_SIGN_IDENTITY = "SIGN_APP_IDENTITY"
# The author of this package
AUTHOR = "TVB Team"
# Full path to the anaconda environment folder to package
# Make sure it is the full path (and not a relative one, also to the homedir with ~) so this can be
# correctly replaced later. Conda usßes hardcoded paths, which we convert to `/Applications/<APP_NAME>`
CONDA_ENV_PATH = "/Applications/anaconda3/envs/mac-distribution"
# Folders to include from Anaconda environment, if ommitted everything will be copied
# CONDA_FOLDERS = ["lib", "bin", "share", "qsci", "ssl", "translations"]
# Paths of files and folders to remove from the copied anaconda environment,
# relative to the environment's root.
# For instance, this could be the qt4 apps (an app inside an app is useless)
CONDA_EXCLUDE_FILES = [
    'bin/*.app',
    'bin/*.prl',
    'bin/qmake',
    'bin/2to3*',
    'bin/autopoint',
    'conda-meta',
    'include',
    'lib/*.prl',
    'lib/pkg-config',
    'org.freedesktop.dbus-session.plist'
]

CONDA_EXCLUDE_FILES += map(lambda x: f'translations/{x}', [
    'assistant*', 'designer*', 'linguist*', 'qt_*', 'qtbase*', 'qtconnectivity*', 'qtdeclarative*',
    'qtlocation*', 'qtmultimedia*', 'qtquickcontrols*', 'qtscript*', 'qtserialport*',
    'qtwebsockets*', 'qtxmlpatterns*'
])

# Path to the icon of the app
ICON_PATH = os.path.join(TVB_ROOT, "tvb_build", "icon.icns")
# Absolute path towards TVB license file, to be included in the .app
LICENSE_PATH = os.path.join(TVB_ROOT, "LICENSE")
# The entry script of the application in the environment's bin folder
ENTRY_SCRIPT = "-m tvb_bin.app"
# Folder to place created APP and DMG in.
OUTPUT_FOLDER = os.path.join(TVB_ROOT, "dist")

# Information about file types that the app can handle
APP_SUPPORTED_FILES = {
    "CFBundleDocumentTypes": [
        {
            'CFBundleTypeName': "TVB Distribution",
            'CFBundleTypeRole': "Editor",
            'LSHandlerRank': "Owner",
            'CFBundleTypeIconFile': os.path.basename(ICON_PATH),
            'LSItemContentTypes': ["nl.cogsci.osdoc.osexp"],
            'NSExportableTypes': ["nl.cogsci.osdoc.osexp"]
        }
    ],
    "UTExportedTypeDeclarations": [
        {
            'UTTypeConformsTo': ['org.gnu.gnu-zip-archive'],
            'UTTypeDescription': "TVB Distribution",
            'UTTypeIdentifier': "nl.cogsci.osdoc.osexp",
            'UTTypeTagSpecification': {
                'public.filename-extension': 'osexp',
                'public.mime-type': 'application/gzip'
            }
        }
    ]
}
# Placed here to not let linter go crazy. Will be overwritten by main program
RESOURCE_DIR = ""

# Optional config entries
try:
    ICON_PATH = os.path.expanduser(ICON_PATH)
    ICON_FILE = os.path.basename(ICON_PATH)
except NameError:
    ICON_FILE = None

# Account for relative paths to home folder
CONDA_ENV_PATH = os.path.expanduser(CONDA_ENV_PATH)
OUTPUT_FOLDER = os.path.expanduser(OUTPUT_FOLDER)

# Physical location of the app
APP_FILE = os.path.join(OUTPUT_FOLDER, APP_NAME + u'.app')
# Set up the general structure of the app
MACOS_DIR = os.path.join(APP_FILE, u'Contents/MacOS')
# Create APP_NAME/Contents/Resources
RESOURCE_DIR = os.path.join(APP_FILE, u'Contents/Resources')
# Execution script in app
APP_SCRIPT = os.path.join(MACOS_DIR, APP_NAME)

# ===== Settings specific to dmgbuild =====
# DMG format
DMG_FORMAT = 'UDZO'
# Locations of shortcuts in DMG window
DMG_ICON_LOCATIONS = {
    APP_NAME + '.app': (5, 452),
    'Applications': (200, 450)
}
# Size of DMG window when mounted
DMG_WINDOW_RECT = ((300, 200), (358, 570))
# Size of icons in DMG
DMG_ICON_SIZE = 80


def extra():
    _fix_paths()


def _find_and_replace(path, search, replace, exclusions=None):
    if not type(exclusions) in ['list', 'tuple']:
        exclusions = []

    exclusion_valid = False
    for root, _, files in os.walk(path):
        for entry in exclusions:
            if entry in root:
                exclusion_valid = True
                break
        if exclusion_valid:
            continue
        # Do not traverse into python site-packages folders
        logger.debug('Scanning {}'.format(root))
        candidates = []
        for f in files:
            full_path = os.path.join(root, f)

            try:
                filetype = magic.from_file(full_path)
            except UnicodeDecodeError:
                logger.warning(f'Unable to infer type of {full_path}')
                continue

            if filetype == 'empty':
                continue

            if re.search(r'\stext(?:\s+executable)?', filetype):
                candidates.append(full_path)

        if len(candidates) == 0:
            continue

        logger.debug(list(map(os.path.basename, candidates)))

        with fileinput.input(candidates, inplace=True) as stream:
            finished = False
            while not finished:
                try:
                    for line in stream:
                        print(line.replace(search, replace), end='')
                    finished = True
                except Exception as e:
                    logger.warning(f'Unable to process: {stream.filename()} - {e}')
                    stream.nextfile()


def _replace_conda_abs_paths():
    app_path = os.path.join(os.path.sep, 'Applications', APP_NAME + '.app', 'Contents', 'Resources')
    print('Replacing occurences of {} with {}'.format(CONDA_ENV_PATH, app_path))
    _find_and_replace(
        RESOURCE_DIR,
        CONDA_ENV_PATH,
        app_path,
        exclusions=['site-packages', 'doc']
    )


def create_app():
    """ Create an app bundle """
    print("Output Dir {}".format(OUTPUT_FOLDER))

    if os.path.exists(APP_FILE):
        shutil.rmtree(APP_FILE)

    print("\n++++++++++++++++++++++++ Creating APP +++++++++++++++++++++++++++")
    start_t = time.time()

    _create_app_structure()
    _copy_anaconda_env()
    if ICON_FILE:
        _copy_icon_and_license()
    _create_plist()

    # Do some package specific stuff, which is defined in the extra() function
    # in settings.py (and was imported at the top of this module)
    if "extra" in globals() and callable(extra):
        print("Performing application specific actions.")
        extra()

    _replace_conda_abs_paths()

    print("============ APP CREATION FINISHED in {} seconds ====================".format(int(time.time() - start_t)))

    _sign_app()


def _create_app_structure():
    """ Create folder structure comprising a Mac app """
    print("Creating app structure")
    try:
        os.makedirs(MACOS_DIR)
    except OSError as e:
        print('Could not create app structure: {}'.format(e))
        sys.exit(1)

    print("Creating app entry script")
    with open(APP_SCRIPT, 'w') as fp:
        # Write the contents
        try:
            fp.write("#!/usr/bin/env bash\n"
                     "script_dir=$(dirname \"$(dirname \"$0\")\")\n"
                     "$script_dir/Resources/bin/python "
                     "{} $@".format(ENTRY_SCRIPT))
        except IOError as e:
            logger.exception("Could not create Contents/OpenSesame script")
            sys.exit(1)

    # Set execution flags
    current_permissions = stat.S_IMODE(os.lstat(APP_SCRIPT).st_mode)
    os.chmod(APP_SCRIPT, current_permissions |
             stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_anaconda_env():
    """ Copy anaconda environment """
    print("Copying Anaconda environment (this may take a while)")
    try:
        if "CONDA_FOLDERS" in globals():
            # IF conda folders is specified, copy only those folders.
            for item in CONDA_FOLDERS:
                shutil.copytree(
                    os.path.join(CONDA_ENV_PATH, item),
                    os.path.join(RESOURCE_DIR, item),
                    symlinks=True)
        else:
            # Copy everything
            shutil.copytree(CONDA_ENV_PATH, RESOURCE_DIR, True)
    except OSError as e:
        logger.error("Error copying Anaconda environment: {}".format(e))
        sys.exit(1)

    # Delete unnecessary files (such as all Qt apps included with conda)
    if "CONDA_EXCLUDE_FILES" in globals():
        for excl_entry in CONDA_EXCLUDE_FILES:
            full_path = os.path.join(RESOURCE_DIR, excl_entry)
            # Expand wild cards and such
            filelist = glob.glob(full_path)
            for item in filelist:
                try:
                    if os.path.isdir(item):
                        logger.debug("Removing folder: {}".format(item))
                        shutil.rmtree(item)
                    elif os.path.isfile(item):
                        logger.debug("Removing file: {}".format(item))
                        os.remove(item)
                    else:
                        logger.warning("File not found: {}".format(item))
                except (IOError, OSError) as e:
                    logger.error("WARNING: could not delete {}".format(item))


def _copy_icon_and_license():
    """ Copy icon to Resources folder """
    global ICON_PATH
    print("Copying icon file")
    try:
        shutil.copy(ICON_PATH, os.path.join(RESOURCE_DIR, ICON_FILE))
    except OSError as e:
        logger("Error copying icon file from: {}".format(ICON_PATH))

    global LICENSE_PATH
    print("Copying license file")
    try:
        unnecessary_file = os.path.join(RESOURCE_DIR, "LICENSE.txt")
        if os.path.exists(unnecessary_file):
            os.remove(unnecessary_file)
        shutil.copy(LICENSE_PATH, RESOURCE_DIR)
    except OSError as e:
        logger("Error copying license file from: {}".format(LICENSE_PATH))


def _create_plist():
    print("Creating Info.plist")

    global ICON_FILE
    global VERSION

    if 'LONG_VERSION' in globals():
        global LONG_VERSION
    else:
        LONG_VERSION = VERSION

    info_plist_data = {
        'CFBundleDevelopmentRegion': 'en',
        'CFBundleExecutable': APP_NAME,
        'CFBundleIdentifier': IDENTIFIER,
        'CFBundleInfoDictionaryVersion': '6.0',
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': APP_NAME,
        'CFBundlePackageType': 'APPL',
        'CFBundleVersion': LONG_VERSION,
        'CFBundleShortVersionString': VERSION,
        'CFBundleSignature': '????',  # ok not to be setup
        'LSMinimumSystemVersion': '10.7.0',
        'LSUIElement': False,
        'NSAppTransportSecurity': {'NSAllowsArbitraryLoads': True},
        'NSHumanReadableCopyright': "(c) 2012-2023, Baycrest Centre for Geriatric Care ('Baycrest') and others",
        'NSMainNibFile': 'MainMenu',
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': True,
    }

    if ICON_FILE:
        info_plist_data['CFBundleIconFile'] = ICON_FILE

    if 'APP_SUPPORTED_FILES' in globals():
        global APP_SUPPORTED_FILES
        info_plist_data['CFBundleDocumentTypes'] = APP_SUPPORTED_FILES['CFBundleDocumentTypes']

        if 'UTExportedTypeDeclarations' in APP_SUPPORTED_FILES:
            info_plist_data['UTExportedTypeDeclarations'] = \
                APP_SUPPORTED_FILES['UTExportedTypeDeclarations']

    with open(os.path.join(APP_FILE, 'Contents', 'Info.plist'), 'wb') as fp:
        plistlib.dump(info_plist_data, fp)


def _sign_app(app_path=APP_FILE):
    """
    Sign a .APP file, with an Apple Developer Identity previously installed on the current machine.
    The identity needs to show when executing command "security find-identity"
    """
    if KEY_SIGN_IDENTITY not in os.environ:
        print(f"!! We can not sign the resulting .app because the {KEY_SIGN_IDENTITY} variable is not in ENV defined!!")
    dev_identity = os.environ.get(KEY_SIGN_IDENTITY)
    print(f"Preparing to sign: {app_path} with {dev_identity}")
    # Create app.entitlements file with the application allowed security allowed points
    ent_file = "app.entitlements"
    if os.path.exists(ent_file):
        os.remove(ent_file)
    with open(ent_file, 'w') as fp:
        fp.write("""
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
        <dict>
            <key>com.apple.security.app-sandbox</key>
            <true/>
            <key>com.apple.security.network.client</key>
            <true/>
            <key>com.apple.security.network.server</key>
            <true/>
        </dict>
    </plist>
    """)

    # Uncomment the following command if needed for debug purposes
    # subprocess.Popen(["security", "find-identity"], shell=False).communicate()
    subprocess.Popen(["codesign", "-s", dev_identity, "-f", "--timestamp", "-o", "runtime", "--entitlements", "app.entitlements", app_path], shell=False).communicate()
    subprocess.Popen(["spctl", "-a", "-t", "exec", "-vv", app_path], shell=False).communicate()

    if os.path.exists(ent_file):
        os.remove(ent_file)


def create_dmg():
    """ Create a dmg of the app """

    # Check if app to exists
    if not os.path.isdir(APP_FILE):
        logger.error("Could not find app file at {}".format(APP_FILE))
        sys.exit(1)

    dmg_file = os.path.join(OUTPUT_FOLDER, APP_NAME + u'.dmg')

    if os.path.exists(dmg_file):
        os.remove(dmg_file)

    print("\n+++++++++++++++++++++ Creating DMG from app +++++++++++++++++++++++")

    # Get file size of APP
    app_size = subprocess.check_output(
        ['du', '-sh', APP_FILE]).split()[0].decode('utf-8')
    # returns tuple with format ('3.0', 'G')
    (size, unit) = re.findall('(\d+\.?\d?)(\w)', app_size)[0]

    # Add a bit of extra to the disk image size
    app_size = str(float(size) * 1.25) + unit

    print("Creating disk image of {}".format(app_size))

    # Create a dmgbuild config file in same folder as
    dmgbuild_config_file = os.path.join(os.getcwd(),
                                        'dmgbuild_settings.py')

    dmg_config = {
        'filename': dmg_file,
        'volume_name': APP_NAME,
        'size': app_size,
        'files': [APP_FILE],
        'symlinks': {'Applications': '/Applications'},
    }

    if ICON_FILE:
        dmg_config['badge_icon'] = ICON_PATH
    dmg_config['format'] = DMG_FORMAT
    dmg_config['icon_size'] = DMG_ICON_SIZE
    dmg_config['icon_locations'] = DMG_ICON_LOCATIONS
    dmg_config['window_rect'] = DMG_WINDOW_RECT

    _write_vars_to_file(dmgbuild_config_file, dmg_config)
    print("Copying files to DMG and compressing it. Please wait.")
    dmgbuild.build_dmg(dmg_file, APP_NAME, settings_file=dmgbuild_config_file)

    # Clean up!
    os.remove(dmgbuild_config_file)


def _write_vars_to_file(file_path, var_dict):
    with open(file_path, 'w') as fp:
        fp.write("# -*- coding: utf-8 -*-\n")
        fp.write("from __future__ import unicode_literals\n\n")

        for var, value in var_dict.items():
            if isinstance(value, six.string_types):
                fp.write('{} = "{}"\n'.format(var, value))
            else:
                fp.write('{} = {}\n'.format(var, value))


def _fix_paths():
    kernel_json = os.path.join(
        RESOURCE_DIR, 'share', 'jupyter', 'kernels', 'python3', 'kernel.json')
    if os.path.exists(kernel_json):
        print('Fixing kernel.json')
        with open(kernel_json, 'r') as fp:
            kernelCfg = json.load(fp)
            kernelCfg['argv'][0] = 'python'
        with open(kernel_json, 'w+') as fp:
            json.dump(kernelCfg, fp)


if __name__ == "__main__":
    create_app()
    create_dmg()
