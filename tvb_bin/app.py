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
.. moduleauthor:: Ciprian Tomoiaga <ciprian.tomoiaga@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

This script will start TVB using a given profile, stop all TVB processes or clean a TVB installation.
Usage: Run python app.py --help
"""
import argparse
import os
import subprocess
import signal
import shutil
import sys
import socket
import logging

try:
    import psycopg2
except ImportError:
    print "Could not find compatible psycopg2/postgresql bindings. Postgresql support not available."
        
from tvb.basic.profile import TvbProfile

PARAM_START = "start"
PARAM_STOP = "stop"
PARAM_CLEAN = "clean"
PARAM_COMMAND = "command"
PARAM_LIBRARY = "library"
PARAM_MODEL_VALIDATION = "validate"
SUBPARAM_WEB = "web"
SUBPARAM_DESKTOP = "desktop"
SUBPARAM_RESET = "reset"
SUBPARAM_HEADLESS = "headless"
SUBPARAM_PROFILE = TvbProfile.SUBPARAM_PROFILE

RUN_WEB_PROFILES = [TvbProfile.DEPLOYMENT_PROFILE, TvbProfile.DEVELOPMENT_PROFILE,
                    TvbProfile.TEST_POSTGRES_PROFILE, TvbProfile.TEST_SQLITE_PROFILE]


def parse_commandline():
    def add_profile_arg(com):
        help = 'Use a specific profile. Allowed values are: '
        help += ', '.join(TvbProfile.ALL) + '. '
        help += 'These profiles will launch the web interface : '
        help += ', '.join(RUN_WEB_PROFILES)
        com.add_argument('profile', metavar='profile', nargs='?', help=help,
                         choices=TvbProfile.ALL, default=TvbProfile.DEPLOYMENT_PROFILE)

    parser = argparse.ArgumentParser(description="Control TVB instances.", prog='tvb_start.sh')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')

    start = subparsers.add_parser('start', help='launch a TVB interface')
    add_profile_arg(start)

    start.add_argument('-reset', action='store_true', help='reset database')
    start.add_argument('-headless', action='store_true', help='launch python instead of IDLE')

    stop = subparsers.add_parser('stop', help='stop all TVB processes')
    stop.set_defaults(profile=TvbProfile.DEPLOYMENT_PROFILE)

    clean = subparsers.add_parser('clean', help='stop all TVB processes and delete all TVB data')
    add_profile_arg(clean)

    return parser.parse_args()


ARGS = parse_commandline()

### This is a TVB Framework initialization call. Make sure a good default profile gets set.
TvbProfile.set_profile([TvbProfile.SUBPARAM_PROFILE, ARGS.profile])

### Initialize TVBSettings only after a profile was set
from tvb.basic.config.settings import TVBSettings


SETTER = TVBSettings()
if TvbProfile.is_mac_deployment() or TvbProfile.is_windows_deployment():
    ## Import libraries to be found by the introspection.
    import imghdr
    import sndhdr  

# will happen when a mac build is done
if 'py2app' in sys.argv:
    import tvb.interfaces.web.run
    
PYTHON_EXE_PATH = SETTER.get_python_path()

CONSOLE_TVB = 'tvb_bin.run_IDLE'
VALIDATE_TVB = 'tvb.core.model_validations'
CONSOLE_PROFILE_SET = ('from tvb.basic.profile import TvbProfile as tvb_profile; '
                       'tvb_profile.set_profile(["-profile", "CONSOLE_PROFILE"], try_reload=False);')
LIBRARY_PROFILE_SET = ('from tvb.basic.profile import TvbProfile as tvb_profile; '
                       'tvb_profile.set_profile(["-profile", "LIBRARY_PROFILE"], try_reload=False);')

WEB_TVB = 'tvb.interfaces.web.run'
DESKTOP_TVB = 'tvb.interfaces.desktop.run'

TVB_PID_FILE = os.path.join(TVBSettings.TVB_STORAGE, "pid.tvb")


def find_free_port(tested_port):
    """
    Given an input port that can be either a string or a integer, find the closest port to it that is free.
    """
    tested_port = int(tested_port)
    test_socket = socket.socket()
    port_in_use = True
    try_no = 0
    while port_in_use and try_no < 1000:
        try:
            test_socket.connect((TVBSettings.LOCALHOST, tested_port))
            test_socket.close()
            sys.stdout.write('Port ' + str(tested_port) + ' seems to be in use.\n')
            tested_port += 1
        except socket.error:
            port_in_use = False
        except Exception, excep1:
            sys.stdout.write(str(excep1) + '\n')
        try_no += 1
    if not port_in_use:
        sys.stdout.write('Found the first free port: ' + str(tested_port) + '.\n')
    else:
        sys.exit("Could not find a free port for back-end to start!")
    return tested_port
      
      
def execute_clean():
    """
    Remove TVB folder, TVB File DB, and log files.
    """
    try:
        if os.path.isdir(TVBSettings.TVB_STORAGE):
            shutil.rmtree(TVBSettings.TVB_STORAGE, ignore_errors=True)
        elif os.path.exists(TVBSettings.TVB_STORAGE):
            os.remove(TVBSettings.TVB_STORAGE)
    except Exception, excep1:
        sys.stdout.write("Could not remove TVB folder!")
        sys.stdout.write(str(excep1)) 
 
 
def execute_stop(): 
    """
    Stop registered TVB processes in .tvb file
    """  
    logging.shutdown()
    if os.path.exists(TVB_PID_FILE):  
        pid_file = open(TVB_PID_FILE, 'r')
        has_processes = False
        for pid in pid_file.read().split('\n'):
            if len(pid.strip()):
                try:
                    if sys.platform == 'win32':
                        import ctypes
                        handle = ctypes.windll.kernel32.OpenProcess(1, False, int(pid))
                        ctypes.windll.kernel32.TerminateProcess(handle, -1)
                        ctypes.windll.kernel32.CloseHandle(handle)
                    else:
                        os.kill(int(pid), signal.SIGKILL) 
                except Exception:
                    has_processes = True
        if has_processes:
            sys.stdout.write("Some old PIDs were still registered. They have been stopped.")                    
        pid_file.close()
    pid_file = open(TVB_PID_FILE, "w")
    pid_file.close()


def execute_start_web(profile, reset):
    """
    Start the web server on a free port.
    :return: reference towards CherryPy process
    """
    pid_file_reference = open(TVB_PID_FILE, 'a')
    free_ports = {}
    cherrypy_port = find_free_port(TVBSettings.WEB_SERVER_PORT)
    if not os.path.isfile(TVBSettings.TVB_CONFIG_FILE) or TVBSettings.WEB_SERVER_PORT != cherrypy_port:
        free_ports['WEB_SERVER_PORT'] = cherrypy_port
        TVBSettings.update_config_file(free_ports)

    web_args_list = [PYTHON_EXE_PATH, '-m', WEB_TVB, 'tvb.config']
    web_args_list.extend([SUBPARAM_PROFILE, profile])

    if reset:
        web_args_list.append(SUBPARAM_RESET)
        pid_file_reference.close()
        execute_clean()
        if not os.path.exists(TVBSettings.TVB_STORAGE):
            os.mkdir(TVBSettings.TVB_STORAGE)
        pid_file_reference = open(TVB_PID_FILE, 'a')

    cherrypy_process = subprocess.Popen(web_args_list, shell=False)

    pid_file_reference.write(str(cherrypy_process.pid) + "\n")
    pid_file_reference.close()
    return cherrypy_process


def execute_start_desktop(profile, reset):
    """
    Fire TVB with Desktop interface.
    :return: reference towards tvb new started process
    """
    pid_file_reference = open(TVB_PID_FILE, 'a')

    desktop_args_list = [PYTHON_EXE_PATH, '-m', DESKTOP_TVB, 'tvb.config']

    desktop_args_list.extend([SUBPARAM_PROFILE, profile])

    if reset:
        desktop_args_list.append(SUBPARAM_RESET)
        pid_file_reference.close()
        execute_clean()
        if not os.path.exists(TVBSettings.TVB_STORAGE):
            os.mkdir(TVBSettings.TVB_STORAGE)
        pid_file_reference = open(TVB_PID_FILE, 'a')
        tvb_process = subprocess.Popen(desktop_args_list, shell=False)
    else:
        tvb_process = subprocess.Popen(desktop_args_list, shell=False)

    pid_file_reference.write(str(tvb_process.pid) + "\n")
    pid_file_reference.close()
    return tvb_process


def execute_start_console(console_profile_set, headless):
    PID_FILE = open(TVB_PID_FILE, 'a')
    if headless:
        # Launch a python interactive shell.
        # We use a new process to make sure that the initialization of TVB is straightforward and
        # not influenced by this launcher.
        TVB = subprocess.Popen([PYTHON_EXE_PATH, '-i', '-c', console_profile_set])
    else:
        TVB = subprocess.Popen([PYTHON_EXE_PATH, '-m', CONSOLE_TVB, '-c', console_profile_set])

    PID_FILE.write(str(TVB.pid) + "\n")
    PID_FILE.close()

    if headless:
        # The child inherits the stdin stdout descriptors of the launcher so we keep the launcher alive
        # by calling wait. It would be good if this could be avoided.
        TVB.wait()


# def execute_model_validation():
#     from tvb.core.model_validations import main
#     sys.argv.remove(PARAM_MODEL_VALIDATION)
#     if TvbProfile.CONSOLE_PROFILE in sys.argv:
#         sys.argv.remove(TvbProfile.CONSOLE_PROFILE)
#     if TvbProfile.SUBPARAM_PROFILE in sys.argv:
#         sys.argv.remove(TvbProfile.SUBPARAM_PROFILE)
#     if len(sys.argv) < 2:
#         raise Exception("Validation settings file needs to be provided!")
#     main(sys.argv[1])


def wait_for_tvb_process(tvb_process):
    """
    On MAC devices do not let this process die, to keep TVB icon in the dock bar.
    :param tvb_process: TVB sub process, to wait until it finishes.
    """
    if tvb_process is not None and TvbProfile.is_mac_deployment():

        import AppKit
        import Foundation
        from PyObjCTools import AppHelper

        class ApplicationDelegate(Foundation.NSObject):
            """ Cocoa specific Delegate class """
            check_tvb = None

            def applicationDidFinishLaunching_(self, notification):
                """ Register a checking to follow TVB subprocess """
                start_time = Foundation.NSDate.date()
                self.check_tvb = AppKit.NSTimer.alloc().initWithFireDate_interval_target_selector_userInfo_repeats_(
                                        start_time, 2.0, self, 'tick:', None, True)
                AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(self.check_tvb, AppKit.NSDefaultRunLoopMode)
                self.check_tvb.fire()

            def tick_(self, notification):
                """ Keep alive only as long as TVB subprocess is still running """
                if tvb_process.poll():
                    AppHelper.stopEventLoop()

            def applicationWillTerminate_(self, notification):
                """ Make sure no TVB subprocess is left behind running """
                execute_stop()

        app = AppKit.NSApplication.sharedApplication()
        delegate = ApplicationDelegate.alloc().init()
        app.setDelegate_(delegate)
        AppHelper.runEventLoop()


if __name__ == "__main__":
    if not os.path.exists(TVBSettings.TVB_STORAGE):
        try:
            os.makedirs(TVBSettings.TVB_STORAGE)
        except Exception:
            sys.exit("You do not have enough rights to use TVB storage folder:" + str(TVBSettings.TVB_STORAGE))

    if ARGS.subcommand == 'start':
        if ARGS.profile == TvbProfile.CONSOLE_PROFILE:
            # warn: not executing stop overwrites prev pid!
            execute_start_console(CONSOLE_PROFILE_SET, ARGS.headless)
        elif ARGS.profile == TvbProfile.LIBRARY_PROFILE:
            execute_start_console(LIBRARY_PROFILE_SET, ARGS.headless)
        elif ARGS.profile in RUN_WEB_PROFILES :
            execute_stop()
            TVB_PROCESS = execute_start_web(ARGS.profile, ARGS.reset)
            wait_for_tvb_process(TVB_PROCESS)
        elif ARGS.profile == TvbProfile.DESKTOP_PROFILE:
            execute_stop()
            TVB_PROCESS = execute_start_desktop(ARGS.profile, ARGS.reset)
            wait_for_tvb_process(TVB_PROCESS)
    elif ARGS.subcommand == 'stop':
        # Kill all Python processes which have their PID registered in .tvb file
        execute_stop()
    elif ARGS.subcommand == 'clean':
        execute_stop()
        # Remove TVB folder, TVB File DB, and log file.
        # ignore clean for console profiles
        if not ARGS.profile.lower().startswith('test'): # todo: check this
            execute_clean()
            if os.path.exists(TVBSettings.TVB_CONFIG_FILE):
                os.remove(TVBSettings.TVB_CONFIG_FILE)