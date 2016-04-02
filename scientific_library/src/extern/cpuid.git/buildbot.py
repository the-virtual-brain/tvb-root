#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import json
import subprocess

project_name = 'cpuid'


def run_command(args):
    print("Running: {}".format(args))
    sys.stdout.flush()
    subprocess.check_call(args)


def get_tool_options(properties):
    options = []
    if 'tool_options' in properties:
        # Make sure that the values are correctly comma separated
        for key, value in properties['tool_options'].items():
            if value is None:
                options += ['--{0}'.format(key)]
            else:
                options += ['--{0}={1}'.format(key, value)]

    return options


def configure(properties):
    command = [sys.executable, 'waf']

    if properties.get('build_distclean'):
        command += ['distclean']

    command += ['configure', '--git-protocol=git@']

    if 'waf_bundle_path' in properties:
        command += ['--bundle-path=' + properties['waf_bundle_path']]

    if 'dependency_project' in properties:
        command += ['--{0}-use-checkout={1}'.format(
            properties['dependency_project'],
            properties['dependency_checkout'])]

    command += ["--cxx_mkspec={}".format(properties['cxx_mkspec'])]
    command += get_tool_options(properties)

    run_command(command)


def build(properties):
    command = [sys.executable, 'waf', 'build', '-v']
    run_command(command)


def run_tests(properties):
    command = [sys.executable, 'waf', '-v', '--run_tests']
    run_cmd = '%s'

    if properties.get('valgrind_run'):
        run_cmd = 'valgrind --error-exitcode=1 %s --no_fail'

    if 'cpu_capabilities' in properties:
        for cpu_capability in properties['cpu_capabilities'].items():
            run_cmd += ' --has_{0}={1}'.format(*cpu_capability)

    if run_cmd:
        command += ["--run_cmd={}".format(run_cmd)]

    command += get_tool_options(properties)

    run_command(command)


def install(properties):
    command = [sys.executable, 'waf', '-v', 'install']

    if 'install_path' in properties:
        command += ['--install_path={0}'.format(properties['install_path'])]
    if properties.get('install_relative'):
        command += ['--install_relative']

    run_command(command)


def coverage_settings(options):
    options['required_line_coverage'] = 100.0
    options['run_cmd'] = '%s --no_fail'


def main():
    argv = sys.argv

    if len(argv) != 3:
        print("Usage: {} <command> <properties>".format(argv[0]))
        sys.exit(0)

    cmd = argv[1]
    properties = json.loads(argv[2])

    if cmd == 'configure':
        configure(properties)
    elif cmd == 'build':
        build(properties)
    elif cmd == 'run_tests':
        run_tests(properties)
    elif cmd == 'install':
        install(properties)
    else:
        print("Unknown command: {}".format(cmd))


if __name__ == '__main__':
    main()
