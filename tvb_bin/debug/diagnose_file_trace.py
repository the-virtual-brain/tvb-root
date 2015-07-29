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
This is a diagnostic tool that is used to find file leaks
Usage :

import from this module the  monkey_patch_file_to_trace function and call it very early in the program
Once the trace is collected run  python diagnose_file_trace.py [filter_path_prefix]> analysed_trace then study the analysed_trace file
"""

import __builtin__
import traceback

def monkey_patch_file_to_trace(trace_file='file_open_trace'):

    _file_open_trace = open(trace_file, 'w', buffering=0)

    # openfiles = set()
    global oldfile
    global oldopen

    oldfile = __builtin__.file

    class newfile(oldfile):
        def __init__(self, *args):
            self.x = args[0]
            _file_open_trace.write("OPENING %s\n" % str(self.x))
            # traceback module will open python files to get the trace. We must exclude them from tracing otherwise stackoverflow
            if not self.x.endswith('py'):
                _file_open_trace.write(''.join(traceback.format_stack()))
            oldfile.__init__(self, *args)
            # openfiles.add(self)

        def close(self):
            _file_open_trace.write("CLOSING %s\n" % str(self.x))
            if not self.x.endswith('py'):
                _file_open_trace.write(''.join(traceback.format_stack()))
            oldfile.close(self)
            # openfiles.remove(self)

    oldopen = __builtin__.open

    def newopen(*args):
        return newfile(*args)

    __builtin__.file = newfile
    __builtin__.open = newopen


def restore_monkey_patch():
    __builtin__.file = oldfile
    __builtin__.open = oldopen


def parse_trace():
    result = []

    with open('file_open_trace') as f:
        for line in f:
            if line.startswith("OPENING") or line.startswith("CLOSING"):
                idx = line.find(' ')
                op = line[:idx]
                pth = line[idx+1:-1]
                result.append((pth, op, []))
            else:
                result[-1][2].append(line)

    return result


def analyse_trace(filter_path_prefix=None):
    trace = parse_trace()

    if filter_path_prefix is not None:
        trace = [a for a in trace if a[0].startswith(filter_path_prefix)]

    result = {}

    for pth, op, stack in trace:
        if pth not in result:
            result[pth] = 0
        if op == "OPENING":
            result[pth] += 1
        else:
            result[pth] -= 1

    for guilty_pth in result:
        if result[guilty_pth] != 0:
            print
            print '===================='
            print 'this file is guilty %s %d' % (guilty_pth, result[guilty_pth])
            print 'printing access log'
            print

            count = 0
            for pth, op, stack in trace:
                if pth == guilty_pth:
                    if op == "OPENING":
                        count += 1
                    else:
                        count -= 1

                    print '---------------- %s %d' %(op, count)
                    for st in stack:
                        print st,


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 2:
        analyse_trace(sys.argv[1])
    else:
        analyse_trace()
