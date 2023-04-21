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
This is a diagnostic tool that is used to find file leaks
Usage :

import from this module the  monkey_patch_file_to_trace function and call it very early in the program
Once the trace is collected run  python diagnose_file_trace.py [filter_path_prefix]> analysed_trace then study the analysed_trace file
"""

import __builtin__
import traceback
from datetime import datetime


def timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")


def monkey_patch_file_to_trace(trace_file='file_open_trace'):
    _file_open_trace = open(trace_file, 'w', buffering=0)

    # openfiles = set()
    global oldfile
    global oldopen

    oldfile = __builtin__.file

    class newfile(oldfile):
        def __init__(self, *args):
            self.x = args[0]
            _file_open_trace.write("$$ %s OPENING %s\n" % (timestamp(), str(self.x)))
            # traceback module will open python files to get the trace. We must exclude them from tracing otherwise stackoverflow
            if not self.x.endswith('py'):
                _file_open_trace.write(''.join(traceback.format_stack()))
            oldfile.__init__(self, *args)
            # openfiles.add(self)

        def close(self):
            _file_open_trace.write("$$ %s CLOSING %s\n" % (timestamp(), str(self.x)))
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
            if line.startswith("$$"):
                idx1 = line.find(' ')
                idx2 = line.find(' ', idx1 + 1)
                idx3 = line.find(' ', idx2 + 1)

                time = line[idx1 + 1:idx2]
                op = line[idx2 + 1:idx3]
                pth = line[idx3 + 1:-1]

                result.append((pth, op, time, []))
            else:
                result[-1][3].append(line)

    return result


def analyse_trace(filter_path_prefix=None):
    trace = parse_trace()

    if filter_path_prefix is not None:
        trace = [a for a in trace if a[0].startswith(filter_path_prefix)]

    result = {}

    for pth, op, time, stack in trace:
        if pth not in result:
            result[pth] = 0
        if op == "OPENING":
            result[pth] += 1
        else:
            result[pth] -= 1

    print()
    print('  List of files with unbalanced open close calls ')
    print('=================================================')
    print()

    for guilty_pth in result:
        if result[guilty_pth] != 0:
            print('%s %d' % (guilty_pth, result[guilty_pth]))
    print()
    print(' TRACES ')
    print('========')
    print()

    for guilty_pth in result:
        if result[guilty_pth] != 0:
            print()
            print('=' * 80)
            print('file: %s ' % guilty_pth)
            print('open-close: %d' % result[guilty_pth])
            print('printing access log')
            print()

            count = 0
            for pth, op, time, stack in trace:
                if pth == guilty_pth:
                    if op == "OPENING":
                        count += 1
                    else:
                        count -= 1

                    print('%s  %s %d -----------' % (time, op, count))
                    for st in stack:
                        print st,


if __name__ == '__main__':
    import sys

    if len(sys.argv) == 2:
        analyse_trace(sys.argv[1])
    else:
        analyse_trace()
