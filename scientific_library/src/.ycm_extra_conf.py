# this script is for developers working with the YouCompleteMe completion
# and diagnostics.

import os
import ycm_core
import sys
sys.path.append(os.path.dirname(__file__))
import compile_info

flags = compile_info.cpp_flags[:]
flags.append('-I' + os.getcwd())

flags.append('-I/usr/include/python2.7')

for inc in compile_info.extern_includes:
  flags.append('-Iextern/' + inc)

def FlagsForFile( filename, **kwargs ):
  return { 'flags': flags, 'do_cache': True }
