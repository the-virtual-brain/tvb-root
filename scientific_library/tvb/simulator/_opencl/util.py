"""
Utilities for working with OpenCL.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import pyopencl


def create_cpu_context():
    for platform in pyopencl.get_platforms():
        for device in platform.get_devices(pyopencl.device_type.CPU):
            return pyopencl.Context([device])

def context_and_queue(context):
    return context, pyopencl.CommandQueue(context)