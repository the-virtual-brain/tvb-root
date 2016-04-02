cpuid
-----

cpuid is a C++ library for CPU dispatching. Currently the
project can detect the following CPU capabilities:

- Instruction sets detected on x86: FPU, MMX, SSE, SSE2, SSE3, SSSE3,
  SSE 4.1, SSE 4.2, PCLMULQDQ, AVX, and AVX2
- Instruction sets detected on ARM: NEON

.. image:: http://buildbot.steinwurf.dk/svgstatus?project=cpuid
    :target: http://buildbot.steinwurf.dk/stats?projects=cpuid

License
-------
cpuid license is based on the BSD License. Please refer to the LICENSE.rst
file for more details.

Platforms
---------
We have tested cpuid on various versions of Windows, Linux and Mac. We run
automated tests on x86 and ARM architectures with different compilers like
g++, clang and Microsoft Visual Studio.

You can see the status by selecting the cpuid project on the
`Steinwurf buildbot page <http://buildbot.steinwurf.dk:12344/>`_.

Build
-----
We use the ``waf`` build system to build the cpuid static library.
We have some additional waf tools which can be found at waf_.

.. _waf: https://github.com/steinwurf/waf

To configure and build cpuid, run the following commands::

  python waf configure
  python waf build

The ``waf configure`` command will download several auxiliary libraries
into a folder called ``bundle_dependencies`` within the cpuid folder.
You can also use the ``--bundle-path`` option to specify the download
location for the project dependencies::

  python waf configure --bundle-path=/my/path/to/bundle_dependencies

When building the static lib, waf will also build the ``print_cpuinfo_example``
executable which is useful to print the available CPU instruction sets.
The compiled binary is located in the ``build/[platform]/examples/print_cpuinfo``
folder (where ``[platform]`` denotes your current platform,
e.g. ``linux``, ``win32`` or ``darwin``).

Credits
-------
We have created cpuid to fit our specific needs, however we hope
that others may also find it useful. When designing cpuid we found
inspiration in these other nice projects:

* CPUID article on Wikipedia: http://en.wikipedia.org/wiki/CPUID
* zchotia's gist: https://gist.github.com/zchothia/3078968
* Facebook CPU ID implementation: https://github.com/facebook/folly/blob/master/folly/CpuId.h
* ARM Cortex-A Programmer's guide: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.den0013d/index.html
