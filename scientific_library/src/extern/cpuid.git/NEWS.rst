News for cpuid
==============

This file lists the major changes between versions. For a more detailed list of
every change, see the Git log.

Latest
------
* Minor: Added buildbot.py for coverage reports.

4.0.0
-----
* Major: Upgrade to waf-tools 3
* Major: Upgrade to platform 2
* Minor: Upgrade to boost 2
* Minor: Upgrade to gtest 3

3.3.1
-----
* Patch: Added missing ``cstdint`` includes

3.3.0
-----
* Minor: Added init_msvc_arm.hpp to enable the NEON instruction set on
  Windows Phone 8 (and above).

3.2.1
-----
* Patch: Fix version define.

3.2.0
-----
* Patch: On win32 use __cpuidex when detecting extended features (such as
  AVX2). This bug was reported by Ilya Lavrenov <ilya.lavrenov@itseez.com>.
* Minor: Update to waf 1.8.0-pre1
* Minor: Made python files comply with pep8
* Minor: Added version define.

3.1.0
-----
* Minor: Added detection for AVX2 instruction set
* Patch: Fixed issue with the -fPIC compiler flag on x86.

3.0.0
-----
* Major: Platform detection is now handled by the 'platform' library

2.0.0
-----
* Major: Renamed Clang detection macros to always use the name CLANG instead of
  LLVM.

1.0.0
-----
* Major: Initial release with support for x86 and ARM architectures.
