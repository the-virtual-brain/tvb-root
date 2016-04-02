// Copyright (c) 2013 Steinwurf ApS
// All Rights Reserved
//
// Distributed under the "BSD License". See the accompanying LICENSE.rst file.

#pragma once

#include <intrin.h>

#include "cpuinfo_impl.hpp"
#include "extract_x86_flags.hpp"

namespace cpuid
{
    /// @todo docs
    void init_cpuinfo(cpuinfo::impl& info)
    {
        int registers[4];

        /// According to the msvc docs eax, ebx, ecx and edx are
        /// stored (in that order) in the array passed to the __cpuid
        /// function.

        // The register information per input can be extracted from here:
        // http://en.wikipedia.org/wiki/CPUID

        // Set registers for basic flag extraction, eax=1
        __cpuid(registers, 1);
        extract_x86_flags(info, registers[2], registers[3]);

        // Set registers for extended flags extraction, eax=7 and ecx=0
        // using __cpuidex ensures that ecx = 0.

        __cpuidex(registers, 7, 0);
        extract_x86_extended_flags(info, registers[1]);
    }
}
