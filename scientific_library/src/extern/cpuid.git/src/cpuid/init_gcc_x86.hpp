// Copyright (c) 2013 Steinwurf ApS
// All Rights Reserved
//
// Distributed under the "BSD License". See the accompanying LICENSE.rst file.

#pragma once

#include <cstdint>

#include "cpuinfo_impl.hpp"
#include "extract_x86_flags.hpp"

namespace cpuid
{
    // Reference for this code is Intel's recommendation for detecting AVX2
    // on Haswell located here: http://goo.gl/c6IkGX
    void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd)
    {
        uint32_t ebx = 0, edx = 0;

# if defined( __i386__ ) && defined ( __PIC__ )
        // If PIC used under 32-bit, EBX cannot be clobbered
        // EBX is saved to EDI and later restored
        __asm__ ( "movl %%ebx, %%edi;"
                  "cpuid;"
                  "xchgl %%ebx, %%edi;"
                  : "=D"(ebx),
# else
        __asm__ ( "cpuid;"
                  : "+b"(ebx),
# endif
                  "+a"(eax), "+c"(ecx), "=d"(edx));

        abcd[0] = eax;
        abcd[1] = ebx;
        abcd[2] = ecx;
        abcd[3] = edx;
    }

    /// @todo Document
    void init_cpuinfo(cpuinfo::impl& info)
    {
        // Note: We need to capture these 4 registers, otherwise we get
        // a segmentation fault on 32-bit Linux
        uint32_t output[4];

        // The register information per input can be extracted from here:
        // http://en.wikipedia.org/wiki/CPUID

        // Set registers for basic flag extraction
        run_cpuid(1, 0, output);
        extract_x86_flags(info, output[2], output[3]);

        // Set registers for extended flags extraction
        run_cpuid(7, 0, output);
        extract_x86_extended_flags(info, output[1]);
    }
}
