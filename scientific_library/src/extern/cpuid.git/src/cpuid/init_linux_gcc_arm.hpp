// Copyright (c) 2013 Steinwurf ApS
// All Rights Reserved
//
// Distributed under the "BSD License". See the accompanying LICENSE.rst file.

#pragma once

#include <cstdio>
#include <cassert>
#include <cstring>

#include <unistd.h>
#include <fcntl.h>
#include <elf.h>
#include <linux/auxvec.h>

#include "cpuinfo_impl.hpp"

namespace cpuid
{
    /// @todo docs
    void init_cpuinfo(cpuinfo::impl& info)
    {
        // Check NEON instruction set flag
        // Follow recommendation from Cortex-A Series Programmer's guide
        // on Section 20.1.7 Detecting NEON. The guide is available at:
        // Steinwurf's Google drive: steinwurf/technical/experimental/cpuid

        auto cpufile = open("/proc/self/auxv", O_RDONLY);
        assert(cpufile);

        Elf32_auxv_t auxv;

        if (cpufile >= 0)
        {
            const auto size_auxv_t = sizeof(Elf32_auxv_t);
            while (read(cpufile, &auxv, size_auxv_t) == size_auxv_t)
            {
                if (auxv.a_type == AT_HWCAP)
                {
                    info.m_has_neon = (auxv.a_un.a_val & 4096) != 0;
                    break;
                }
            }

            close (cpufile);
        }
        else
        {
            info.m_has_neon = false;
        }
    }
}
