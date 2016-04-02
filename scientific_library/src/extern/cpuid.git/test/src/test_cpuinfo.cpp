// Copyright (c) 2013 Steinwurf ApS
// All Rights Reserved
//
// Distributed under the "BSD License". See the accompanying LICENSE.rst file.

#include <cpuid/cpuinfo.hpp>

#include <cstdint>
#include <iostream>
#include <gtest/gtest.h>

#include "../commandline_arguments.hpp"

TEST(cpuinfo_tests, check_instruction_sets)
{
    cpuid::cpuinfo m_cpuinfo;

    // Check CPU capabilities
    EXPECT_EQ(variable_map["has_fpu"].as<bool>(), m_cpuinfo.has_fpu());
    EXPECT_EQ(variable_map["has_mmx"].as<bool>(), m_cpuinfo.has_mmx());
    EXPECT_EQ(variable_map["has_sse"].as<bool>(), m_cpuinfo.has_sse());
    EXPECT_EQ(variable_map["has_sse2"].as<bool>(), m_cpuinfo.has_sse2());
    EXPECT_EQ(variable_map["has_sse3"].as<bool>(), m_cpuinfo.has_sse3());
    EXPECT_EQ(variable_map["has_ssse3"].as<bool>(), m_cpuinfo.has_ssse3());
    EXPECT_EQ(variable_map["has_sse4_1"].as<bool>(), m_cpuinfo.has_sse4_1());
    EXPECT_EQ(variable_map["has_sse4_2"].as<bool>(), m_cpuinfo.has_sse4_2());
    EXPECT_EQ(variable_map["has_pclmulqdq"].as<bool>(),
              m_cpuinfo.has_pclmulqdq());
    EXPECT_EQ(variable_map["has_avx"].as<bool>(), m_cpuinfo.has_avx());
    EXPECT_EQ(variable_map["has_avx2"].as<bool>(), m_cpuinfo.has_avx2());
    EXPECT_EQ(variable_map["has_neon"].as<bool>(), m_cpuinfo.has_neon());
}
