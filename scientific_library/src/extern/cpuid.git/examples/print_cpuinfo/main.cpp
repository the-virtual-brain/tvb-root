// Copyright (c) 2013 Steinwurf ApS
// All Rights Reserved
//
// Distributed under the "BSD License". See the accompanying LICENSE.rst file.

#include <iostream>

#include <cpuid/cpuinfo.hpp>

int main()
{
    cpuid::cpuinfo m_cpuid;

    std::cout << "CPU has FPU?: "
        << (m_cpuid.has_fpu() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has MMX?: "
        << (m_cpuid.has_mmx() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has SSE?: "
        << (m_cpuid.has_sse() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has SSE2?: "
        << (m_cpuid.has_sse2() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has SSE3?: "
        << (m_cpuid.has_sse3() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has SSSE3?: "
        << (m_cpuid.has_ssse3() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has SSE4.1?: "
        << (m_cpuid.has_sse4_1() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has SSE4.2?: "
        << (m_cpuid.has_sse4_2() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has PCLMULQDQ?: "
        << (m_cpuid.has_pclmulqdq() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has AVX?: "
        << (m_cpuid.has_avx() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has AVX2?: "
        << (m_cpuid.has_avx2() ? "Yes" : "No") << std::endl;

    std::cout << "CPU has ARM NEON?: "
        << (m_cpuid.has_neon() ? "Yes" : "No") << std::endl;

    return 0;
}
