/*
 * test_fcnPerformance.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */
#include "vdtdiag_random.h"
#include "vdtdiag_fcnPerformance.h"
#include "vdtMath.h"
#include "vdtdiag_helper.h"

#include <iostream>
  
using namespace vdt;

int main(){
	
    constexpr uint32_t size = 10000;
    constexpr uint32_t repetitions = 100;

    randomPool<double> dpRandomPool(-500,500,size);


    // Test the FcnResponse in double precision
    fcnPerformance<double> dpExpPerf("Exp",
                                     dpRandomPool.getNumbers(), 
                                    (vdth::dpdpfunction) exp,repetitions);
    dpExpPerf.print();

    fcnPerformance<double> dpFastExpPerf("Fast Exp",
                                             dpRandomPool.getNumbers(), 
                                             (vdth::dpdpfunction) fast_exp,repetitions);
    dpFastExpPerf.print(); //--------------------------------------------------


   // Test the FcnResponse in double precision, array signature
    fcnPerformance<double> dpExpvPerf("Expv",
                                         dpRandomPool.getNumbers(), 
                                         (vdth::dpdpfunctionv) expv,repetitions);
    dpExpvPerf.print();

    fcnPerformance<double> dpFastExpvPerf("Fast Expv",
                                             dpRandomPool.getNumbers(), 
                                             (vdth::dpdpfunctionv) fast_expv,repetitions);
    dpFastExpvPerf.print(); //--------------------------------------------------


    // Test the FcnResponse in double precision
    randomPool<float> spRandomPool(-500,500,size);
    fcnPerformance<float> spExpPerf("Expf",
                                          spRandomPool.getNumbers(), 
                                          (vdth::spspfunction) expf,repetitions);
    spExpPerf.print();

    fcnPerformance<float> spFastExpPerf("Fast Expf",
                                             spRandomPool.getNumbers(), 
                                             (vdth::spspfunction) fast_expf,repetitions);
    spFastExpPerf.print(); //--------------------------------------------------

   // Test the FcnResponse in double precision, array signature
    fcnPerformance<float> spExpvPerf("Expfv",
                                         spRandomPool.getNumbers(), 
                                         (vdth::spspfunctionv) expfv,repetitions);
    spExpvPerf.print();

    fcnPerformance<float> spFastExpvPerf("Fast Expfv",
                                             spRandomPool.getNumbers(), 
                                             (vdth::spspfunctionv) fast_expfv,repetitions);
    spFastExpvPerf.print(); //--------------------------------------------------



}
