/*
 * inv_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef INV_TEST_CPP_
#define INV_TEST_CPP_

#include "inv.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;
	
	// dp
	double dpvals[size]={1e200, 1e-17, 1, 0.5, .3, -10, -1e300, -.2, -101., -88};
	printFuncDiff("inv", (dpdpfunction) inv,(dpdpfunction)fast_inv,dpvals,size);
	printFuncDiff ("invv", (dpdpfunctionv) invv, (dpdpfunctionv) fast_invv, dpvals, size );
        printFuncDiff("approx_inv", (dpdpfunction) inv,(dpdpfunction)fast_approx_inv,dpvals,size);
        printFuncDiff ("approx_invv", (dpdpfunctionv) invv, (dpdpfunctionv) fast_approx_invv, dpvals, size );	
	
	//sp 
	float spvals[size]={1e20f, 1e-7f, 1.f, 0.5f, .3f, -10.f, -1e9f, -.2f, -101.f, -88.f};
        printFuncDiff("invf", (spspfunction) invf,(spspfunction)fast_invf,spvals,size);
        printFuncDiff ("invfv", (spspfunctionv) invfv, (spspfunctionv) fast_invfv, spvals, size );	
        printFuncDiff("approx_invf", (spspfunction) invf,(spspfunction) fast_approx_invf,spvals,size);
        printFuncDiff ("approx_invfv", (spspfunctionv) invfv, (spspfunctionv) fast_approx_invfv, spvals, size );  	

}

#endif /* SIN_TEST_CPP_ */
