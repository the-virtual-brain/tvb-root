/*
 * sqrt_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef SQRT_TEST_CPP_
#define SQRT_TEST_CPP_

#include "sqrt.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;

	// dp
	double dpvals[size]={1e200,1.34e101,2,1e-16,0,1e-50,4,10,500,.1};
	// the vanilla one
	printFuncDiff("isqrt", (dpdpfunction) isqrt,(dpdpfunction)fast_isqrt,dpvals,size);
	printFuncDiff ("sqrtv", (dpdpfunctionv) isqrtv, (dpdpfunctionv) fast_isqrtv, dpvals, size );
	// the approximated one!
	printFuncDiff("approx_isqrt", (dpdpfunction) isqrt,(dpdpfunction)fast_approx_isqrt,dpvals,size);
	printFuncDiff ("approx_isqrtv", (dpdpfunctionv) isqrtv, (dpdpfunctionv) fast_approx_isqrtv, dpvals, size );
	
	//sp 
	float spvals[size]={87.f,50.f,2.f,1e-5f,0.f,1e-8f,4.f,10.f,50.f,95.f};
	printFuncDiff("sqrtf", (spspfunction) isqrtf,(spspfunction) fast_isqrtf,spvals,size);
	printFuncDiff ("sqrtvf", (spspfunctionv) isqrtfv, (spspfunctionv) fast_isqrtfv, spvals, size );		
	
	printFuncDiff("approx_sqrtf", (spspfunction) isqrtf,(spspfunction)fast_approx_isqrtf,spvals,size);
	printFuncDiff ("approx_sqrtvf", (spspfunctionv) isqrtfv, (spspfunctionv) fast_approx_isqrtfv, spvals, size );
	
}


#endif /* SQRT_TEST_CPP_ */
