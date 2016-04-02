/*
 * exp_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef EXP_TEST_CPP_
#define EXP_TEST_CPP_

#include "exp.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;

	// dp
	double dpvals[size]={-705,-100,-2,-1e-16,0,1e-50,4,10,500,805};
	printFuncDiff("exp", (dpdpfunction)exp,(dpdpfunction)fast_exp,dpvals,size);
	printFuncDiff ("expv", (dpdpfunctionv) expv, (dpdpfunctionv) fast_expv, dpvals, size );
	
	//sp 
	float spvals[size]={-87.f,-50.f,-2.f,-1e-10f,0.f,1e-10f,4.f,10.f,50.f,95.f};
	printFuncDiff("expf", (spspfunction)expf,(spspfunction)fast_expf,spvals,size);
	printFuncDiff ("expvf", (spspfunctionv) expfv, (spspfunctionv) fast_expfv, spvals, size );	
	

}

#endif /* EXP_TEST_CPP_ */
