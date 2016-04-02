/*
 * atan_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef ATAN_TEST_CPP_
#define ATAN_TEST_CPP_

#include "atan.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;
	
	// dp
	double dpvals[size]={-1e200,-1e50,-300.,-20.,0.,13.,230.,1e20,1e303};
	printFuncDiff("atan", (dpdpfunction)atan,(dpdpfunction)fast_atan,dpvals,size);
	printFuncDiff ("atanv", (dpdpfunctionv) atanv, (dpdpfunctionv) fast_atanv, dpvals, size );

	//sp 
        float spvals[size]={-1e30f,-1e19f,-300.f,-20.f,0.f,13.f,230.f,1e20f,1e30f};
        printFuncDiff("atanf", (spspfunction)atanf,(spspfunction)fast_atanf,spvals,size);
        printFuncDiff ("atanfv", (spspfunctionv) atanfv, (spspfunctionv) fast_atanfv, spvals, size );
	
}

#endif /* ATAN_TEST_CPP_ */
