/*
 * sin_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef SIN_TEST_CPP_
#define SIN_TEST_CPP_

#include "sin.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;

	constexpr double PI = M_PI;
	
	// dp
	double dpvals[size]={PI,PI/2.9,PI/4.1,PI/6.2,PI/7,PI/7,PI/85,PI/19,PI/10,PI/20.2};
	printFuncDiff("sin", (dpdpfunction)sin,(dpdpfunction)fast_sin,dpvals,size);
	printFuncDiff ("sinv", (dpdpfunctionv) sinv, (dpdpfunctionv) fast_sinv, dpvals, size );
	
	
	//sp 
	float spvals[size]={PI,PI/2.9,PI/4.1,PI/6.2,PI/7,PI/7,PI/85,PI/19,PI/10,PI/20.2};
	printFuncDiff("sinf", (spspfunction)sinf,(spspfunction)fast_sinf,spvals,size);
	printFuncDiff ("sinfv", (spspfunctionv) sinfv, (spspfunctionv) fast_sinfv, spvals, size );	
	

}

#endif /* SIN_TEST_CPP_ */
