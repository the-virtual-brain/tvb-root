/*
 * tan_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef TAN_TEST_CPP_
#define TAN_TEST_CPP_

#include "tan.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;

	constexpr double PI = M_PI;
	
	// dp
	double dpvals[size]={0.01, 0.1+PI/4, 0.01+PI/2, 1176.2*PI,0,-PI/7.5,PI/85,-PI/19,PI/10,-0.000001};
	printFuncDiff("tan", (dpdpfunction)tan,(dpdpfunction)fast_tan,dpvals,size);
	printFuncDiff ("tanv", (dpdpfunctionv) tanv, (dpdpfunctionv) fast_tanv, dpvals, size );
	
	//sp 
	float spvals[size]={0.01, 0.1+PI/4, 0.01+PI/2, 1176.2*PI,PI/7,-PI/7.5,PI/85,-PI/19,PI/10,-0.000001};
	printFuncDiff("tanf", (spspfunction)tanf,(spspfunction)fast_tanf,spvals,size);
	printFuncDiff ("tanfv", (spspfunctionv) tanfv, (spspfunctionv) fast_tanfv, spvals, size );	
	
}

#endif /* TAN_TEST_CPP_ */
