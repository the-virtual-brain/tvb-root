/*
 * cos_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef COS_TEST_CPP_
#define COS_TEST_CPP_

#include "cos.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;

	constexpr double PI = M_PI;
	
	// dp
	double dpvals[size]={32.*PI,5.6*PI,PI/4.1,PI/6.2,PI/7,PI/7.5,PI/85,PI/19,PI/10,0.000001};
	printFuncDiff("cos", (dpdpfunction)cos,(dpdpfunction)fast_cos,dpvals,size);
	printFuncDiff ("cosv", (dpdpfunctionv) cosv, (dpdpfunctionv) fast_cosv, dpvals, size );
// 	printFuncDiff("cos2", (dpdpfunction)cos,(dpdpfunction)fast_cos2,dpvals,size);
	
	//sp 
	float spvals[size]={PI,PI/2.9,PI/4.1,PI/6.2,PI/7,PI/7,PI/85,PI/19,PI/10,PI/20.2};
	printFuncDiff("cosf", (spspfunction)cosf,(spspfunction)fast_cosf,spvals,size);
	printFuncDiff ("cosfv", (spspfunctionv) cosfv, (spspfunctionv) fast_cosfv, spvals, size );	
	
}

#endif /* COS_TEST_CPP_ */
