/*
 * sin_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef SIN_TEST_CPP_
#define SIN_TEST_CPP_

#include "asin.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;
	
	// dp
	double dpvals[size]={1,.9,.8,.6,1e-200,0,-0.00004,-.2,-.8,-0.9999999999};
	printFuncDiff("asin", (dpdpfunction)asin,(dpdpfunction)fast_asin,dpvals,size);
	printFuncDiff ("asinv", (dpdpfunctionv) asinv, (dpdpfunctionv) fast_asinv, dpvals, size );
	
	
	//sp 
	float spvals[size]={1,.9,.8,.12,1e-200,0,-0.004,-.2,-.8,-0.9999999999};
	printFuncDiff("asinf", (spspfunction)asinf,(spspfunction)fast_asinf,spvals,size);
	printFuncDiff ("asinfv", (spspfunctionv) asinfv, (spspfunctionv) fast_asinfv, spvals, size );	
        
}

#endif /* SIN_TEST_CPP_ */
