/*
 * log_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef LOG_TEST_CPP_
#define LOG_TEST_CPP_

#include "log.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;

	// dp
	double dpvals[size]={1e200,1.34e101,2,1e-16,0,1e-50,4,10,500,.1};
	printFuncDiff("log", (dpdpfunction)log,(dpdpfunction)fast_log,dpvals,size);
	printFuncDiff ("logv", (dpdpfunctionv) logv, (dpdpfunctionv) fast_logv, dpvals, size );
	
	//sp 
	float spvals[size]={-87.f,-50.f,-2.f,-1e-1f,0.f,1e-5f,4.f,10.f,50.f,95.f};
	printFuncDiff("logf", (spspfunction)logf,(spspfunction)fast_logf,spvals,size);
	printFuncDiff ("logvf", (spspfunctionv) logfv, (spspfunctionv) fast_logfv, spvals, size );		
	
}


#endif /* LOG_TEST_CPP_ */
