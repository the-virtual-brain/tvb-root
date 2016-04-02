/*
 * sin_test.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */

#ifndef COS_TEST_CPP_
#define COS_TEST_CPP_

#include "asin.h"
#include "vdtdiag_helper.h"
using namespace vdt;
using namespace vdth;

int main(){

	constexpr uint32_t size=10;
	
        //dp
        double dpvals[size]={1,.9,.8,.6,1e-200,0,-0.00004,-.2,-.8,-0.9999999999};
        printFuncDiff("acos", (dpdpfunction)acos,(dpdpfunction)fast_acos, dpvals, size);
        printFuncDiff ("acosv", (dpdpfunctionv) acosv, (dpdpfunctionv) fast_acosv, dpvals, size );        
        
        //sp 
        float spvals[size]={1.f,.9f,.8f,.12f,1e-20f,0.f,-0.004f,-.2f,-.8f,-0.9999999999f};
        printFuncDiff("acosf", (spspfunction)acosf,(spspfunction)fast_acosf,spvals,size);
        printFuncDiff ("acosfv", (spspfunctionv) acosfv, (spspfunctionv) fast_acosfv, spvals, size );           
        
}

#endif /* COS_TEST_CPP_ */
