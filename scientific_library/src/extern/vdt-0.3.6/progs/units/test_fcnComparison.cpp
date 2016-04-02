/*
 * test_fcnResponse.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */
#include "vdtdiag_random.h"
#include "vdtdiag_fcnResponse.h"
#include "vdtdiag_fcnComparison.h"
#include "vdtdiag_helper.h"
#include <cmath>
#include "log.h"
#include "atan2.h"

#include <iostream>

int main(){
	
	const uint32_t size = 1000000;
	
	// Test the FcnResponse in double precision
	std::string dpofilename("test_dpfunctionComparison.txt");
	randomPool<double> dpRandomPool(100,1000,size);
	fcnResponse<double> dpLogResp("Log",dpRandomPool.getNumbers(), (vdth::dpdpfunction) log);
    fcnResponse<double> dpFastLogResp("Fast Log",dpRandomPool.getNumbers(), (vdth::dpdpfunction) vdt::fast_log);
    fcnComparison<double> dpLogComp("Log - libmVSvdt",
                                    dpRandomPool.getNumbers(),
                                    dpLogResp.getOutput(),
                                    dpFastLogResp.getOutput());
    dpLogComp.printStats();
	dpLogComp.writeFile(dpofilename);

	std::cout <<"Read from file: -----------------\n";
	fcnComparison<double> dpLogFromFile(dpofilename);
	dpLogFromFile.printStats();
        
	// Test the FcnResponse in single precision
	std::string spofilename("test_spfunctionComparison.txt");
	randomPool<float> spRandomPool(1,1000,size);
	fcnResponse<float> spLogResp("Logf",spRandomPool.getNumbers(), (vdth::spspfunction) logf);
    fcnResponse<float> spFastLogResp("Fast Logf",spRandomPool.getNumbers(), (vdth::spspfunction) vdt::fast_logf);
    fcnComparison<float> spLogComp("Logf - libmVSvdt",
                                    spRandomPool.getNumbers(),
                                    spLogResp.getOutput(),
                                    spFastLogResp.getOutput());
    spLogComp.printStats();
	spLogComp.writeFile(spofilename);

	std::cout <<"Read from file: -----------------\n";
	fcnComparison<float> spLogFromFile(spofilename);
	spLogFromFile.printStats();

	// Test the FcnResponse in single precision with 2 inputs
	std::string spofilename2D("test_spfunctionComparison2D.txt");
	randomPool2D<float> spRandomPool2D(-1,-1,1,1,size);
	fcnResponse2D<float> spLogResp2D("Atan2f",
									 spRandomPool2D.getNumbersX(),
									 spRandomPool2D.getNumbersY(),
									 (vdth::spsp2function) atan2f);
    fcnResponse2D<float> spFastLogResp2D("Fast Atan2f",
    								spRandomPool2D.getNumbersX(),
    								spRandomPool2D.getNumbersY(),
    								(vdth::spsp2function) vdt::fast_atan2f);
    fcnComparison2D<float> spLogComp2D("Atan2f - libmVSvdt",
                                    spRandomPool2D.getNumbersX(),
                                    spRandomPool2D.getNumbersY(),
                                    spLogResp2D.getOutput(),
                                    spFastLogResp2D.getOutput());
    spLogComp2D.printStats();
	spLogComp2D.writeFile(spofilename2D);

	std::cout <<"Read from file: -----------------\n";
	fcnComparison<float> spLogFromFile2D(spofilename2D);
	spLogFromFile2D.printStats();

	
}
