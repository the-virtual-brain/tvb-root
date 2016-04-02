/*
 * test_fcnResponse.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */
#include "vdtdiag_random.h"
#include "vdtdiag_fcnResponse.h"
#include "vdtdiag_helper.h"
#include <cmath>
#include <exp.h>
#include <atan2.h>

#include <iostream>

int main(){
	
	const uint32_t size = 10;
	
	// Test the FcnResponse in double precision
	const std::string dpfilename("testDpFcnPerf.txt");
	randomPool<double> dpRandomPool(-500,500,size);
	fcnResponse<double> dpExpResp("Exp",dpRandomPool.getNumbers(), (vdth::dpdpfunction) exp);
	dpExpResp.writeFile(dpfilename);
	fcnResponse<double> dpExpRespFromFile(dpfilename);
	dpExpRespFromFile.print();
	dpExpRespFromFile.writeFile("testDpFcnPerf_fromFile.txt");

	// Test the FcnResponse in single precision
	const std::string spfilename("testSpFcnPerf.txt");
	randomPool<float> spRandomPool(-50,50,size);
	fcnResponse<float> spExpResp("Exp",spRandomPool.getNumbers(), (vdth::spspfunction) expf);
	spExpResp.writeFile(spfilename);
	fcnResponse<float> spExpRespFromFile(spfilename);
	spExpRespFromFile.print();
	spExpRespFromFile.writeFile("testSpFcnPerf_fromFile.txt");
	
	// 2 inputs

	// Test the FcnResponse in double precision
	const std::string dp2filename("testDp2FcnPerf.txt");
	randomPool2D<double> dp2RandomPool(-500,-500,500,500, size);
	fcnResponse2D<double> dpAtan2Resp("Atan2",
									   dp2RandomPool.getNumbersX(),
									   dp2RandomPool.getNumbersY(),
									   (vdth::dpdp2function) atan2);
	dpAtan2Resp.print();
	dpAtan2Resp.writeFile(dp2filename);

	fcnResponse2D<double> dpAtan2RespFromFile(dp2filename);
	dpAtan2RespFromFile.print();
	dpAtan2RespFromFile.writeFile("testDp2FcnPerf_fromFile.txt");

	// Test the FcnResponse in single precision
	const std::string sp2filename("testSp2FcnPerf.txt");
	randomPool2D<float> sp2RandomPool(-500,-500,500,500, size);
	fcnResponse2D<float> spAtan2Resp("Atan2",
							    	  sp2RandomPool.getNumbersX(),
								      sp2RandomPool.getNumbersY(),
								     (vdth::spsp2function) atan2);
	spAtan2Resp.print();
	spAtan2Resp.writeFile(sp2filename);

	fcnResponse2D<float> spAtan2RespFromFile(sp2filename);
	spAtan2RespFromFile.print();
	spAtan2RespFromFile.writeFile("testSp2FcnPerf_fromFile.txt");

}
