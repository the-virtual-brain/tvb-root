/*
 * test_randomPool.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: danilopiparo
 */
#include "vdtdiag_random.h"

#include <iostream>

int main(){
	
	constexpr uint32_t size = 10;

    // 1D
	
	// Test the Random Pool in double precision
	std::string dpfilename("testDpRandomNumbers.txt");
	randomPool<double> dprp_fromScratch(1,2,size, 3);
	dprp_fromScratch.writeFile(dpfilename);
	dprp_fromScratch.print();
	randomPool<double> dprp_fromFile(dpfilename);
	dprp_fromFile.print();
	dpfilename="testDpRandomNumbers_rewritten.txt";
	dprp_fromFile.writeFile(dpfilename);
	
	// Test the Random Pool in single precision
	std::string spfilename("testSpRandomNumbers.txt");
	randomPool<float> sprp_fromScratch(-2e2,2e2,size);
	sprp_fromScratch.print();
	sprp_fromScratch.writeFile(spfilename);
	randomPool<float> sprp_fromFile(spfilename);	
	sprp_fromFile.print();
	spfilename="testSpRandomNumbers_rewritten.txt";
	sprp_fromFile.writeFile(spfilename);
	
	// 2D
	// Test the Random Pool in double precision
	std::string dpfilename2D("testDpRandomNumbers2D.txt");
	randomPool2D<double> dprp_fromScratch2D(1,2,4,7,size, 3);
	dprp_fromScratch2D.print();
	dprp_fromScratch2D.writeFile(dpfilename2D);
	randomPool2D<double> dprp_fromFile2D(dpfilename2D);
	dprp_fromFile2D.print();
	dpfilename2D="testDpRandomNumbers2D_rewritten.txt";
	dprp_fromFile2D.writeFile(dpfilename2D);
	
	// Test the Random Pool in single precision
	std::string spfilename2D("testSpRandomNumbers2D.txt");
	randomPool2D<float> sprp_fromScratch2D(-2e2, -1e3 ,2e2, 1e4,size);
	sprp_fromScratch2D.print();
	sprp_fromScratch2D.writeFile(spfilename2D);
	randomPool2D<float> sprp_fromFile2D(spfilename2D);
	sprp_fromFile2D.print();
	spfilename2D="testSpRandomNumbers2D_rewritten.txt";
	sprp_fromFile2D.writeFile(spfilename2D);

}



