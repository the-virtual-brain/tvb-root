/* vdtArithmBenchmark.cpp
*
*	created: 13.7.2012
*
*	Dumps fcnResponce of every function to the file.
*	Created for bunch-testing as preceeding step to the
*	bunch-fcnComparison
*
*	Author: Ladislav Horky
*/

#include "vdtMath.h"
#include "vdtdiag_random.h"
#include "vdtdiag_fcnResponse.h"
#include "vdtdiag_fcnTuples.h"
#include "vdtdiag_simpleCmd.h"

#include <iostream>
#include <string>

const double spool_max = 5000; //1.0e8;
const double spool_min = -spool_max;
const double ppool_min = .0001; //1e-10;

const float spool_maxf = 5000;//8192;
const float spool_minf = -spool_maxf;
const float ppool_minf = .0001f;//8192;


template<typename T, template<class> class Tuple>
void saveResponses(const Tuple<T>& fcn_tuple, const std::string nickname){
	fcnResponse<T> response(std::get<0>(fcn_tuple),std::get<2>(fcn_tuple),std::get<1>(fcn_tuple));
	// create filename: <nickname>_<fcnname>_response.txt
	std::string fname = nickname;
	// two dashes to prevent i.e Fast_something mess
	fname += "__";
	fname += std::get<0>(fcn_tuple);
	fname += "__response.txt";
	// dump to file
	response.writeFile(fname);
}

template<typename T,class FUNC>
void saveResponses2D(const std::string& fcn_name,
        			 FUNC fcn,
		             const std::vector<T>& randomX,
		             const std::vector<T>& randomY,
		             const std::string& nickname){
	fcnResponse2D<T> response(fcn_name, randomX,randomY,fcn);
	// create filename: <nickname>_<fcnname>_response.txt
	std::string fname = nickname;
	// two dashes to prevent i.e Fast_something mess
	fname += "__";
	fname += fcn_name;
	fname += "__response.txt";
	// dump to file
	response.writeFile(fname);
}


int main(int argc, char **argv){

	// set and parse commandline options
	CmdOptions opt;
	opt.addOption("-n","--nick","Nickname to distinguish different runs/libraries used (required)");
	opt.addOption("-s","--size","# of numbers to be tested (default 50000)");

	std::string nick = " ";
	uint32_t SIZE = 50000;

	if(!opt.parseCmd(argc,argv)){
		std::cout << "Something is wrong with cmd options, try --help\n"
			<<"usage: vdtArithmBenchmark -n=<run88>";
		return 0;
	}
	// if help was printed, exit
	if(opt.isSet("-h"))
		return 1;

	// process cmd options
	nick = opt.getArgument("-n");
	if(nick == ""){
		std::cout << "Error: Nickname was not specified!\n";
		opt.printHelp("-n");
		return 0;
	}
	

	//getArgument() contains isSet check 
	if(opt.getArgument("-s") != "")
		SIZE = atoi(opt.getArgument("-s").c_str());

	// print basic info
	std::cout << "Nick = " << nick << ", size = " << SIZE << "\n";
	std::cout<<"Starting...\n";

	randomPool<double> *spool,*ppool,*onepool,*exppool;
	randomPool2D<double> *spool2D;
	spool = new randomPool<double>(spool_min,spool_max,SIZE);
	spool2D = new randomPool2D<double>(-1.,-1.,1.,1.,SIZE);
	ppool = new randomPool<double>(ppool_min,spool_max,SIZE);
	onepool = new randomPool<double>(-1.0,1.0,SIZE);
	exppool = new randomPool<double>(-705.,705.,SIZE);

	//======================DP=========================
	std::vector<genfpfcn_tuple<double>> dpTuples;

	//retrieve tuples
	getFunctionTuples(&dpTuples,*spool,*ppool,*onepool,*exppool);

	//loop over & save to file
	std::cout <<"Saving double precision\n";
	for(const auto& i_tuple : dpTuples){
                std::cout << " - Processing " << std::get<0>(i_tuple) << std::endl;
		saveResponses<double,genfpfcn_tuple>(i_tuple,nick);
                }

	//========================DPv======================
	std::vector<genfpfcnv_tuple<double>> dpvTuples;

	//retrieve tuples
	getFunctionTuplesvect(&dpvTuples,*spool,*ppool,*onepool,*exppool);

	//loop over & save to file
	std::cout <<"Saving double precision vector form\n";
	for(const auto& i_tuple : dpvTuples){
                std::cout << " - Processing " << std::get<0>(i_tuple) << std::endl;
		saveResponses<double,genfpfcnv_tuple>(i_tuple,nick);
        }

	// Add atan2
	std::cout << " - Processing atan2 (all flavours)\n";
	saveResponses2D<double,vdth::dpdp2function> ("Atan2",
    		atan2,
    		spool2D->getNumbersX(),spool2D->getNumbersY(),
    		nick);
    saveResponses2D<double,vdth::dpdp2function> ("Fast_Atan2",
    		vdt::fast_atan2,
    		spool2D->getNumbersX(),spool2D->getNumbersY(),
    		nick);

    saveResponses2D<double,vdth::dpdp2functionv> ("Atan2v",
    		vdt::atan2v,
    		spool2D->getNumbersX(),spool2D->getNumbersY(),
    		nick);
    saveResponses2D<double,vdth::dpdp2functionv> ("Fast_Atan2v",
    		vdt::fast_atan2v,
    		spool2D->getNumbersX(),spool2D->getNumbersY(),
    		nick);

	//delete to spare memeory..
	delete spool;
	delete ppool;
	delete onepool;
	delete exppool;
	delete spool2D;

	randomPool<float>* fspool,*fppool,*fonepool,*fexppool;
	randomPool2D<float>* fspool2D;
	fspool = new randomPool<float>(spool_minf,spool_maxf,SIZE);
	fppool = new randomPool<float>(ppool_minf,spool_maxf,SIZE);
	fonepool = new randomPool<float>(-1.0,1.0,SIZE);
	fexppool = new randomPool<float>(-85,85,SIZE);
	fspool2D = new randomPool2D<float>(-1.f,-1.f,1.f,1.f,SIZE);

	//======================SP=========================
	std::vector<genfpfcn_tuple<float>> spTuples;

	//retrieve tuples
	getFunctionTuples(&spTuples,*fspool,*fppool,*fonepool,*fexppool);

	//loop over & save to file
	std::cout <<"Saving single precision\n";
	for(const auto& i_tuple : spTuples){
                std::cout << " - Processing " << std::get<0>(i_tuple) << std::endl;
		saveResponses<float,genfpfcn_tuple>(i_tuple,nick);
        }

	//======================SPv=========================
	std::vector<genfpfcnv_tuple<float>> spvTuples;

	//retrieve tuples
	getFunctionTuplesvect(&spvTuples,*fspool,*fppool,*fonepool,*fexppool);

	//loop over & save to file
	std::cout <<"Saving single precision vector form\n";
	for(const auto& i_tuple : spvTuples){
                std::cout << " - Processing " << std::get<0>(i_tuple) << std::endl;
		saveResponses<float,genfpfcnv_tuple>(i_tuple,nick);
        }

	// Add atan2
	std::cout << " - Processing atan2 (all flavours)\n";
    saveResponses2D<float,vdth::spsp2function> ("Atan2f",
    		atan2f,
    		fspool2D->getNumbersX(),fspool2D->getNumbersY(),
    		nick);
    saveResponses2D<float,vdth::spsp2function> ("Fast_Atan2f",
    		vdt::fast_atan2f,
    		fspool2D->getNumbersX(),fspool2D->getNumbersY(),
    		nick);
    saveResponses2D<float,vdth::spsp2functionv> ("Atan2fv",
    		vdt::atan2fv,
    		fspool2D->getNumbersX(),fspool2D->getNumbersY(),
    		nick);
    saveResponses2D<float,vdth::spsp2functionv> ("Fast_Atan2fv",
    		vdt::fast_atan2fv,
    		fspool2D->getNumbersX(),fspool2D->getNumbersY(),
    		nick);


	//delete to spare memeory
	delete fspool;
	delete fppool;
	delete fonepool;
	delete fexppool;
	delete fspool2D;

	return 0;
}
  
