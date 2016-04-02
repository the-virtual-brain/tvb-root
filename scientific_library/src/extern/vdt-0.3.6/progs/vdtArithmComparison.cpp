/* vdtArithmComparison.cpp
*
*	created: 16.7.2012
*
*	Reads files dumped by vdtAtritmBenchmark.cpp 
*
*	Author: Ladislav Horky
*/

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "vdtdiag_fcnResponse.h"
#include "vdtdiag_fcnTuples.h"
#include "vdtdiag_fcnComparison.h"
#include "vdtdiag_simpleCmd.h"


//converts (comma) separated list (in string) to vector of strings
void list2vector(std::string csList, std::vector<std::string>* vect, char separator = ','){
	int lastCommaPos = -1, commaPos;

	vect->clear();
	while(true){
		//no other comma, break
		commaPos = csList.find(separator,lastCommaPos+1);
		if(commaPos < 0)
			break;
		//store string between commas
		vect->push_back(csList.substr(lastCommaPos+1,commaPos-lastCommaPos-1));
		lastCommaPos = commaPos;
	}
	//process the last string after the last separator if there is any
	if(int(csList.size()) > lastCommaPos+1)
		vect->push_back(csList.substr(lastCommaPos+1));
}

/// Performs comparison of arithmetic precision

int main(int argc, char **argv){

	// set and parse commandline options
	CmdOptions opt;
	opt.addOption("-n","--nick","Output (!) file nickname. Make sure it contains all information about compared function "
		"(libraries used, vect VS single signature comaprison...).");
	opt.addOption("-R","--reference","List of comma-separated filenames that will be used as a reference values"
		"(i.e. will be in the second column of output comparison file). The propper function names for output files will "
		"be refined from the filenames, so do not rename the files from vdtArithmBenchmark. See -T for more.");
	opt.addOption("-T","--test","List of comma-separated filenames that will be compared to reference files"
		"(their values will appear in the third column of output file). It is the responsibility of the user"
		"to make sure the reference and the test files are compatibile (same functions, same order of files, same # of entries, "
		"same inputs values...).");
	//opt.addOption("-s","--separator","Alternative seperator for the list of files (default: ',')");

	// parse and process cmd options
	if(!opt.parseCmd(argc,argv)){
		std::cout << "Something is wrong with cmd options, try --help\n"
			<< "usage: vdtArithmComparison -n=<out_nick> -R=<reffile1,reffile2...> -T=<testfile1,testfile2...>";
		return 1;
	}
	// if help was printed, exit
	if(opt.isSet("-h"))
		return 0;

	std::string nick;
	std::vector<std::string> ref, test;

	// process cmd options
	nick = opt.getArgument("-n");
	if(nick == ""){
		std::cout << "Error: Nickname was not specified!\n";
		opt.printHelp("-n");
		return 1;
	}

	std::string tmp;
	tmp = opt.getArgument("-R");
	list2vector(tmp,&ref);
	if(!ref.size()){
		std::cout << "Error: No reference files specified!\n";
		opt.printHelp("-R");
		return 1;
	}
	tmp = opt.getArgument("-T");
	list2vector(tmp,&test);
	if(!test.size()){
		std::cout << "Error: No test files specified!\n";
		opt.printHelp("-T");
		return 1;
	}
	if(test.size() != ref.size()){
		std::cout << "Error: Number of test and ref files not equal!\n";
		opt.printHelp();
		return 1;
	}
	// pheeew.. all errors hopefully checked

	//print table column names:--------------------------------------------------------------------
	std::cout << std::setw(30) << std::left << "function name"
			<< std::setw(5) << std::right << "max"
			<< std::setw(5) << std::right << "min"
			<< std::setw(6) << std::right << "mean"
			<< std::setw(6) << std::right << "RMS\n";

	for(unsigned int i=0; i<ref.size(); i++){
		//try to open files
		std::ifstream ifile(ref[i]);
		if(!ifile.is_open()){
			std::cout << "Cannot find the " << ref[i] << " file, skipping.\n";
			continue;
		}
		std::string fcname;
		bool singlePrec;
		std::getline(ifile,tmp);
		tmp.clear();

		//determine type
		std::getline(ifile,tmp);
		singlePrec = (tmp == "Double Precision")?false:true;

		//store function name (without Fast_prefix, if there is any)
		tmp.clear();
		std::getline(ifile,tmp);
		tmp = tmp.substr(tmp.find('=')+2);		//get rid of the 'Function name = ' string
		fcname = tmp.substr(tmp.find('_')+1);	//remove Fast_/VC_ prefix if present
		ifile.close();

		//check for test file;
		ifile.open(test[i]);
		if(!ifile.is_open()){
			std::cout << "Cannot find the " << test[i] << " file, skipping.\n";
			continue;
		}
		ifile.close();

		//read responses, perform comparison
		bool is_atan2 = ref[i].find("Atan2") != std::string::npos ? true:false;
		if(!singlePrec){
			if (is_atan2){
				fcnResponse2D<double> rrefDP(ref[i]),rtestDP(test[i]);
				fcnComparison2D<double> cDP(fcname+" "+ref[i].substr(0,ref[i].find("__"))+" VS "+
												   	test[i].substr(0,test[i].find("__")),
										  rrefDP.getInput1(),
										  rrefDP.getInput2(),
										  rrefDP.getOutput(),rtestDP.getOutput());
				cDP.printStats(true);
				cDP.writeFile("comparison__"+nick+"__"+fcname+".txt");
			}
			else{
				fcnResponse<double> rrefDP(ref[i]),rtestDP(test[i]);
				fcnComparison<double> cDP(fcname+" "+ref[i].substr(0,ref[i].find("__"))+" VS "+
													test[i].substr(0,test[i].find("__")),
										  rrefDP.getInput(),rrefDP.getOutput(),rtestDP.getOutput());
				cDP.printStats(true);
				cDP.writeFile("comparison__"+nick+"__"+fcname+".txt");
				}
		}else{
			if (is_atan2){
				fcnResponse2D<float> rrefSP(ref[i]),rtestSP(test[i]);
				fcnComparison2D<float> cSP(fcname+" "+ref[i].substr(0,ref[i].find("__"))+" VS "+
													test[i].substr(0,test[i].find("__")),
													rrefSP.getInput1(),
													rrefSP.getInput2(),
													rrefSP.getOutput(),rtestSP.getOutput());
				cSP.printStats(true);
				cSP.writeFile("comparison__"+nick+"__"+fcname+".txt");
			}
			else{
				fcnResponse<float> rrefSP(ref[i]),rtestSP(test[i]);
				fcnComparison<float> cSP(fcname+" "+ref[i].substr(0,ref[i].find("__"))+" VS "+
													test[i].substr(0,test[i].find("__")),
										 rrefSP.getInput(),rrefSP.getOutput(),rtestSP.getOutput());
				cSP.printStats(true);
				cSP.writeFile("comparison__"+nick+"__"+fcname+".txt");
			}
		}
	}


	return 0;
}
