/*
* 
*	Very simple object for storing and parsing 
*	commandline parameters in unix fashion
*
*	Author: Ladislav Horky
*/

/* 
 * VDT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include<iostream>
#include<tuple>
#include<vector>
#include<string>
#include<map>

// tuple containing short indentifier, long identifier and help string
using opt_tuple = std::tuple<const std::string, const std::string, const std::string>;
using strpair = std::pair<std::string,std::string>;


class CmdOptions{
	std::map<std::string,std::string> receivedOpts;
	std::vector<opt_tuple> allowedOpts;
public:
	CmdOptions(){
		receivedOpts.clear();
		allowedOpts.clear();
		addOption("-h","--help","Prints help (no full functionality). All parametrized options must be in the form with "
			"'=' i.e. -o=parameter or --option=parameter, otherwise parameters will not be processed. Also, manual command"
			"specific help (like -o --hepl) does not work - works only in case of errorneous option.");
	}

        ~CmdOptions(){}

	int addOption(const std::string shrt, const std::string lng, const std::string help){
		allowedOpts.push_back(opt_tuple(shrt,lng,help));
		//std::cout << std::get<0>(allowedOpts[allowedOpts.size()-1]) << " " << std::get<1>(allowedOpts[allowedOpts.size()-1]) << "\n"; 
		return 1;
	}

	int parseCmd(int argc, char** argv){
		// omit the first argument
		for(int i=1; i<argc;i++){
			// prepare option string
			std::string rawOpt(argv[i]), optFlag;
			int eqPos = rawOpt.find('=');
			// cut the argument
			if(eqPos > 0){
				optFlag = rawOpt.substr(0,eqPos);
				//std::cout<< optFlag;
			}else
				optFlag = rawOpt;

			//find (and parse) option
			bool found = false;
			for(unsigned int j=0;j<allowedOpts.size();j++){
				std::string shortFlag = std::get<0>(allowedOpts[j]);

				// was short or long name of option used?
				if(shortFlag == optFlag || std::get<1>(allowedOpts[j]) == optFlag){
					found = true;
					// multiple specification fails
					if(receivedOpts.find(shortFlag) != receivedOpts.end()){
						std::cout << "Option " << shortFlag << "specified multiple times, which is forbidden.\n";
						return 0;
					}
					// if ok, add to received together with possible argument
					receivedOpts.insert(strpair(shortFlag,eqPos>0?rawOpt.substr(eqPos+1):""));
					break;
				}
			}
			if(!found){
				std::cout << "Unknown option " << optFlag << "\n";
				return 0;
			}
		}

		//print just help
		if(isSet("-h")){
			printHelp();
			//std::cout << "\nHelp printed, ignore any subsequent error messages.\n";
			return 1;
		}

		return 1;
	}

	bool isSet(const std::string shortFlag){
		if(receivedOpts.find(shortFlag) != receivedOpts.end())
			return true;
		else
			return false;
	}

	std::string getArgument(const std::string shortFlag){
		if(!isSet(shortFlag))
			return "";

		return std::get<1>(*receivedOpts.find(shortFlag));
	}

	//print help, if no command specified, print all
	void printHelp(const std::string opt = ""){
		
		// print option-specific help
		if(opt != ""){
			//find option
			bool found = false;
			for(unsigned int j=0;j<allowedOpts.size();j++){
					std::string shortFlag = std::get<0>(allowedOpts[j]);

					// try both short and long option name
					if(shortFlag == opt || std::get<1>(allowedOpts[j]) == opt){
						std::cout << "Option-specific help:\n" << shortFlag << "  " << std::get<1>(allowedOpts[j])
							<< "\n   " << std::get<2>(allowedOpts[j]) << "\n";
						found = true;
					}
			}
			if(!found)
				std::cout << "Unknown option " << opt << "\n";

		// print whole help
		}else{
			std::cout << "Help:\n";
			for(unsigned int j=0;j<allowedOpts.size();j++)
				std::cout << std::get<0>(allowedOpts[j]) << "  " << std::get<1>(allowedOpts[j]) 
					<< "\n   " << std::get<2>(allowedOpts[j]) << "\n";
		}
	}

};
