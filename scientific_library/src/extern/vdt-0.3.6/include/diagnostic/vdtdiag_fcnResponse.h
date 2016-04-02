/**
 * This file contains the classes to store the 
 * arithmetical performance of the mathematical functions.
 * 
 * Author Danilo Piparo
 * 
 **/

#include "assert.h"
#include <algorithm>
#include <limits>
#include <vector>
#include "vdtdiag_interfaces.h"
#include "vdtdiag_filePersistence.h"


/**
 * Class that represents the response of a mathematical function.
 * The quantities stored are the input numbers and the output numbers.
 * The Ascii file i/o is supported. A dump on an ascii file can be performed
 * as well as the object construction from ascii file.
 **/

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

template<class T>
class fcnResponse1D:IfcnResponse<T>{
public:
	fcnResponse1D(const std::string& fcnName, const std::vector<T>& input, const std::function<T(T)>& fcn):
		IfcnResponse<T>(fcnName,input),
		m_fcn(fcn){
		for (auto& inp : this->getInput() )
			pushOutputVal(fcn(inp));
	}
	//-----------------------------------------------------------------------------

	/// Construct from name, input and vector function
	fcnResponse1D(const std::string& fcnName, const std::vector<T>& input, const std::function<void (uint32_t, T*, T*)>& fcnv):
		IfcnResponse<T>(fcnName,input),
		m_fcnv(fcnv){
		//m_output.reserve(size);
		const uint64_t size = this->getInput().size();
		T* input_arr = const_cast<T*> (&this->getInput()[0]);
        T* output_arr = new T[size];
		fcnv(size,input_arr,output_arr);
		for (uint32_t i=0;i<size;++i )
		  pushOutputVal(output_arr[i]);
		delete [] output_arr;
	}
	//-----------------------------------------------------------------------------

	/// Construct from ascii file
	fcnResponse1D(const std::string& input_filename):
		IfcnResponse<T>(input_filename){
		std::ifstream ifile ( input_filename );
		std::string line;
		//skip the 5 header lines
		for (uint16_t i=0;i<5;++i)
			std::getline(ifile,line);
		//read data from file
		fpFromHex<T> in_val, out_val;
		while(ifile >> in_val >> out_val) {
			pushInputVal(in_val.getValue());
			pushOutputVal(out_val.getValue());
		}
	}
	//-----------------------------------------------------------------------------

	~fcnResponse1D(){};
    //-----------------------------------------------------------------------------

	/// Return the input
    inline std::vector<T>& getInput() {return IfcnResponse<T>::getInput1();};
    void pushInputVal(T val) {IfcnResponse<T>::pushInput1Val(val);};
    inline const T outputVal(uint64_t index) {return IfcnResponse<T>::outputVal(index);};

    //-----------------------------------------------------------------------------

	/// Return the input
    inline std::vector<T>& getOutput() {return IfcnResponse<T>::getOutput();};
    void pushOutputVal(T val) {IfcnResponse<T>::pushOutputVal(val);};
    inline const T inputVal(uint64_t index) {return IfcnResponse<T>::input1Val (index);};

	//-----------------------------------------------------------------------------

	/// Dump on ascii file
	void writeFile(const std::string& output_filename) {
		const std::string preamble("VDT function arithmetics performance 1 input file (the first 5 lines are the header)\n");
		std::ofstream ofile ( output_filename );
		// Copy the input file if the object was created from file
		if (IfcnResponse<T>::isFromFile()){
			std::string line;
			std::ifstream ifile ( IfcnResponse<T>::getIfileName() );
			getline(ifile,line);
			ofile << "Dumped by an object initialised by " << IfcnResponse<T>::getIfileName() << " - "
					<< preamble;
			ofile << ifile.rdbuf() ;
		}
		else{ // Write an header and the numbers in the other case
			ofile << preamble;
			if (sizeof(T)==8) // some kind of RTTC
				ofile << "Double Precision\n";
			else
				ofile << "Single Precision\n";
			ofile << "Function Name = " <<  IfcnResponse<T>::getFcnName() << std::endl
				  << "--\n--\n";
			// Do not write dec, but HEX!
			for (uint32_t i=0;i<this->getInput().size();++i)
				ofile << fpToHex<T>(inputVal(i)) << fpToHex<T>(outputVal(i)) << std::endl;
		}
	}

	//-----------------------------------------------------------------------------

	/// Print to screen
	void print() {
		const uint64_t size=getInput().size();
		std::cout << "Function Performance (1 single input):\n";
		std::cout << std::setprecision(std::numeric_limits<T>::digits10);
		for (uint64_t i=0;i<size;++i)
			std::cout << i << "/" << size << " " << IfcnResponse<T>::getFcnName()
					<< "(" << inputVal(i) << ") = " << outputVal(i)
					<< std::endl;

	}
    //-----------------------------------------------------------------------------
private:
	const std::function<T(T)> m_fcn;
	const std::function<void(const uint32_t, T*, T*)> m_fcnv;
};

template<class T>
class fcnResponse2D:IfcnResponse<T>{
public:
	fcnResponse2D(const std::string& fcnName,
				const std::vector<T>& input1,
				const std::vector<T>& input2,
				const std::function<T(T,T)>& fcn):
		IfcnResponse<T>(fcnName,input1,input2),
		m_fcn(fcn){
		for (uint64_t i=0;i<input1.size();++i){
		    //std::cout << "fcn("<<input1[i]<<", "<< input2[i] <<")="<< fcn(input1[i], input2[i])<<"\n";
			pushOutputVal( fcn(input1[i], input2[i]) );
		}
	}
	//-----------------------------------------------------------------------------

	/// Construct from name, input and vector function
	fcnResponse2D(const std::string& fcnName,
			const std::vector<T>& input1,
			const std::vector<T>& input2,
			const std::function<void (const uint32_t, T*, T*, T*)>& fcnv):
		IfcnResponse<T>(fcnName,input1,input2),
		m_fcnv(fcnv){
		//m_output.reserve(size);
		const uint64_t size = this->getInput1().size();
		T* input1_arr = const_cast<T*> (&this->getInput1()[0]);
		T* input2_arr = const_cast<T*> (&this->getInput2()[0]);
        T* output_arr = new T[size];
		fcnv(size,input1_arr,input2_arr,output_arr);
		for (uint32_t i=0;i<size;++i )
		  pushOutputVal(output_arr[i]);
		delete [] output_arr;
	}
	//-----------------------------------------------------------------------------

	/// Construct from ascii file
	fcnResponse2D(const std::string& input_filename):
		IfcnResponse<T>(input_filename){
		std::ifstream ifile ( input_filename );
		std::string line;
		//skip the 5 header lines
		for (uint16_t i=0;i<5;++i)
			std::getline(ifile,line);
		//read data from file
		fpFromHex<T> in_val1, in_val2, out_val;
		while(ifile >> in_val1 >> in_val2 >> out_val) {
			pushInput1Val(in_val1.getValue());
			pushInput2Val(in_val2.getValue());
			pushOutputVal(out_val.getValue());
		}
	}
	//-----------------------------------------------------------------------------

	~fcnResponse2D(){};
    //-----------------------------------------------------------------------------

	/// Return the input
    inline std::vector<T>& getInput1() {return IfcnResponse<T>::getInput1();};
    void pushInput1Val(T val) {IfcnResponse<T>::pushInput1Val(val);};
    inline const T input1Val(uint64_t index) {return IfcnResponse<T>::input1Val (index);};

    inline std::vector<T>& getInput2() {return IfcnResponse<T>::getInput2();};
    void pushInput2Val(T val) {IfcnResponse<T>::pushInput2Val(val);};
    inline const T input2Val(uint64_t index) {return IfcnResponse<T>::input2Val (index);};
    //-----------------------------------------------------------------------------

	/// Return the input
    inline std::vector<T>& getOutput() {return IfcnResponse<T>::getOutput();};
    void pushOutputVal(T val) {IfcnResponse<T>::pushOutputVal(val);};
    inline const T outputVal(uint64_t index) {return IfcnResponse<T>::outputVal(index);};
	//-----------------------------------------------------------------------------

	/// Dump on ascii file
	void writeFile(const std::string& output_filename) {
		const std::string preamble("VDT function arithmetics performance 1 input file (the first 5 lines are the header)\n");
		std::ofstream ofile ( output_filename );
		// Copy the input file if the object was created from file
		if (IfcnResponse<T>::isFromFile()){
			std::string line;
			std::ifstream ifile ( IfcnResponse<T>::getIfileName() );
			getline(ifile,line);
			ofile << "Dumped by an object initialised by " << IfcnResponse<T>::getIfileName() << " - "
					<< preamble;
			ofile << ifile.rdbuf() ;
		}
		else{ // Write an header and the numbers in the other case
			ofile << preamble;
			if (sizeof(T)==8) // some kind of RTTC
				ofile << "Double Precision\n";
			else
				ofile << "Single Precision\n";
			ofile << "Function Name = " <<  IfcnResponse<T>::getFcnName() << std::endl
				  << "--\n--\n";
			// Do not write dec, but HEX!
			for (uint32_t i=0;i<this->getInput1().size();++i)
				ofile << fpToHex<T>(input1Val(i)) << fpToHex<T>(input2Val(i))
				<< fpToHex<T>(outputVal(i)) << std::endl;
		}
	}

	//-----------------------------------------------------------------------------

	/// Print to screen
	void print() {
		const uint64_t size=getInput1().size();
		std::cout << "Function Performance (2 inputs):\n";
		std::cout << std::setprecision(std::numeric_limits<T>::digits10);
		for (uint64_t i=0;i<size;++i)
			std::cout << i << "/" << size << " " << IfcnResponse<T>::getFcnName()
					<< "(" << input1Val(i) << ", "<<  input2Val(i) <<" ) = " << outputVal(i)
					<< std::endl;

	}
    //-----------------------------------------------------------------------------
private:
	const std::function<T(T,T)> m_fcn;
	const std::function<void(const uint32_t, T*, T*, T*)> m_fcnv;
};




template<class T>
class fcnResponse_old{
public:
	
	//-----------------------------------------------------------------------------
	
	/// Construct from name, input and scalar function
	fcnResponse_old(const std::string& fcnName, const std::vector<T>& input, const std::function<T(T)>& fcn):
		m_from_file(false),
		m_fcn_name(fcnName),
		m_ifile_name("From Scratch"),
		m_fcn(fcn),
		m_input(input){
		
		m_output.reserve(m_input.size());
		for (auto& inp : m_input )
			m_output.push_back(fcn(inp));
	}
	
	//-----------------------------------------------------------------------------
	
	/// Construct from name, input and vector function
	fcnResponse_old(const std::string& fcnName, const std::vector<T>& input, const std::function<void (uint32_t, T*, T*)>& fcnv):
		m_from_file(false),
		m_fcn_name(fcnName),
		m_ifile_name("From Scratch"),
		m_fcnv(fcnv),
		m_input(input){
		const uint32_t size=m_input.size();
		m_output.reserve(size);
                T* input_arr = const_cast<T*> (&m_input[0]);
		T* output_arr = new T[size];
		fcnv(size,input_arr,output_arr);
		for (uint32_t i=0;i<size;++i )
			m_output.push_back(output_arr[i]);
		delete [] output_arr;
	}
	//-----------------------------------------------------------------------------
		
	/// Construct from ascii file
	fcnResponse_old(const std::string& input_filename):
		m_from_file(true),
		m_fcn_name(std::string("From ")+input_filename),
		m_ifile_name(input_filename){
		
			std::ifstream ifile ( input_filename );
			std::string line;
			//skip the 5 header lines
			for (uint16_t i=0;i<5;++i)
				std::getline(ifile,line);
			//read data from file	
			fpFromHex<T> in_val, out_val;
			while(ifile >> in_val >> out_val) {
				m_input.push_back(in_val.getValue());
				m_output.push_back(out_val.getValue());
			}
	}
	
	//-----------------------------------------------------------------------------
	
	/// Nothing to do
	~fcnResponse_old(){};
	
    //-----------------------------------------------------------------------------
        
    /// Return the output
    const std::vector<T>& getOutput() const {return m_output;};

	/// Return the input
    const std::vector<T>& getInput() const {return m_input;};
        
	//-----------------------------------------------------------------------------
	
	/// Dump on ascii file
	void writeFile(const std::string& output_filename){
		const std::string preamble("VDT function arithmetics performance file (the first 5 lines are the header)\n");
		std::ofstream ofile ( output_filename );
		// Copy the input file if the object was created from file
		if (m_from_file){
			std::string line;
			std::ifstream ifile ( m_ifile_name );
			getline(ifile,line);
			ofile << "Dumped by an object initialised by " << m_ifile_name << " - "
					<< preamble;
			ofile << ifile.rdbuf() ;
		}
		else{ // Write an header and the numbers in the other case
			ofile << preamble;
			if (sizeof(T)==8) // some kind of RTTC
				ofile << "Double Precision\n";
			else
				ofile << "Single Precision\n";
			ofile << "Function Name = " << m_fcn_name << std::endl
					<< "--\n--\n";
			//ofile << std::setprecision(std::numeric_limits<T>::digits10);
			// Do not write dec, but HEX!
			for (uint32_t i=0;i<m_input.size();++i)
				ofile << fpToHex<T>(m_input[i]) << fpToHex<T>(m_output[i]) << std::endl;
		}
	}

	//-----------------------------------------------------------------------------
	
	/// Print to screen
	void print(){
		uint32_t counter=0;
		const uint32_t size=m_input.size();
		std::cout << "Function Performance:\n";
		for (uint32_t i=0;i<size;++i){
			std::cout << std::setprecision(std::numeric_limits<T>::digits10);
			std::cout <<  counter++ << "/" << size << " " << m_fcn_name 
					<< "(" << m_input[i] << ") = " << m_output[i] << std::endl;
		}
	}
	
private:
	const bool m_from_file;
	const std::string m_fcn_name;
	const std::string m_ifile_name;
	const std::function<T(T)> m_fcn;
	const std::function<void(const uint32_t, T*, T*)> m_fcnv;
	std::vector<T> m_input;
	std::vector<T> m_output;
		
};


// For compatibility
template <typename T>
using fcnResponse = fcnResponse1D<T>;



