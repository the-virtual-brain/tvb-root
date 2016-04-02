/**
 * This file contains the routines for the creation and 
 * i/o of random numbers.
 * 
 * Author Danilo Piparo
 * 
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

#ifndef _VDT_RANDOM_
#define _VDT_RANDOM_

#include <vector>
#include <fstream>
#include "vdtcore_common.h"
#include "vdtdiag_filePersistence.h"
#include "vdtdiag_interfaces.h"

/**
 * Class that represents a floating point random numbers pool.
 * It has 2 constructors:
 * 1) specifying min and max number and the size of the collection.
 * 2) from ascii file
 * The objects are able to dump on disk in the form of ascii files with 5 lines header
 * the random numbers. The header is hardcoded!! No comments, nothing. Just 5 lines in 
 * the header. In order not to loose precision, the numbers are written in HEX.
 **/


template<typename T>
class randomPool1D:public IrandomPool{
public:
	randomPool1D(const T min, const T max, const uint64_t size, const int32_t seed=1):
		IrandomPool(size, seed),
		m_min(min),
		m_max(max){
		// Fill the numbers
		fillVector<T>(m_numbers,m_min,m_max);
	};
	//-----------------------------------------------------------------------------

	/// Initialise with an ascii file. The numbers are in hex format.
	randomPool1D(const std::string& input_filename):
		IrandomPool(input_filename),
		m_min(0),
		m_max(0){
		std::ifstream ifile ( getFileName() );
		std::string line;
		// skip the 5 header lines
		for (uint16_t i=0;i<5;++i)
			std::getline(ifile,line);
		// read from file
		fpFromHex<T> input_val;
		while(ifile >> input_val) {
			m_numbers.push_back(input_val.getValue());
		}
	}

	//-----------------------------------------------------------------------------
	~randomPool1D(){};
	//-----------------------------------------------------------------------------

	/// Write numbers on disk in the form of an ascii file. The numbers are in HEX format.
	void writeFile(const std::string& output_filename) {
		const std::string preamble("VDT Random numbers file (the first 5 lines are the header)\n");
		std::ofstream ofile ( output_filename );
		// Copy the input file if the object was created from file
		if (getSize()==0 ){
			std::string line;
			std::ifstream ifile ( getFileName() );
			getline(ifile,line);
			ofile << "Dumped by an object initialised by " << getFileName() << " - "
					<< preamble;
			ofile << ifile.rdbuf() ;
		}
		else{ // Write an header and the numbers in the other case
			ofile << preamble;
			if (sizeof(T)==8) // some kind of RTTC
				ofile << "Double Precision\n";
			else
				ofile << "Single Precision\n";
			ofile << "Minimum Number = " << m_min << std::endl
			<< "Maximum Number = " << m_max << std::endl
			<< "Total numbers = " << getSize() << std::endl;
			// Do not write dec, but HEX!
			for (uint64_t i=0;i<m_numbers.size();++i)
					ofile << fpToHex<T>(m_numbers[i]);
			ofile << std::setbase(10);
		}
	}

	//-----------------------------------------------------------------------------

	/// Return a vector of the numbers
	const std::vector<T>& getNumbers() const {return m_numbers;};

	//-----------------------------------------------------------------------------

	// Prints the random numbers on screen
	void print() {
		std::cout << "Random Pool 1D:\n";
		const uint64_t numbers_size = m_numbers.size();
		std::cout << std::setprecision(std::numeric_limits<T>::digits10);
		for(uint64_t i=0;i<numbers_size;++i)
			std::cout << i << "/" << numbers_size << " - " << m_numbers[i] << std::endl;
	}

	//-----------------------------------------------------------------------------

private:
	T m_min;
	T m_max;
	std::vector<T> m_numbers;
};


template<typename T>
class randomPool2D:public IrandomPool{
public:
	randomPool2D(const T minX, const T minY,
				const T maxX, const T maxY,
				const uint64_t size, const int32_t seed=1):
		IrandomPool(size, seed),
		m_minX(minX),
		m_minY(minY),
		m_maxX(maxX),
		m_maxY(maxY){
		// Fill the numbers
		fillVector<T>(m_numbersX,m_minX,m_maxX);
		fillVector<T>(m_numbersY,m_minY,m_maxY);
	};
	//-----------------------------------------------------------------------------

	/// Initialise with an ascii file. The numbers are in hex format.
	randomPool2D(const std::string& input_filename):
		IrandomPool(input_filename),
		m_minX(0),
		m_minY(0),
		m_maxX(0),
		m_maxY(0){
		std::ifstream ifile ( getFileName() );
		std::string line;
		// skip the 5 header lines
		for (uint16_t i=0;i<5;++i)
			std::getline(ifile,line);
		// read from file
		fpFromHex<T> input_val;
		uint64_t counter=0;
		while(ifile >> input_val) {
			if(counter%2 == 0)
			  m_numbersX.push_back(input_val.getValue());
			else
			  m_numbersY.push_back(input_val.getValue());
			counter++;
		}
	}

	//-----------------------------------------------------------------------------
	~randomPool2D(){};
	//-----------------------------------------------------------------------------

	/// Write numbers on disk in the form of an ascii file. The numbers are in HEX format.
	void writeFile(const std::string& output_filename) {
		const std::string preamble("VDT Random numbers file 2D (the first 5 lines are the header)\n");
		std::ofstream ofile ( output_filename );
		// Copy the input file if the object was created from file
		if (getSize()==0 ){
			std::string line;
			std::ifstream ifile ( getFileName() );
			getline(ifile,line);
			ofile << "Dumped by an object initialised by " << getFileName() << " - "
				  << preamble;
			ofile << ifile.rdbuf() ;
		}
		else{ // Write an header and the numbers in the other case
			ofile << preamble;
			if (sizeof(T)==8) // some kind of RTTC
				ofile << "Double Precision\n";
			else
				ofile << "Single Precision\n";
			ofile << "Minimum Number = " << m_minX << " " << m_minY << std::endl
			<< "Maximum Number = " << m_maxX << " " << m_maxY << std::endl
			<< "Total numbers = " << getSize() << std::endl;
			// Do not write dec, but HEX!
			for (uint64_t i=0;i<m_numbersX.size()*2;++i){
				uint64_t j=i/2;
				if (i%2 ==0)
					ofile << fpToHex<T>(m_numbersX[j]);
				else
					ofile << fpToHex<T>(m_numbersY[j]);
				}
			ofile << std::setbase(10);
		}
	}

	//-----------------------------------------------------------------------------

	/// Return a vector of the numbers
	const std::vector<T>& getNumbersX() const {return m_numbersX;};
	/// Return a vector of the numbers
	const std::vector<T>& getNumbersY() const {return m_numbersY;};
	//-----------------------------------------------------------------------------

	// Prints the random numbers on screen
	void print() {
		std::cout << "Random Pool 2D:\n";
		const uint64_t numbers_size = m_numbersX.size();
		std::cout << std::setprecision(std::numeric_limits<T>::digits10);
		for(uint64_t i=0;i<numbers_size;++i)
			std::cout << i << "/" << numbers_size << " - "
			<< m_numbersX[i] << " " << m_numbersY[i] << std::endl;
	}

	//-----------------------------------------------------------------------------

private:
	T m_minX;
	T m_minY;
	T m_maxX;
	T m_maxY;
	std::vector<T> m_numbersX;
	std::vector<T> m_numbersY;
};


// Old implementation, deprecated
template<class T>
class randomPool_old{

public:
	/// Initialise with min and max numbers as well as desired size of the pool.
	randomPool_old(const T min, const T max, const uint64_t size, const uint32_t seed=1):
		m_min(min), 
		m_max(max), 
		m_size(size){

		// Allocate the engine with seed one. Always the same numbers.
		std::mt19937_64 mtwister_engine(seed); 

		// allocate the distribution
		// use C++11 long double to be able to generate whole double range
		// This generates pure uniform distribution, which may be not suitable for
		// longer ranges
		std::uniform_real_distribution<long double> uniform_dist(m_min, m_max);

		auto uniform_gen = std::bind(uniform_dist, mtwister_engine);
		
		// Fill the numbers
		m_numbers.reserve(size);
		for (uint64_t i = 0; i < m_size; ++i)
			m_numbers.push_back((T)uniform_gen());
	}

	//-----------------------------------------------------------------------------

	/// Initialise with an ascii file. The numbers are in hex format.
	randomPool_old(const std::string& input_filename){
		m_ifile_name=input_filename;
		std::ifstream ifile ( input_filename );
		std::string line;
		// skip the 5 header lines
		for (uint16_t i=0;i<5;++i)
			std::getline(ifile,line);
		// read from file
		fpFromHex<T> input_val;
		while(ifile >> input_val) {
			m_numbers.push_back(input_val.getValue());
		}
		// particular of file init!!!
		m_min=m_max=m_size=0;
	}

	//-----------------------------------------------------------------------------

	/// Really nothing to do
	~randomPool_old(){}

	//-----------------------------------------------------------------------------

	/// Write numbers on disk in the form of an ascii file. The numbers are in HEX format.
	void writeFile(const std::string output_filename) const{
		const std::string preamble("VDT Random numbers file (the first 5 lines are the header)\n");
		std::ofstream ofile ( output_filename );
		// Copy the input file if the object was created from file
		if (m_size==0 ){
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
			ofile << "Minimum Number = " << m_min << std::endl
			<< "Maximum Number = " << m_max << std::endl
			<< "Total numbers = " << m_size << std::endl;
			// Do not write dec, but HEX!
			for (uint64_t i=0;i<m_numbers.size();++i)
					ofile << fpToHex<T>(m_numbers[i]);
			ofile << std::setbase(10);
		}
	}

	//-----------------------------------------------------------------------------

	/// Return a vector of the numbers
	const std::vector<T>& getNumbers() const {return m_numbers;};

	//-----------------------------------------------------------------------------
	
	// Prints the random numbers on screen
	void print() const{
		uint64_t counter=1;
		std::cout << "Random Pool:\n";
		const uint64_t numbers_size = m_numbers.size();
		std::cout << std::setprecision(std::numeric_limits<T>::digits10);
		for(uint64_t i=0;i<numbers_size;++i)
			std::cout << counter++ << "/" << numbers_size << " - " << m_numbers[i] << std::endl;
	}

	//-----------------------------------------------------------------------------

private:
	T m_min;
	T m_max;
	uint64_t m_size;
	std::string m_ifile_name;
	std::vector<T> m_numbers;
};

// For compatibility
//using  randomPool = randomPool_old;
template <typename T>
using randomPool = randomPool1D<T>;

#endif
