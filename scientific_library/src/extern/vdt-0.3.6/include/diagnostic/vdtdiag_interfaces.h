/**
 * This file contains the abstract interfaces for the diagnostic classes
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

#include <random>
#include <string>
#include <vector>
#include <functional>
#include <iomanip>
#include "assert.h"
#include "vdtdiag_helper.h"

#ifndef _VDT_INTERFACES_
#define _VDT_INTERFACES_

/**
 *  Abstract interface for classes that can be printed on screen and written from a file.
 **/
class Iprintable{
public:
    Iprintable(){};
    virtual ~Iprintable(){};
	  virtual void writeFile(const std::string& output_filename) =0;
	  virtual void print() =0;
};

//------------------------------------------------------------------------------

template<typename T>
class IfcnComparison:Iprintable{
	using vectT=std::vector<T>;
public:
//------------------------------------------------------------------------------
	// Ctor from 2 outputs
	IfcnComparison(const std::string& name,
				   const vectT& out1,
			       const vectT& out2):
		m_from_file(false),
		m_ifile_name("From scratch"),
		m_name(name),
		m_out1(out1),
		m_out2(out2){
	    // A basic consistency check
	    assert(out2.size()==out1.size());

	    // Calculate the differences in bit
	    m_fillBitDiffs();

	    // Calculate the stats
	    m_calcStats();
	};
//------------------------------------------------------------------------------
	/// Ctor from file
	IfcnComparison(const std::string& input_filename):
		m_from_file(true),
		m_ifile_name(input_filename),
		m_name(std::string("From ")+input_filename){}
//------------------------------------------------------------------------------
	~IfcnComparison(){};
//------------------------------------------------------------------------------

	void printStats(const bool tabular = false){
		if(!tabular){
			std::cout << "Stats for " << m_name << ":\n"
	              << std::setprecision(2)
	              << "Max diff bit: " << m_max << "\n"
	              << "Min diff bit: " << m_min << "\n"
	              << "Mean diff bit: " << m_mean << "\n"
	              << "RMS diff bit: " << m_RMS << "\n";
		}else{
			std::cout << std::setw(30) << std::left << m_name
				<< std::setw(5) << std::right << m_max
				<< std::setw(5) << std::right << m_min
				<< std::setiosflags(std::ios::fixed)
				<< std::setw(7) << std::right << std::setprecision(2) << m_mean
				<< std::setw(7) << std::right << std::setprecision(2) << m_RMS << "\n";
		}
	  };
//------------------------------------------------------------------------------

// Handy functions:
inline bool hasDifference(){return (m_max > 0);}

protected:
	const bool m_from_file;
	const std::string m_ifile_name;
	const std::string m_name;
	vectT m_out1;
	vectT m_out2;
	std::vector<uint16_t> m_diff_bitv;
	double m_mean = 0;
	double m_RMS = 0;
	uint16_t m_min=255;
	uint16_t m_max=0;

private:

	/// Fill the vector of different bits
	void m_fillBitDiffs(){
	  const uint32_t size = m_out1.size();
	  m_diff_bitv.reserve(size);
	  for (uint32_t i=0;i<size;++i)
		m_diff_bitv.push_back(vdth::diffbit(m_out1[i],m_out2[i]));
	}

	/// Calculate min,max,mean and RMS
	void m_calcStats(){
	double sum_x=0.;
	double sum_x2=0.;
	// Loop on the vectors and caculate the 2 momenta
	for (auto& bitdiff:m_diff_bitv){
	  // momenta
	  sum_x+=bitdiff;
	  sum_x2+=bitdiff*bitdiff;
	  // min and max
	  if (bitdiff<m_min)m_min=bitdiff;
	  if (bitdiff>m_max)m_max=bitdiff;
	}
	// Now the mean!
	const uint32_t size=m_out1.size();
	m_mean=sum_x/size;
	if (size==1)
	  m_RMS=-1;
	else
	  m_RMS=(sum_x2 - size*m_mean*m_mean)/(size-1);

	}

};


template<typename T>
class IfcnResponse:Iprintable{
public:
	IfcnResponse(const std::string& fcnName, std::vector<T> input):
		m_input1(input),
		m_from_file(false),
		m_fcn_name(fcnName),
		m_ifile_name("From scratch"){m_output.reserve(input.size());};
//-----------------------------------------------------------------------------
	IfcnResponse(const std::string& fcnName, std::vector<T> input1, std::vector<T> input2):
		m_input1(input1),
		m_input2(input2),
		m_from_file(false),
		m_fcn_name(fcnName),
		m_ifile_name("From scratch"){m_output.reserve(input1.size());};
//-----------------------------------------------------------------------------
	/// Construct from ascii file
	IfcnResponse(const std::string& input_filename):
		m_from_file(true),
		m_fcn_name(std::string("From ")+input_filename),
		m_ifile_name(input_filename){};
//-----------------------------------------------------------------------------
	~IfcnResponse(){};
//-----------------------------------------------------------------------------
	const std::string& getFcnName() const {return m_fcn_name;}
//-----------------------------------------------------------------------------
	const std::string& getIfileName() const {return m_ifile_name;}
//-----------------------------------------------------------------------------
	bool isFromFile() const {return m_from_file;}
//-----------------------------------------------------------------------------
   inline std::vector<T>& getOutput()  {return m_output;};
   void pushOutputVal(T value) {m_output.push_back(value);};
   inline const T outputVal(uint64_t index) const {return m_output[index];};
//-----------------------------------------------------------------------------
   inline std::vector<T>& getInput1()  {return m_input1;};
   void pushInput1Val(T value) {m_input1.push_back(value);};
   inline const T input1Val(uint64_t index) const {return m_input1[index];};
//-----------------------------------------------------------------------------
   inline std::vector<T>& getInput2()  {return m_input2;};
   void pushInput2Val(T value) {m_input2.push_back(value);};
   inline const T input2Val(uint64_t index) const {return m_input2[index];};
private:
	std::vector<T> m_output;
    std::vector<T> m_input1;
	std::vector<T> m_input2;
	const bool m_from_file;
	const std::string m_fcn_name;
	const std::string m_ifile_name;
};


class IrandomPool:public Iprintable{
public:
	IrandomPool(const uint64_t size, const int32_t seed=1):
		m_size(size),
		m_mtwister_engine(seed),
		m_ifile_name(""){};
	IrandomPool(const std::string& input_filename):
		m_size(0),
		m_mtwister_engine(0),
		m_ifile_name(input_filename){};
	~IrandomPool(){};
	uint64_t getSize() const {return m_size;};
	const std::string& getFileName() const {return m_ifile_name;};
protected:
	template <typename T> void fillVector(std::vector<T>& randomv,T min, T max){
		// allocate the distribution
		// use C++11 long double to be able to generate whole double range
		// This generates pure uniform distribution, which may be not suitable for
		// longer ranges
		std::uniform_real_distribution<long double> uniform_dist(min, max);
		// Fill the numbers
		randomv.reserve(m_size);
		for (uint64_t i = 0; i < m_size; ++i){
			T temp = (T)uniform_dist(m_mtwister_engine);
			//std::cout << "Generated num = " << temp << "\n";
			randomv.push_back(temp);
			}
		std::cout << "\n";
		}
private:
	const uint64_t m_size;
	std::mt19937_64 m_mtwister_engine;
	const std::string m_ifile_name;
};


#endif
